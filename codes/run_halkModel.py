#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import ctypes
#ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
from torch.utils.data import DataLoader

from halkModel import HaLk
from dataloader_negation import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='HaLkCone', type=str)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--task', default='1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc', type=str)
    parser.add_argument('--stepsforpath', type=int, default=0)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gamma2', default=0, type=float)
    parser.add_argument('--center_reg', default=0.0, type=float, help='alpha in the paper')
    parser.add_argument('--activation', default='relu', type=str, help='relu or none or softplus')

    parser.add_argument('--drop', type=float, default=0., help='dropout rate')
    parser.add_argument('--test_steps', default=50000, type=int)
    parser.add_argument('--evaluate_train_data', action='store_true')
    parser.add_argument('--evaluate_train_steps', default=50000, type=int)
    parser.add_argument('--negation_structures', action='store_true', help='true for generating negation structures, false for other structures')

    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args, before_finetune=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json' if not before_finetune else 'config_before.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint' if not before_finetune else 'checkpoint_before')
    )

    cone_entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'cone_entity_embedding' if not before_finetune else 'cone_entity_embedding_before'),
        cone_entity_embedding
    )

    axis_embedding = model.relation_center_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'axis_embedding' if not before_finetune else 'axis_embedding_before'),
        axis_embedding
    )

    shift_embedding = model.shift_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'shift_embedding' if not before_finetune else 'shift_embedding_before'),
        shift_embedding
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def configure_optimizers(model, current_learning_rate1, current_learning_rate2):
    param_frozen_list = []
    param_active_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'group_adj_weight_multi':
                param_frozen_list.append(param)
            else:
                param_active_list.append(param)

    grouped_parameters = [
        {"params": param_active_list, 'lr': current_learning_rate1},
        {"params": param_frozen_list, 'lr': current_learning_rate2},
    ]

    return grouped_parameters

def main(args):
    set_global_seed(args.seed)
    args.test_batch_size = 1
    assert args.max_steps == args.stepsforpath

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train_data):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    args.save_path = 'logs/%s/cone/'%(args.data_path.split('/')[-1])
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('#stepsforpath: %d' % args.stepsforpath)
    logging.info('#negative-sample-size: %d' % args.negative_sample_size)

    if args.negation_structures:
        args.task = '1c.2c.3c.2i.3i.2d.3d.2in.3in.cin.cni.inc'
    else:
        args.task = '1c.3c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc'
        
    tasks = args.task.split('.')

    train_ans = dict()
    valid_ans = dict()
    valid_ans_hard = dict()
    test_ans = dict()
    test_ans_hard = dict()

    if '1c' in tasks:
        with open('%s/train_triples_1c.pkl'%args.data_path, 'rb') as handle:
            train_triples = pickle.load(handle)
        with open('%s/train_ans_1c.pkl'%args.data_path, 'rb') as handle:
            train_ans_1 = pickle.load(handle)
        with open('%s/valid_triples_1c.pkl'%args.data_path, 'rb') as handle:
            valid_triples = pickle.load(handle)
        with open('%s/valid_ans_1c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1 = pickle.load(handle)
        with open('%s/valid_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1_hard = pickle.load(handle)
        with open('%s/test_triples_1c.pkl'%args.data_path, 'rb') as handle:
            test_triples = pickle.load(handle)
        with open('%s/test_ans_1c.pkl'%args.data_path, 'rb') as handle:
            test_ans_1 = pickle.load(handle)
        with open('%s/test_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_1_hard = pickle.load(handle)
        train_ans.update(train_ans_1)
        valid_ans.update(valid_ans_1)
        valid_ans_hard.update(valid_ans_1_hard)
        test_ans.update(test_ans_1)
        test_ans_hard.update(test_ans_1_hard)

    if '2c' in tasks:
        with open('%s/train_triples_2c.pkl'%args.data_path, 'rb') as handle:
            train_triples_2 = pickle.load(handle)
        with open('%s/train_ans_2c.pkl'%args.data_path, 'rb') as handle:
            train_ans_2 = pickle.load(handle)
        with open('%s/valid_triples_2c.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2 = pickle.load(handle)
        with open('%s/valid_ans_2c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2 = pickle.load(handle)
        with open('%s/valid_ans_2c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2_hard = pickle.load(handle)
        with open('%s/test_triples_2c.pkl'%args.data_path, 'rb') as handle:
            test_triples_2 = pickle.load(handle)
        with open('%s/test_ans_2c.pkl'%args.data_path, 'rb') as handle:
            test_ans_2 = pickle.load(handle)
        with open('%s/test_ans_2c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2_hard = pickle.load(handle)
        train_ans.update(train_ans_2)
        valid_ans.update(valid_ans_2)
        valid_ans_hard.update(valid_ans_2_hard)
        test_ans.update(test_ans_2)
        test_ans_hard.update(test_ans_2_hard)

    if '3c' in tasks:
        with open('%s/train_triples_3c.pkl'%args.data_path, 'rb') as handle:
            train_triples_3 = pickle.load(handle)
        with open('%s/train_ans_3c.pkl'%args.data_path, 'rb') as handle:
            train_ans_3 = pickle.load(handle)
        with open('%s/valid_triples_3c.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3 = pickle.load(handle)
        with open('%s/valid_ans_3c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3 = pickle.load(handle)
        with open('%s/valid_ans_3c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3_hard = pickle.load(handle)
        with open('%s/test_triples_3c.pkl'%args.data_path, 'rb') as handle:
            test_triples_3 = pickle.load(handle)
        with open('%s/test_ans_3c.pkl'%args.data_path, 'rb') as handle:
            test_ans_3 = pickle.load(handle)
        with open('%s/test_ans_3c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3_hard = pickle.load(handle)
        train_ans.update(train_ans_3)
        valid_ans.update(valid_ans_3)
        valid_ans_hard.update(valid_ans_3_hard)
        test_ans.update(test_ans_3)
        test_ans_hard.update(test_ans_3_hard)

    if '2i' in tasks:
        with open('%s/train_triples_2i.pkl'%args.data_path, 'rb') as handle:
            train_triples_2i = pickle.load(handle)
        with open('%s/train_ans_2i.pkl'%args.data_path, 'rb') as handle:
            train_ans_2i = pickle.load(handle)
        with open('%s/valid_triples_2i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2i = pickle.load(handle)
        with open('%s/valid_ans_2i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i = pickle.load(handle)
        with open('%s/valid_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i_hard = pickle.load(handle)
        with open('%s/test_triples_2i.pkl'%args.data_path, 'rb') as handle:
            test_triples_2i = pickle.load(handle)
        with open('%s/test_ans_2i.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i = pickle.load(handle)
        with open('%s/test_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i_hard = pickle.load(handle)
        train_ans.update(train_ans_2i)
        valid_ans.update(valid_ans_2i)
        valid_ans_hard.update(valid_ans_2i_hard)
        test_ans.update(test_ans_2i)
        test_ans_hard.update(test_ans_2i_hard)

    if '3i' in tasks:
        with open('%s/train_triples_3i.pkl'%args.data_path, 'rb') as handle:
            train_triples_3i = pickle.load(handle)
        with open('%s/train_ans_3i.pkl'%args.data_path, 'rb') as handle:
            train_ans_3i = pickle.load(handle)
        with open('%s/valid_triples_3i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3i = pickle.load(handle)
        with open('%s/valid_ans_3i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i = pickle.load(handle)
        with open('%s/valid_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i_hard = pickle.load(handle)
        with open('%s/test_triples_3i.pkl'%args.data_path, 'rb') as handle:
            test_triples_3i = pickle.load(handle)
        with open('%s/test_ans_3i.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i = pickle.load(handle)
        with open('%s/test_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i_hard = pickle.load(handle)
        train_ans.update(train_ans_3i)
        valid_ans.update(valid_ans_3i)
        valid_ans_hard.update(valid_ans_3i_hard)
        test_ans.update(test_ans_3i)
        test_ans_hard.update(test_ans_3i_hard)
    
    if 'ic' in tasks:
        with open('%s/valid_triples_ic.pkl'%args.data_path, 'rb') as handle:
            valid_triples_ic = pickle.load(handle)
        with open('%s/valid_ans_ic.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ic = pickle.load(handle)
        with open('%s/valid_ans_ic_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ic_hard = pickle.load(handle)
        with open('%s/test_triples_ic.pkl'%args.data_path, 'rb') as handle:
            test_triples_ic = pickle.load(handle)
        with open('%s/test_ans_ic.pkl'%args.data_path, 'rb') as handle:
            test_ans_ic = pickle.load(handle)
        with open('%s/test_ans_ic_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_ic_hard = pickle.load(handle)
        valid_ans.update(valid_ans_ic)
        valid_ans_hard.update(valid_ans_ic_hard)
        test_ans.update(test_ans_ic)
        test_ans_hard.update(test_ans_ic_hard)

    if 'ci' in tasks:
        with open('%s/valid_triples_ci.pkl'%args.data_path, 'rb') as handle:
            valid_triples_ci = pickle.load(handle)
        with open('%s/valid_ans_ci.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ci = pickle.load(handle)
        with open('%s/valid_ans_ci_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ci_hard = pickle.load(handle)
        with open('%s/test_triples_ci.pkl'%args.data_path, 'rb') as handle:
            test_triples_ci = pickle.load(handle)
        with open('%s/test_ans_ci.pkl'%args.data_path, 'rb') as handle:
            test_ans_ci = pickle.load(handle)
        with open('%s/test_ans_ci_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_ci_hard = pickle.load(handle)
        valid_ans.update(valid_ans_ci)
        valid_ans_hard.update(valid_ans_ci_hard)
        test_ans.update(test_ans_ci)
        test_ans_hard.update(test_ans_ci_hard)
        
    if '2u' in tasks:
        with open('%s/valid_triples_2u.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2u = pickle.load(handle)
        with open('%s/valid_ans_2u.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2u = pickle.load(handle)
        with open('%s/valid_ans_2u_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2u_hard = pickle.load(handle)
        with open('%s/test_triples_2u.pkl'%args.data_path, 'rb') as handle:
            test_triples_2u = pickle.load(handle)
        with open('%s/test_ans_2u.pkl'%args.data_path, 'rb') as handle:
            test_ans_2u = pickle.load(handle)
        with open('%s/test_ans_2u_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2u_hard = pickle.load(handle)
        valid_ans.update(valid_ans_2u)
        valid_ans_hard.update(valid_ans_2u_hard)
        test_ans.update(test_ans_2u)
        test_ans_hard.update(test_ans_2u_hard)

    if 'uc' in tasks:
        with open('%s/valid_triples_uc.pkl'%args.data_path, 'rb') as handle:
            valid_triples_uc = pickle.load(handle)
        with open('%s/valid_ans_uc.pkl'%args.data_path, 'rb') as handle:
            valid_ans_uc = pickle.load(handle)
        with open('%s/valid_ans_uc_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_uc_hard = pickle.load(handle)
        with open('%s/test_triples_uc.pkl'%args.data_path, 'rb') as handle:
            test_triples_uc = pickle.load(handle)
        with open('%s/test_ans_uc.pkl'%args.data_path, 'rb') as handle:
            test_ans_uc = pickle.load(handle)
        with open('%s/test_ans_uc_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_uc_hard = pickle.load(handle)
        valid_ans.update(valid_ans_uc)
        valid_ans_hard.update(valid_ans_uc_hard)
        test_ans.update(test_ans_uc)
        test_ans_hard.update(test_ans_uc_hard)

    if '2d' in tasks:
        with open('%s/train_triples_2d.pkl'%args.data_path, 'rb') as handle:
            train_triples_2d = pickle.load(handle)
        with open('%s/train_ans_2d.pkl'%args.data_path, 'rb') as handle:
            train_ans_2d = pickle.load(handle)
        with open('%s/valid_triples_2d.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2d = pickle.load(handle)
        with open('%s/valid_ans_2d.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2d = pickle.load(handle)
        with open('%s/valid_ans_2d_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2d_hard = pickle.load(handle)
        with open('%s/test_triples_2d.pkl'%args.data_path, 'rb') as handle:
            test_triples_2d = pickle.load(handle)
        with open('%s/test_ans_2d.pkl'%args.data_path, 'rb') as handle:
            test_ans_2d = pickle.load(handle)
        with open('%s/test_ans_2d_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2d_hard = pickle.load(handle)
        train_ans.update(train_ans_2d)
        valid_ans.update(valid_ans_2d)
        valid_ans_hard.update(valid_ans_2d_hard)
        test_ans.update(test_ans_2d)
        test_ans_hard.update(test_ans_2d_hard)

    if '3d' in tasks:
        with open('%s/train_triples_3d.pkl'%args.data_path, 'rb') as handle:
            train_triples_3d = pickle.load(handle)
        with open('%s/train_ans_3d.pkl'%args.data_path, 'rb') as handle:
            train_ans_3d = pickle.load(handle)
        with open('%s/valid_triples_3d.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3d = pickle.load(handle)
        with open('%s/valid_ans_3d.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3d = pickle.load(handle)
        with open('%s/valid_ans_3d_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3d_hard = pickle.load(handle)
        with open('%s/test_triples_3d.pkl'%args.data_path, 'rb') as handle:
            test_triples_3d = pickle.load(handle)
        with open('%s/test_ans_3d.pkl'%args.data_path, 'rb') as handle:
            test_ans_3d = pickle.load(handle)
        with open('%s/test_ans_3d_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3d_hard = pickle.load(handle)
        train_ans.update(train_ans_3d)
        valid_ans.update(valid_ans_3d)
        valid_ans_hard.update(valid_ans_3d_hard)
        test_ans.update(test_ans_3d)
        test_ans_hard.update(test_ans_3d_hard)

    if 'dc' in tasks:
        with open('%s/valid_triples_dc.pkl'%args.data_path, 'rb') as handle:
            valid_triples_dc = pickle.load(handle)
        with open('%s/valid_ans_dc.pkl'%args.data_path, 'rb') as handle:
            valid_ans_dc = pickle.load(handle)
        with open('%s/valid_ans_dc_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_dc_hard = pickle.load(handle)
        with open('%s/test_triples_dc.pkl'%args.data_path, 'rb') as handle:
            test_triples_dc = pickle.load(handle)
        with open('%s/test_ans_dc.pkl'%args.data_path, 'rb') as handle:
            test_ans_dc = pickle.load(handle)
        with open('%s/test_ans_dc_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_dc_hard = pickle.load(handle)
        valid_ans.update(valid_ans_dc)
        valid_ans_hard.update(valid_ans_dc_hard)
        test_ans.update(test_ans_dc)
        test_ans_hard.update(test_ans_dc_hard)

    if '2in' in tasks:
        with open('%s/train_triples_2in.pkl'%args.data_path, 'rb') as handle:
            train_triples_2in = pickle.load(handle)
        with open('%s/train_ans_2in.pkl'%args.data_path, 'rb') as handle:
            train_ans_2in = pickle.load(handle)
        with open('%s/valid_triples_2in.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2in = pickle.load(handle)
        with open('%s/valid_ans_2in.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2in = pickle.load(handle)
        with open('%s/valid_ans_2in_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2in_hard = pickle.load(handle)
        with open('%s/test_triples_2in.pkl'%args.data_path, 'rb') as handle:
            test_triples_2in = pickle.load(handle)
        with open('%s/test_ans_2in.pkl'%args.data_path, 'rb') as handle:
            test_ans_2in = pickle.load(handle)
        with open('%s/test_ans_2in_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2in_hard = pickle.load(handle)
        train_ans.update(train_ans_2in)
        valid_ans.update(valid_ans_2in)
        valid_ans_hard.update(valid_ans_2in_hard)
        test_ans.update(test_ans_2in)
        test_ans_hard.update(test_ans_2in_hard)

    if '3in' in tasks:
        with open('%s/train_triples_3in.pkl'%args.data_path, 'rb') as handle:
            train_triples_3in = pickle.load(handle)
        with open('%s/train_ans_3in.pkl'%args.data_path, 'rb') as handle:
            train_ans_3in = pickle.load(handle)
        with open('%s/valid_triples_3in.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3in = pickle.load(handle)
        with open('%s/valid_ans_3in.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3in = pickle.load(handle)
        with open('%s/valid_ans_3in_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3in_hard = pickle.load(handle)
        with open('%s/test_triples_3in.pkl'%args.data_path, 'rb') as handle:
            test_triples_3in = pickle.load(handle)
        with open('%s/test_ans_3in.pkl'%args.data_path, 'rb') as handle:
            test_ans_3in = pickle.load(handle)
        with open('%s/test_ans_3in_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3in_hard = pickle.load(handle)
        train_ans.update(train_ans_3in)
        valid_ans.update(valid_ans_3in)
        valid_ans_hard.update(valid_ans_3in_hard)
        test_ans.update(test_ans_3in)
        test_ans_hard.update(test_ans_3in_hard)

    if 'cin' in tasks:
        with open('%s/train_triples_cin.pkl'%args.data_path, 'rb') as handle:
            train_triples_cin = pickle.load(handle)
        with open('%s/train_ans_cin.pkl'%args.data_path, 'rb') as handle:
            train_ans_cin = pickle.load(handle)
        with open('%s/valid_triples_cin.pkl'%args.data_path, 'rb') as handle:
            valid_triples_cin = pickle.load(handle)
        with open('%s/valid_ans_cin.pkl'%args.data_path, 'rb') as handle:
            valid_ans_cin = pickle.load(handle)
        with open('%s/valid_ans_cin_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_cin_hard = pickle.load(handle)
        with open('%s/test_triples_cin.pkl'%args.data_path, 'rb') as handle:
            test_triples_cin = pickle.load(handle)
        with open('%s/test_ans_cin.pkl'%args.data_path, 'rb') as handle:
            test_ans_cin = pickle.load(handle)
        with open('%s/test_ans_cin_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_cin_hard = pickle.load(handle)
        train_ans.update(train_ans_cin)
        valid_ans.update(valid_ans_cin)
        valid_ans_hard.update(valid_ans_cin_hard)
        test_ans.update(test_ans_cin)
        test_ans_hard.update(test_ans_cin_hard)

    if 'cni' in tasks:
        with open('%s/train_triples_cni.pkl'%args.data_path, 'rb') as handle:
            train_triples_cni = pickle.load(handle)
        with open('%s/train_ans_cni.pkl'%args.data_path, 'rb') as handle:
            train_ans_cni = pickle.load(handle)
        with open('%s/valid_triples_cni.pkl'%args.data_path, 'rb') as handle:
            valid_triples_cni = pickle.load(handle)
        with open('%s/valid_ans_cni.pkl'%args.data_path, 'rb') as handle:
            valid_ans_cni = pickle.load(handle)
        with open('%s/valid_ans_cni_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_cni_hard = pickle.load(handle)
        with open('%s/test_triples_cni.pkl'%args.data_path, 'rb') as handle:
            test_triples_cni = pickle.load(handle)
        with open('%s/test_ans_cni.pkl'%args.data_path, 'rb') as handle:
            test_ans_cni = pickle.load(handle)
        with open('%s/test_ans_cni_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_cni_hard = pickle.load(handle)
        train_ans.update(train_ans_cni)
        valid_ans.update(valid_ans_cni)
        valid_ans_hard.update(valid_ans_cni_hard)
        test_ans.update(test_ans_cni)
        test_ans_hard.update(test_ans_cni_hard)

    if 'inc' in tasks:
        with open('%s/train_triples_inc.pkl'%args.data_path, 'rb') as handle:
            train_triples_inc = pickle.load(handle)
        with open('%s/train_ans_inc.pkl'%args.data_path, 'rb') as handle:
            train_ans_inc = pickle.load(handle)
        with open('%s/valid_triples_inc.pkl'%args.data_path, 'rb') as handle:
            valid_triples_inc = pickle.load(handle)
        with open('%s/valid_ans_inc.pkl'%args.data_path, 'rb') as handle:
            valid_ans_inc = pickle.load(handle)
        with open('%s/valid_ans_inc_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_inc_hard = pickle.load(handle)
        with open('%s/test_triples_inc.pkl'%args.data_path, 'rb') as handle:
            test_triples_inc = pickle.load(handle)
        with open('%s/test_ans_inc.pkl'%args.data_path, 'rb') as handle:
            test_ans_inc = pickle.load(handle)
        with open('%s/test_ans_inc_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_inc_hard = pickle.load(handle)
        train_ans.update(train_ans_inc)
        valid_ans.update(valid_ans_inc)
        valid_ans_hard.update(valid_ans_inc_hard)
        test_ans.update(test_ans_inc)
        test_ans_hard.update(test_ans_inc_hard)

# log info
    if '1c' in tasks:
        logging.info('#train_1c: %d' % len(train_triples))
        logging.info('#valid_1c: %d' % len(valid_triples))
        logging.info('#test_1c: %d' % len(test_triples))

    if '2c' in tasks:
        logging.info('#train_2c: %d' % len(train_triples_2))
        logging.info('#valid_2c: %d' % len(valid_triples_2))
        logging.info('#test_2c: %d' % len(test_triples_2))

    if '3c' in tasks:
        logging.info('#train_3c: %d' % len(train_triples_3))
        logging.info('#valid_3c: %d' % len(valid_triples_3))
        logging.info('#test_3c: %d' % len(test_triples_3))

    if '2i' in tasks:
        logging.info('#train_2i: %d' % len(train_triples_2i))
        logging.info('#valid_2i: %d' % len(valid_triples_2i))
        logging.info('#test_2i: %d' % len(test_triples_2i))

    if '3i' in tasks:
        logging.info('#train_3i: %d' % len(train_triples_3i))
        logging.info('#valid_3i: %d' % len(valid_triples_3i))
        logging.info('#test_3i: %d' % len(test_triples_3i))

    if 'ci' in tasks:
        logging.info('#valid_ci: %d' % len(valid_triples_ci))
        logging.info('#test_ci: %d' % len(test_triples_ci))

    if 'ic' in tasks:
        logging.info('#valid_ic: %d' % len(valid_triples_ic))
        logging.info('#test_ic: %d' % len(test_triples_ic))

    if '2u' in tasks:
        logging.info('#valid_2u: %d' % len(valid_triples_2u))
        logging.info('#test_2u: %d' % len(test_triples_2u))

    if 'uc' in tasks:
        logging.info('#valid_uc: %d' % len(valid_triples_uc))
        logging.info('#test_uc: %d' % len(test_triples_uc))

    if '2d' in tasks:
        logging.info('#train_2d: %d' % len(train_triples_2d))
        logging.info('#valid_2d: %d' % len(valid_triples_2d))
        logging.info('#test_2d: %d' % len(test_triples_2d))

    if '3d' in tasks:
        logging.info('#train_3d: %d' % len(train_triples_3d))
        logging.info('#valid_3d: %d' % len(valid_triples_3d))
        logging.info('#test_3d: %d' % len(test_triples_3d))

    if 'dc' in tasks:
        logging.info('#valid_dc: %d' % len(valid_triples_dc))
        logging.info('#test_dc: %d' % len(test_triples_dc))

    if '2in' in tasks:
        logging.info('#train_2in: %d' % len(train_triples_2in))
        logging.info('#valid_2in: %d' % len(valid_triples_2in))
        logging.info('#test_2in: %d' % len(test_triples_2in))

    if '3in' in tasks:
        logging.info('#train_3in: %d' % len(train_triples_3in))
        logging.info('#valid_3in: %d' % len(valid_triples_3in))
        logging.info('#test_3in: %d' % len(test_triples_3in))

    if 'cin' in tasks:
        logging.info('#train_cin: %d' % len(train_triples_cin))
        logging.info('#valid_cin: %d' % len(valid_triples_cin))
        logging.info('#test_cin: %d' % len(test_triples_cin))

    if 'cni' in tasks:
        logging.info('#train_cni: %d' % len(train_triples_cni))
        logging.info('#valid_cni: %d' % len(valid_triples_cni))
        logging.info('#test_cni: %d' % len(test_triples_cni))

    if 'inc' in tasks:
        logging.info('#train_inc: %d' % len(train_triples_inc))
        logging.info('#valid_inc: %d' % len(valid_triples_inc))
        logging.info('#test_inc: %d' % len(test_triples_inc))


    with open('%s/node_group_one_hot_vector.pkl' % args.data_path, 'rb') as handle:
        node_group_one_hot_vector_single = pickle.load(handle)

    with open('%s/group_adj_matrix.pkl' % args.data_path, 'rb') as handle:
        group_adj_matrix_single = pickle.load(handle)

    with open('%s/node_group_one_hot_vector_multi.pkl' % args.data_path, 'rb') as handle:
        node_group_one_hot_vector_multi = pickle.load(handle)

    with open('%s/group_adj_matrix_multi.pkl' % args.data_path, 'rb') as handle:
        group_adj_matrix_multi = pickle.load(handle)

# model init
    halk = HaLk(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        cen=args.center_reg,
        gamma2=args.gamma2,
        activation=args.activation,
        node_group_one_hot_vector_single=node_group_one_hot_vector_single,
        group_adj_matrix_single=group_adj_matrix_single,
        node_group_one_hot_vector_multi=node_group_one_hot_vector_multi,
        group_adj_matrix_multi=group_adj_matrix_multi,
        drop=args.drop
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in halk.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        halk = halk.cuda()

    if args.do_train:
        if '1c' in tasks:
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn,
                pin_memory=True
            )
            train_iterator = SingledirectionalOneShotIterator(train_dataloader_tail, train_triples[0][-1])

        if '2c' in tasks:
            train_dataloader_2_tail = DataLoader(
                TrainDataset(train_triples_2, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_2 = SingledirectionalOneShotIterator(train_dataloader_2_tail, train_triples_2[0][-1])

        if '3c' in tasks:
            train_dataloader_3_tail = DataLoader(
                TrainDataset(train_triples_3, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_3 = SingledirectionalOneShotIterator(train_dataloader_3_tail, train_triples_3[0][-1])

        if '2i' in tasks:
            train_dataloader_2i_tail = DataLoader(
                TrainInterDataset(train_triples_2i, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_2i = SingledirectionalOneShotIterator(train_dataloader_2i_tail, train_triples_2i[0][-1]) #输入的数据集和数据集的类型

        if '3i' in tasks:
            train_dataloader_3i_tail = DataLoader(
                TrainInterDataset(train_triples_3i, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_3i = SingledirectionalOneShotIterator(train_dataloader_3i_tail, train_triples_3i[0][-1])

        if '2d' in tasks:
            train_dataloader_2d_tail = DataLoader(
                TrainInterDataset(train_triples_2d, nentity, nrelation, args.negative_sample_size, train_ans,'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_2d = SingledirectionalOneShotIterator(train_dataloader_2d_tail, train_triples_2d[0][-1])

        if '3d' in tasks:
            train_dataloader_3d_tail = DataLoader(
                TrainInterDataset(train_triples_3d, nentity, nrelation, args.negative_sample_size, train_ans,
                                  'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_3d = SingledirectionalOneShotIterator(train_dataloader_3d_tail, train_triples_3d[0][-1])

        if '2in' in tasks:
            train_dataloader_2in_tail = DataLoader(
                TrainInterDataset(train_triples_2in, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_2in = SingledirectionalOneShotIterator(train_dataloader_2in_tail, train_triples_2in[0][-1])

        if '3in' in tasks:
            train_dataloader_3in_tail = DataLoader(
                TrainInterDataset(train_triples_3in, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_3in = SingledirectionalOneShotIterator(train_dataloader_3in_tail, train_triples_3in[0][-1])

        if 'cin' in tasks:
            train_dataloader_cin_tail = DataLoader(
                TrainChainInterDataset(train_triples_cin, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainChainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_cin = SingledirectionalOneShotIterator(train_dataloader_cin_tail, train_triples_cin[0][-1])

        if 'cni' in tasks:
            train_dataloader_cni_tail = DataLoader(
                TrainChainInterDataset(train_triples_cni, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainChainInterDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_cni = SingledirectionalOneShotIterator(train_dataloader_cni_tail, train_triples_cni[0][-1])

        if 'inc' in tasks:
            train_dataloader_inc_tail = DataLoader(
                TrainInterChainDataset(train_triples_inc, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterChainDataset.collate_fn,
                pin_memory=True
            )
            train_iterator_inc = SingledirectionalOneShotIterator(train_dataloader_inc_tail, train_triples_inc[0][-1])

        current_learning_rate = args.learning_rate
        x_rate = 0.001
        if "NELL" in args.data_path:
            x_rate = x_rate * 1.5

        grouped_parameters = configure_optimizers(halk, current_learning_rate, x_rate) #构造了active parameterlist和frozen parameterlist
        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    logging.info('task = %s' % args.task)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    def evaluate_test():
        if '1c' in tasks:
            metrics = halk.test_step(halk, test_triples, test_ans, test_ans_hard, args)
            log_metrics('Test 1c', step, metrics)
        if '2c' in tasks:
            metrics = halk.test_step(halk, test_triples_2, test_ans, test_ans_hard, args)
            log_metrics('Test 2c', step, metrics)
        if '3c' in tasks:
            metrics = halk.test_step(halk, test_triples_3, test_ans, test_ans_hard, args)
            log_metrics('Test 3c', step, metrics)
        if '2i' in tasks:
            metrics = halk.test_step(halk, test_triples_2i, test_ans, test_ans_hard, args)
            log_metrics('Test 2i', step, metrics)
        if '3i' in tasks:
            metrics = halk.test_step(halk, test_triples_3i, test_ans, test_ans_hard, args)
            log_metrics('Test 3i', step, metrics)
        if 'ic' in tasks:
            metrics = halk.test_step(halk, test_triples_ic, test_ans, test_ans_hard, args)
            log_metrics('Test ic', step, metrics)
        if 'ci' in tasks:
            metrics = halk.test_step(halk, test_triples_ci, test_ans, test_ans_hard, args)
            log_metrics('Test ci', step, metrics)
        if '2u' in tasks:
            metrics = halk.test_step(halk, test_triples_2u, test_ans, test_ans_hard, args)
            log_metrics('Test 2u', step, metrics)
        if 'uc' in tasks:
            metrics = halk.test_step(halk, test_triples_uc, test_ans, test_ans_hard, args)
            log_metrics('Test uc', step, metrics)
        if '2d' in tasks:
            metrics = halk.test_step(halk, test_triples_2d, test_ans, test_ans_hard, args)
            log_metrics('Test 2d', step, metrics)
        if '3d' in tasks:
            metrics = halk.test_step(halk, test_triples_3d, test_ans, test_ans_hard, args)
            log_metrics('Test 3d', step, metrics)
        if 'dc' in tasks:
            metrics = halk.test_step(halk, test_triples_dc, test_ans, test_ans_hard, args)
            log_metrics('Test dc', step, metrics)

    def evaluate_val():
        if '1c' in tasks:
            metrics = halk.test_step(halk, valid_triples, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 1c', step, metrics)
        if '2c' in tasks:
            metrics = halk.test_step(halk, valid_triples_2, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2c', step, metrics)
        if '3c' in tasks:
            metrics = halk.test_step(halk, valid_triples_3, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3c', step, metrics)
        if '2i' in tasks:
            metrics = halk.test_step(halk, valid_triples_2i, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2i', step, metrics)
        if '3i' in tasks:
            metrics = halk.test_step(halk, valid_triples_3i, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3i', step, metrics)
        if 'ic' in tasks:
            metrics = halk.test_step(halk, valid_triples_ic, valid_ans, valid_ans_hard, args)
            log_metrics('Valid ic', step, metrics)
        if 'ci' in tasks:
            metrics = halk.test_step(halk, valid_triples_ci, valid_ans, valid_ans_hard, args)
            log_metrics('Valid ci', step, metrics)
        if '2u' in tasks:
            metrics = halk.test_step(halk, valid_triples_2u, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2u', step, metrics)
        if 'uc' in tasks:
            metrics = halk.test_step(halk, valid_triples_uc, valid_ans, valid_ans_hard, args)
            log_metrics('Valid uc', step, metrics)
        if '2d' in tasks:
            metrics = halk.test_step(halk, valid_triples_2d, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2d', step, metrics)
        if '3d' in tasks:
            metrics = halk.test_step(halk, valid_triples_3d, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3d', step, metrics)
        if 'dc' in tasks:
            metrics = halk.test_step(halk, valid_triples_dc, valid_ans, valid_ans_hard, args)
            log_metrics('Valid dc', step, metrics)

    def evaluate_train():
        if '1c' in tasks:
            metrics = halk.test_step(halk, train_triples, train_ans, train_ans, args)
            log_metrics('train 1c', step, metrics)
        if '2c' in tasks:
            metrics = halk.test_step(halk, train_triples_2, train_ans, train_ans, args)
            log_metrics('train 2c', step, metrics)
        if '3c' in tasks:
            metrics = halk.test_step(halk, train_triples_3, train_ans, train_ans, args)
            log_metrics('train 3c', step, metrics)
        if '2i' in tasks:
            metrics = halk.test_step(halk, train_triples_2i, train_ans, train_ans, args)
            log_metrics('train 2i', step, metrics)
        if '3i' in tasks:
            metrics = halk.test_step(halk, train_triples_3i, train_ans, train_ans, args)
            log_metrics('train 3i', step, metrics)
        if '2d' in tasks:
            metrics = halk.test_step(halk, train_triples_2d, train_ans, train_ans, args)
            log_metrics('train 2d', step, metrics)
        if '3d' in tasks:
            metrics = halk.test_step(halk, train_triples_3d, train_ans, train_ans, args)
            log_metrics('train 3d', step, metrics)
            
    def evaluate_test_negation():
        if 'cni' in tasks:
            metrics = halk.test_step(halk, test_triples_cni, test_ans, test_ans_hard, args)
            log_metrics('Test cni', step, metrics)
        if 'inc' in tasks:
            metrics = halk.test_step(halk, test_triples_inc, test_ans, test_ans_hard, args)
            log_metrics('Test inc', step, metrics)
        if 'cin' in tasks:
            metrics = halk.test_step(halk, test_triples_cin, test_ans, test_ans_hard, args)
            log_metrics('Test cin', step, metrics)
        if '2in' in tasks:
            metrics = halk.test_step(halk, test_triples_2in, test_ans, test_ans_hard, args)
            log_metrics('Test 2in', step, metrics)
        if '3in' in tasks:
            metrics = halk.test_step(halk, test_triples_3in, test_ans, test_ans_hard, args)
            log_metrics('Test 3in', step, metrics)
    
    def evaluate_val_negation():
        if '2in' in tasks:
            metrics = halk.test_step(halk, valid_triples_2in, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2in', step, metrics)
        if '3in' in tasks:
            metrics = halk.test_step(halk, valid_triples_3in, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3in', step, metrics)
        if 'cin' in tasks:
            metrics = halk.test_step(halk, valid_triples_cin, valid_ans, valid_ans_hard, args)
            log_metrics('Valid cin', step, metrics)
        if 'cni' in tasks:
            metrics = halk.test_step(halk, valid_triples_cni, valid_ans, valid_ans_hard, args)
            log_metrics('Valid cni', step, metrics)
        if 'inc' in tasks:
            metrics = halk.test_step(halk, valid_triples_inc, valid_ans, valid_ans_hard, args)
            log_metrics('Valid inc', step, metrics)
    
    def evaluate_train_negation():
        if '2in' in tasks:
            metrics = halk.test_step(halk, train_triples_2in, train_ans, train_ans, args)
            log_metrics('train 2in', step, metrics)
        if '3in' in tasks:
            metrics = halk.test_step(halk, train_triples_3in, train_ans, train_ans, args)
            log_metrics('train 3in', step, metrics)
        if 'cin' in tasks:
            metrics = halk.test_step(halk, train_triples_cin, train_ans, train_ans, args)
            log_metrics('train cin', step, metrics)
        if 'cni' in tasks:
            metrics = halk.test_step(halk, train_triples_cni, train_ans, train_ans, args)
            log_metrics('train cni', step, metrics)
        if 'inc' in tasks:
            metrics = halk.test_step(halk, train_triples_inc, train_ans, train_ans, args)
            log_metrics('train inc', step, metrics)

    if args.do_train:
        training_logs = []
        if args.negation_structures:
            for step in range(0, args.max_steps):
                if '1c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator, args, step)
                    training_logs.append(log)
                
                if '2c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2, args, step)
                    training_logs.append(log)

                if '3c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3, args, step)
                    training_logs.append(log)
                    
                if '2i' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2i, args, step)
                    training_logs.append(log)

                if '3i' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3i, args, step)
                    training_logs.append(log)

                if '2d' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2d, args, step)
                    training_logs.append(log)

                if '3d' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3d, args, step)
                    training_logs.append(log)
                
                if '2in' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2in, args, step)
                    training_logs.append(log)

                if '3in' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3in, args, step)
                    training_logs.append(log)

                if 'cin' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_cin, args, step)
                    training_logs.append(log)

                if 'cni' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_cni, args, step)
                    training_logs.append(log)

                if 'inc' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_inc, args, step)
                    training_logs.append(log)

                if training_logs == []:
                    raise Exception("No tasks are trained!!")

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    x_rate = 0.0003
                    if "NELL" in args.data_path:
                        x_rate = x_rate * 1.5
                    grouped_parameters = configure_optimizers(halk, current_learning_rate, x_rate)
                    optimizer = torch.optim.Adam(
                        grouped_parameters,
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        if metric == 'inter_loss':
                            continue
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    inter_loss_sum = 0.
                    inter_loss_num = 0.
                    for log in training_logs:
                        if 'inter_loss' in log:
                            inter_loss_sum += log['inter_loss']
                            inter_loss_num += 1
                    if inter_loss_num != 0:
                        metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if args.do_valid and step % args.valid_steps == 0 and step != 0:
                    logging.info('Evaluating on Valid Dataset...')
                    evaluate_val_negation()

                if args.do_test and step % args.test_steps == 0 and step != 0:
                    logging.info('Evaluating on Test Dataset...')
                    evaluate_test_negation()

                if args.evaluate_train_data and step % args.evaluate_train_steps == 0 and step != 0:
                    logging.info('Evaluating on Train Dataset...')
                    evaluate_train_negation()
        else:
            for step in range(0, args.max_steps):
                if '1c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator, args, step)
                    training_logs.append(log)

                if '2c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2, args, step)
                    training_logs.append(log)

                if '3c' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3, args, step)
                    training_logs.append(log)

                if '2i' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2i, args, step)
                    training_logs.append(log)

                if '3i' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3i, args, step)
                    training_logs.append(log)

                if '2d' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_2d, args, step)
                    training_logs.append(log)

                if '3d' in tasks:
                    log = halk.train_step(halk, optimizer, train_iterator_3d, args, step)
                    training_logs.append(log)

                if training_logs == []:
                    raise Exception("No tasks are trained!!")

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    x_rate = 0.0003
                    if "NELL" in args.data_path:
                        x_rate = x_rate * 1.5
                    grouped_parameters = configure_optimizers(halk, current_learning_rate, x_rate)
                    optimizer = torch.optim.Adam(
                        grouped_parameters,
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        if metric == 'inter_loss':
                            continue
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    inter_loss_sum = 0.
                    inter_loss_num = 0.
                    for log in training_logs:
                        if 'inter_loss' in log:
                            inter_loss_sum += log['inter_loss']
                            inter_loss_num += 1
                    if inter_loss_num != 0:
                        metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if args.do_valid and step % args.valid_steps == 0 and step != 0:
                    logging.info('Evaluating on Valid Dataset...')
                    evaluate_val()

                if args.do_test and step % args.test_steps == 0 and step != 0:
                    logging.info('Evaluating on Test Dataset...')
                    evaluate_test()

                if args.evaluate_train_data and step % args.evaluate_train_steps == 0 and step != 0:
                    logging.info('Evaluating on Train Dataset...')
                    evaluate_train()

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(halk, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0

    print('Training finished!!')
    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())
