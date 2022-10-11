#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader_negation import *
import random
import pickle
import math
import time
from multi_group_util_negation import *

pi = 3.14159265358979323846

class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y

def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y

def Identity(x):
    return x

class ConeNegation(nn.Module):
    def __init__(self, axis_dim, arg_dim, hidden_dim):
        super(ConeNegation, self).__init__()
        assert axis_dim == arg_dim
        self.axis_dim = axis_dim
        self.arg_dim = arg_dim
        self.hidden_dim = hidden_dim
        self.medium_axis_layer = nn.Linear(self.axis_dim, self.hidden_dim)
        self.medium_arg_layer = nn.Linear(self.arg_dim, self.hidden_dim)
        self.post_axis_layer = nn.Linear(self.hidden_dim * 2, self.axis_dim)
        self.post_arg_layer = nn.Linear(self.hidden_dim * 2, self.arg_dim)

        nn.init.xavier_uniform_(self.medium_axis_layer.weight)
        nn.init.xavier_uniform_(self.medium_arg_layer.weight)
        nn.init.xavier_uniform_(self.post_axis_layer.weight)
        nn.init.xavier_uniform_(self.post_arg_layer.weight)

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        logit_axis = F.relu(self.medium_axis_layer(axis_embedding))
        logit_arg = F.relu(self.medium_arg_layer(arg_embedding))
        logit = torch.cat([logit_axis, logit_arg], dim=-1)

        axis_embedding = self.post_axis_layer(logit)
        arg_embedding = self.post_arg_layer(logit)
        axis_embedding = convert_to_axis(axis_embedding)
        arg_embedding = convert_to_arg(arg_embedding)

        return axis_embedding, arg_embedding

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_axis0 = nn.Linear(self.entity_dim + self.entity_dim, self.hidden_dim)
        self.layer_axis1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_axis2 = nn.Linear(self.hidden_dim, self.entity_dim)
        nn.init.xavier_uniform_(self.layer_axis0.weight)
        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        self.layer_arg0 = nn.Linear(self.entity_dim + self.entity_dim, self.hidden_dim)
        self.layer_arg1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_arg2 = nn.Linear(self.hidden_dim, self.entity_dim)
        nn.init.xavier_uniform_(self.layer_arg0.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        temp_axis = source_embedding_axis + r_embedding_axis
        relation_arg_min = temp_axis - r_embedding_arg - source_embedding_arg
        relation_arg_max = temp_axis + r_embedding_arg + source_embedding_arg

        x = torch.cat([relation_arg_min, relation_arg_max], dim=-1)
        for nl in range(0, self.num_layers):
            x = F.relu(getattr(self, "layer_axis{}".format(nl))(x))
        x = self.layer_axis2(x)
        axis_embeddings = convert_to_axis(x)

        y = torch.cat([relation_arg_min, relation_arg_max], dim=-1)
        for nl in range(0, self.num_layers):
            y = F.relu(getattr(self, "layer_arg{}".format(nl))(y))
        y = self.layer_arg2(y)
        arg_embeddings = convert_to_arg(y)

        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, hidden_dim, drop):
        super(ConeIntersection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.layer_axis1 = nn.Linear(self.entity_dim * 2, self.hidden_dim)
        self.layer_arg1 = nn.Linear(self.entity_dim * 2, self.hidden_dim)
        self.layer_axis2 = nn.Linear(self.hidden_dim, self.entity_dim)
        self.layer_arg2 = nn.Linear(self.hidden_dim, self.entity_dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings, axis_embedding1, arg_embedding1, multi_hop1, axis_embedding2, arg_embedding2, multi_hop2, axis_embedding3 = [], arg_embedding3 = [], multi_hop3 = []):
        # offset-deepset
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        #center-attention
        if multi_hop1 != None:
            multi_hop1 = multi_hop1.squeeze(1).squeeze(1)
            multi_hop2 = multi_hop2.squeeze(1).squeeze(1)
            if multi_hop1.shape != multi_hop2.shape:
                print("multi_hop1/2:")
                print(multi_hop1.shape)
                print(multi_hop2.shape)
            if len(arg_embedding3) > 0:
                multi_hop3 = multi_hop3.squeeze(1).squeeze(1)
                if multi_hop3.shape != multi_hop1.shape:
                    print("multi_hop1/3:")
                    print(multi_hop1.shape)
                    print(multi_hop3.shape)
                inter_one_hot = multi_hop1 + multi_hop2 + multi_hop3
                inter_one_hot[inter_one_hot < 3] = 0
                inter_one_hot[inter_one_hot == 3] = 1
                similarWeight1 = multi_hop1 - inter_one_hot
                similarWeight2 = multi_hop2 - inter_one_hot
                similarWeight3 = multi_hop3 - inter_one_hot
                similarWeight1 = torch.sum(similarWeight1, dim=1, keepdim=True) + 1
                similarWeight2 = torch.sum(similarWeight2, dim=1, keepdim=True) + 1
                similarWeight3 = torch.sum(similarWeight3, dim=1, keepdim=True) + 1
                similarWeight1 = 1 / similarWeight1
                similarWeight2 = 1 / similarWeight2
                similarWeight3 = 1 / similarWeight3
            else:
                inter_one_hot = multi_hop1 + multi_hop2
                inter_one_hot[inter_one_hot < 2] = 0
                inter_one_hot[inter_one_hot == 2] = 1
                similarWeight1 = multi_hop1 - inter_one_hot
                similarWeight2 = multi_hop2 - inter_one_hot
                similarWeight1 = torch.sum(similarWeight1, dim=1, keepdim=True) + 1
                similarWeight2 = torch.sum(similarWeight2, dim=1, keepdim=True) + 1
                similarWeight1 = 1 / similarWeight1
                similarWeight2 = 1 / similarWeight2
        else:
            similarWeight1 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()
            similarWeight2 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()
            similarWeight3 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()

        logit1 = torch.cat([axis_embedding1 - arg_embedding1, axis_embedding1 + arg_embedding1], dim=-1)
        axis_layer_act1 = self.layer_axis2(F.relu(self.layer_axis1(logit1)))
        logit2 = torch.cat([axis_embedding2 - arg_embedding2, axis_embedding2 + arg_embedding2], dim=-1)
        axis_layer_act2 = self.layer_axis2(F.relu(self.layer_axis1(logit2)))
        axis_attention = F.softmax(torch.stack([axis_layer_act1 * similarWeight1, axis_layer_act2 * similarWeight2]), dim=0)
        x_embeddings = axis_attention[0] * torch.cos(axis_embedding1) + axis_attention[1] * torch.cos(axis_embedding2)
        y_embeddings = axis_attention[0] * torch.sin(axis_embedding1) + axis_attention[1] * torch.sin(axis_embedding2)

        if len(axis_embedding3) > 0:
            logit3 = torch.cat([axis_embedding3 - arg_embedding3, axis_embedding3 + arg_embedding3], dim=-1)
            axis_layer_act3 = self.layer_axis2(F.relu(self.layer_axis1(logit3)))
            axis_attention = F.softmax(torch.stack([axis_layer_act1 * similarWeight1, axis_layer_act2 * similarWeight2, axis_layer_act3 * similarWeight3]), dim=0)
            x_embeddings = axis_attention[0] * torch.cos(axis_embedding1) + axis_attention[1] * torch.cos(axis_embedding2) + axis_attention[2] * torch.cos(axis_embedding3)
            y_embeddings = axis_attention[0] * torch.sin(axis_embedding1) + axis_attention[1] * torch.sin(axis_embedding2) + axis_attention[2] * torch.sin(axis_embedding1)

        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        return axis_embeddings, arg_embeddings

class ConeDifference3(nn.Module):
    def __init__(self, center_dim, offset_dim):
        super(ConeDifference3, self).__init__()
        assert center_dim == offset_dim
        self.center_dim = center_dim
        self.offset_dim = offset_dim
        self.attention_axis_1 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.attention_axis_2 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.attention_axis_3 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.layer_deepset_inside = nn.Linear(self.center_dim * 2, self.center_dim)
        self.layer_deepset_outside = nn.Linear(self.center_dim, self.center_dim)

        nn.init.xavier_uniform(self.attention_axis_1)
        nn.init.xavier_uniform(self.attention_axis_2)
        nn.init.xavier_uniform(self.attention_axis_3)
        nn.init.xavier_uniform_(self.layer_deepset_inside.weight)
        nn.init.xavier_uniform_(self.layer_deepset_outside.weight)

    def forward(self, axis_embedding1, arg_embedding1, axis_embedding2, arg_embedding2, axis_embedding3 = None, arg_embedding3 = None):
        #axis:
        logit1 = torch.cat([axis_embedding1 - arg_embedding1, axis_embedding1 + arg_embedding1], dim=-1)
        axis_weight1 = torch.matmul(logit1, self.attention_axis_1)
        logit2 = torch.cat([axis_embedding2 - arg_embedding2, axis_embedding2 + arg_embedding2], dim=-1)
        axis_weight2 = torch.matmul(logit2, self.attention_axis_2)
        #arg
        logit1_2 = torch.cat([torch.sin((axis_embedding1 - axis_embedding2) / 2), torch.sin((arg_embedding1 - arg_embedding2) / 2)], dim=-1)
        MLP_inside_input = torch.stack([logit1_2], dim=0)
        if axis_embedding3 == None:
            #axis
            combined1 = F.softmax(torch.cat([axis_weight1, axis_weight2], dim=1), dim=1)
            atten1 = (combined1[:, 0].view(logit1.size(0), 1)).unsqueeze(1)
            atten2 = (combined1[:, 1].view(logit2.size(0), 1)).unsqueeze(1)
            x_embedding = torch.cos(axis_embedding1) * atten1 + torch.cos(axis_embedding2) * atten2
            y_embedding = torch.sin(axis_embedding1) * atten1 + torch.sin(axis_embedding2) * atten2
            x_embedding[torch.abs(x_embedding) < 1e-3] = 1e-3
            axis_embedding = torch.atan(y_embedding / x_embedding)

            indicator_x = x_embedding < 0
            indicator_y = y_embedding < 0
            indicator_two = indicator_x & torch.logical_not(indicator_y)
            indicator_three = indicator_x & indicator_y

            axis_embedding[indicator_two] = axis_embedding[indicator_two] + pi
            axis_embedding[indicator_three] = axis_embedding[indicator_three] - pi
        else:
            #arg
            logit1_3 = torch.cat([torch.sin((axis_embedding1 - axis_embedding3) / 2), torch.sin((arg_embedding1 - arg_embedding3) / 2)], dim=-1)
            MLP_inside_input = torch.stack([logit1_2, logit1_3], dim=0)
            # axis:
            logit3 = torch.cat([axis_embedding3 - arg_embedding3, axis_embedding3 + arg_embedding3], dim=-1)
            axis_weight3 = torch.matmul(logit3, self.attention_axis_3)
            combined1 = F.softmax(torch.cat([axis_weight1, axis_weight2, axis_weight3], dim=1), dim=1)
            atten1 = (combined1[:, 0].view(axis_embedding1.size(0), 1)).unsqueeze(1)
            atten2 = (combined1[:, 1].view(axis_embedding2.size(0), 1)).unsqueeze(1)
            atten3 = (combined1[:, 2].view(axis_embedding3.size(0), 1)).unsqueeze(1)
            x_embedding = torch.cos(axis_embedding1) * atten1 + torch.cos(axis_embedding2) * atten2 + torch.cos(
                axis_embedding3) * atten3
            y_embedding = torch.sin(axis_embedding1) * atten1 + torch.sin(axis_embedding2) * atten2 + torch.sin(
                axis_embedding3) * atten3
            x_embedding[torch.abs(x_embedding) < 1e-3] = 1e-3
            axis_embedding = torch.atan(y_embedding / x_embedding)
            indicator_x = x_embedding < 0
            indicator_y = y_embedding < 0
            indicator_two = indicator_x & torch.logical_not(indicator_y)
            indicator_three = indicator_x & indicator_y
            axis_embedding[indicator_two] = axis_embedding[indicator_two] + pi
            axis_embedding[indicator_three] = axis_embedding[indicator_three] - pi

        deepsets_inside = F.relu(self.layer_deepset_inside(MLP_inside_input))
        deepsets_inside_mean = torch.mean(deepsets_inside, dim=0)
        gate = torch.sigmoid(self.layer_deepset_outside(deepsets_inside_mean))
        new_arg_embedding = arg_embedding1 * gate

        return axis_embedding, new_arg_embedding


class HaLk(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 cen=None, gamma2=0, activation='relu',
                 node_group_one_hot_vector_single=None, group_adj_matrix_single=None,
                 node_group_one_hot_vector_multi=None, group_adj_matrix_multi=None, drop=0.):
        super(HaLk, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.cen = cen
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(
            torch.Tensor([gamma2]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.axis_scale = 1.0
        self.arg_scale = 1.0
        self.angle_scale = AngleScale(self.embedding_range.item())
        self.cone_entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True) #entity的axis轴embedding
        nn.init.uniform_(
            tensor=self.cone_entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)#relation的axis轴embedding
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)#relation的arg开角embedding
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.cone_proj = ConeProjection(self.entity_dim, 800, 2)
        self.cone_intersection = ConeIntersection(self.entity_dim, hidden_dim, drop)
        self.cone_difference = ConeDifference3(self.entity_dim, self.relation_dim)
        self.cone_negation = ConeNegation(self.entity_dim, self.relation_dim, hidden_dim)
        self.mudulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.each_relation_a_matrix = True
        if group_adj_matrix_single is not None:
            self.group_adj_matrix = torch.tensor(group_adj_matrix_single.tolist(), requires_grad=False).cuda() #代码指明用GPU进行tensor的计算
        if node_group_one_hot_vector_single is not None:
            self.node_group_one_hot_vector = torch.tensor(node_group_one_hot_vector_single.tolist(),
                                                          requires_grad=False).cuda()
        self.group_adj_weight = nn.Parameter(torch.tensor([1]).float(), requires_grad=True)

        if group_adj_matrix_multi is not None:
            self.group_adj_matrix_multi = []
            for xxx in group_adj_matrix_multi:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.group_adj_matrix_multi.append(torch.tensor(xxx.tolist(), requires_grad=False).cuda())
        if node_group_one_hot_vector_multi is not None:
            self.node_group_one_hot_vector_multi = []
            for xxx in node_group_one_hot_vector_multi:
                self.node_group_one_hot_vector_multi.append(torch.tensor(xxx.tolist(),
                                                          requires_grad=False).cuda())
        self.group_times = len(self.group_adj_matrix_multi)
        self.group_adj_weight_multi = nn.Parameter(torch.tensor([0.1]).float(), requires_grad=True)

        self.disjoin_weight_for_group_matrix = nn.Parameter(torch.tensor([0.5]).float(), requires_grad=True)
        self.disjoin_then_proj_threshold = nn.Parameter(torch.tensor([0.5]).float(), requires_grad=True)

        self.shift_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
        nn.init.uniform_(
            tensor=self.shift_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name not in ['HaLkCone']:
            raise ValueError('model %s not supported' % model_name)

    def forward(self, is_train, sample=None, rel_len=None, qtype=None, mode='single', step=-1):
        shift_of_node = None
        one_hot_head = None
        tail_one_hot = None
        if qtype == 'inter-neg-chain':
            if mode == 'single':
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                relation_11 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)

                offset_11 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, 0]).unsqueeze(1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, 2]).unsqueeze(1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2])

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, -1]).unsqueeze(1)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                    relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                            negative_sample_size,
                                                                                                            -1)

                relation_11 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_12 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)

                offset_11 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 2]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=tail_part.view(-1)).view(batch_size,
                                                                                 negative_sample_size,
                                                                                 -1)
                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0,
                                                            index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0,
                                                            index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 4]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

        elif qtype == 'chain-inter-neg' or qtype == 'chain-neg-inter':
            if mode == 'single':
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 3]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                relation_11 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift1 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                shift2 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                shift_of_node = torch.cat([shift1, shift2], dim=0)

                offset_11 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, 0]).unsqueeze(1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, 3]).unsqueeze(1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0, index=sample[:, -1]).unsqueeze(1)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                    relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                            negative_sample_size,
                                                                                                            -1)
                relation_11 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_12 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1)
                shift2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1)
                shift_of_node = torch.cat([shift1, shift2], dim=0)

                offset_11 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 3]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=tail_part.view(-1)).view(batch_size,
                                                                                 negative_sample_size,
                                                                                 -1)
                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0,
                                                            index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0,
                                                            index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 4]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

        elif qtype == '2-inter-neg' or qtype == '3-inter-neg':
            if mode == 'single':
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=sample[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=sample[:, 2]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                        index=sample[:, 4]).unsqueeze(1)
                    one_hot_head = torch.cat([one_hot_head, one_hot_head_3], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot], dim=0)
                elif rel_len == 3:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot, tail_one_hot], dim=0)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=sample[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 2]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                        index=head_part[:, 4]).unsqueeze(1)
                    one_hot_head = torch.cat([one_hot_head, one_hot_head_3], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                if rel_len == 2:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot], dim=0)
                elif rel_len == 3:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot, tail_one_hot], dim=0)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=head_part[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], dim=0)

        elif qtype == 'chain-inter':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)
            relation_11 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            shift1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1)
            shift2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1)
            shift_of_node = torch.cat([shift1, shift2], dim=0)

            offset_11 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            offset_12 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

            # group info
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                index=head_part[:, 0]).unsqueeze(
                1)
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                index=head_part[:, 3]).unsqueeze(
                1)
            one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

            tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                              index=tail_part.view(-1)).view(batch_size,
                                                                             negative_sample_size,
                                                                             -1)
            # relation matrix
            if self.each_relation_a_matrix:
                relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0,
                                                        index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0,
                                                        index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)


            offset_11 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            offset_12 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

            # group info
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                index=head_part[:, 0]).unsqueeze(
                1)
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                index=head_part[:, 2]).unsqueeze(
                1)
            one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

            tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                              index=tail_part.view(-1)).view(batch_size,
                                                                             negative_sample_size,
                                                                             -1)
            # relation matrix
            if self.each_relation_a_matrix:
                relation_matrix_11 = torch.index_select(self.group_adj_matrix, dim=0,
                                                        index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_12 = torch.index_select(self.group_adj_matrix, dim=0,
                                                        index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                       index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)

        elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin':
            if mode == 'single':
                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=sample[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=sample[:, 2]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                        index=sample[:, 4]).unsqueeze(1)
                    one_hot_head = torch.cat([one_hot_head, one_hot_head_3], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot], dim=0)
                elif rel_len == 3:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot, tail_one_hot], dim=0)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=sample[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_1 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)

                # group info
                one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                    index=head_part[:, 2]).unsqueeze(
                    1)
                one_hot_head = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                        index=head_part[:, 4]).unsqueeze(1)
                    one_hot_head = torch.cat([one_hot_head, one_hot_head_3], dim=0)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                if rel_len == 2:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot], dim=0)
                elif rel_len == 3:
                    tail_one_hot = torch.cat([tail_one_hot, tail_one_hot, tail_one_hot], dim=0)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                           index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=head_part[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            if mode == 'single':
                head = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                relation = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)

                    offset2 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset2], 1)

                if rel_len == 3:
                    relation3 = torch.index_select(self.axis_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    offset3 = torch.index_select(self.arg_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                assert offset.size(1) == rel_len

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                if rel_len == 3:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1)
                elif rel_len == 2:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1)
                elif rel_len == 1:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1)

                # group info
                one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=sample[:, 0]).unsqueeze(1)

                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=sample[:, -1]).unsqueeze(1)

                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)

                    if rel_len == 2 or rel_len == 3:
                        relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=sample[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_2], 1)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=sample[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], 1)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                head = torch.index_select(self.cone_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                relation = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                offset = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    offset2 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset2], 1)

                if rel_len == 3:
                    relation3 = torch.index_select(self.axis_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    offset3 = torch.index_select(self.arg_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset3], 1)

                if rel_len == 3:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                elif rel_len == 2:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                elif rel_len == 1:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)

                assert relation.size(1) == rel_len
                assert offset.size(1) == rel_len

                tail = torch.index_select(self.cone_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                # group info
                one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=head_part[:, 0]).unsqueeze(1)
                tail_one_hot = torch.index_select(self.node_group_one_hot_vector, dim=0,
                                                  index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                # relation matrix
                if self.each_relation_a_matrix:
                    relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    if rel_len == 2 or rel_len == 3:
                        relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=head_part[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_2], 1)
                    if rel_len == 3:
                        relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0,
                                                               index=head_part[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        relation_matrix = torch.cat([relation_matrix, relation_matrix_3], 1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'HaLkCone': self.HaLkCone
        }

        if not is_train or (is_train and step % 1000 == 0):
            one_hot_head_multi, relation_matrix_multi, tail_one_hot_multi = prepare_data(sample, qtype, mode,
                                                                                         self.group_times,
                                                                                         self.node_group_one_hot_vector_multi,
                                                                                         self.group_adj_matrix_multi,
                                                                                         rel_len)

            if self.model_name in model_func:
                if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin' or qtype == '2-inter-neg' or qtype == '3-inter-neg':
                    score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](False, head,
                                                                                                   relation, tail,
                                                                                                   mode, offset, 1,
                                                                                                   qtype,
                                                                                                   shift_of_node,
                                                                                                   one_hot_head,
                                                                                                   relation_matrix,
                                                                                                   tail_one_hot,
                                                                                                   head_one_hot_multi=one_hot_head_multi,
                                                                                                   relation_matrix_multi=relation_matrix_multi,
                                                                                                   tail_one_hot_multi=tail_one_hot_multi)
                else:
                    score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](False, head,
                                                                                                   relation, tail,
                                                                                                   mode, offset, rel_len,
                                                                                                   qtype, shift_of_node,
                                                                                                   one_hot_head,
                                                                                                   relation_matrix,
                                                                                                   tail_one_hot,
                                                                                                   head_one_hot_multi=one_hot_head_multi,
                                                                                                   relation_matrix_multi=relation_matrix_multi,
                                                                                                   tail_one_hot_multi=tail_one_hot_multi)
            else:
                raise ValueError('model %s not supported' % self.model_name)
        else:

            if self.model_name in model_func:
                if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin' or qtype == '2-inter-neg' or qtype == '3-inter-neg':
                    score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](is_train, head, relation, tail,
                                                                                                   mode, offset, 1, qtype,
                                                                                                   shift_of_node,
                                                                                                   one_hot_head,
                                                                                                   relation_matrix,
                                                                                                   tail_one_hot)
                else:
                    score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](is_train, head, relation, tail,
                                                                                                   mode, offset, rel_len,
                                                                                                   qtype, shift_of_node,
                                                                                                   one_hot_head,
                                                                                                   relation_matrix,
                                                                                                   tail_one_hot)
            else:
                raise ValueError('model %s not supported' % self.model_name)

        return score, score_cen, offset_norm, score_cen_plus, None, None

    def HaLkCone(self, is_train, head, relation, tail, mode, offset, rel_len, qtype, shift_of_node, head_one_hot,
                   relation_matrix, tail_one_hot, head_one_hot_multi=None, relation_matrix_multi=None, tail_one_hot_multi=None):
        if not is_train:
            dis_group_multi_whole = run_multi_group(rel_len, qtype, self.group_times, head_one_hot_multi, relation_matrix_multi, tail_one_hot_multi,
                            self.disjoin_weight_for_group_matrix, self.disjoin_then_proj_threshold)

        def distance_outside_inside(entity_embedding, query_axis_embedding, query_arg_embedding, shift_embedding = None):
            if shift_embedding != None:
                delta1 = entity_embedding + shift_embedding - (query_axis_embedding - query_arg_embedding)
                delta2 = entity_embedding + shift_embedding - (query_axis_embedding + query_arg_embedding)
            else:
                delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
                delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)
            distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
            distance_base = torch.abs(torch.sin(query_arg_embedding / 2))
            # distance2axis = torch.abs(torch.sin(entity_embedding - query_axis_embedding) / 2)
            # distance_base = torch.abs(torch.sin(query_arg_embedding * 2) / 2)
            indicator_in = distance2axis < distance_base
            distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
            # distance_out = torch.min(torch.abs(torch.sin(delta1) / 2), torch.abs(torch.sin(delta2) / 2))
            distance_out[indicator_in] = 0.
            distance_in = torch.min(distance2axis, distance_base)
            return distance_out, distance_in

        head_one_hot = torch.unsqueeze(head_one_hot, 2)
        tail_one_hot = torch.unsqueeze(tail_one_hot, 2)
        if qtype == 'chain-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[2][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1
            # negation for 2
            query_one_hot2[query_one_hot2 == 1] = 2
            query_one_hot2[query_one_hot2 < 1] = 1
            query_one_hot2[query_one_hot2 == 2] = 0

            # intersection
            query_one_hot_res = query_one_hot1 + query_one_hot2
            query_one_hot_res[query_one_hot_res < 2] = 0
            query_one_hot_res[query_one_hot_res >= 2] = 1

            group_dist = F.relu(tail_one_hot - query_one_hot_res)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            # projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)
            for i in range(1, 2):
                # projection
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.axis_scale)
                relation_center = convert_to_axis(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.axis_scale)
                relation_offset = convert_to_axis(relation_offset)
                query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1, relation_center,
                                                                relation_offset * 0.5)
            # projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                            relation_offset * 0.5)
            # negation for 2
            query_center_2, query_offset_2 = self.cone_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack,
                                                                        query_center_1, query_offset_1, query_one_hot1,
                                                                        query_center_2, query_offset_2, query_one_hot2)

            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - tail

        elif qtype == 'chain-neg-inter':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1
            # negation for 1
            query_one_hot1[query_one_hot1 == 1] = 2
            query_one_hot1[query_one_hot1 < 1] = 1
            query_one_hot1[query_one_hot1 == 2] = 0

            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[2][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1

            # intersection
            query_one_hot_res = query_one_hot1 + query_one_hot2
            query_one_hot_res[query_one_hot_res < 2] = 0
            query_one_hot_res[query_one_hot_res >= 2] = 1

            group_dist = F.relu(tail_one_hot - query_one_hot_res)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            # projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)
            for i in range(1, 2):
                # projection
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.axis_scale)
                relation_center = convert_to_axis(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.axis_scale)
                relation_offset = convert_to_axis(relation_offset)
                query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1, relation_center,
                                                                relation_offset * 0.5)

            # negation
            query_center_1, query_offset_1 = self.cone_negation(query_center_1, query_offset_1)

            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                            relation_offset * 0.5)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack,
                                                                        query_center_1, query_offset_1, query_one_hot1,
                                                                        query_center_2, query_offset_2, query_one_hot2)

            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - tail

        elif qtype == 'inter-neg-chain':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # pro 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1
            # pro 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1
            # negation for 2
            query_one_hot2[query_one_hot2 == 1] = 2
            query_one_hot2[query_one_hot2 < 1] = 1
            query_one_hot2[query_one_hot2 == 2] = 0

            # intersection
            new_query_one_hot = query_one_hot1 + query_one_hot2
            new_query_one_hot[new_query_one_hot < 2] = 0
            new_query_one_hot[new_query_one_hot >= 2] = 1

            new_query_one_hot = torch.matmul(new_query_one_hot, relation_matrix_chunk[2][:, 0, :, :])
            new_query_one_hot[new_query_one_hot >= 1] = 1
            new_query_one_hot[new_query_one_hot < 1] = 0

            group_dist = F.relu(tail_one_hot - new_query_one_hot)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            # projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)

            #projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.axis_scale)
            relation_center_2 = convert_to_axis(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.axis_scale)
            relation_offset_2 = convert_to_axis(relation_offset_2)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center_2,
                                                            relation_offset_2 * 0.5)
            # negation
            query_center_2, query_offset_2 = self.cone_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack,
                                                                        query_center_1, query_offset_1,
                                                                        query_one_hot1, query_center_2,
                                                                        query_offset_2, query_one_hot2)

            # chain:
            relation_center_3 = relations[2][:, 0, :, :]
            relation_center_3 = self.angle_scale(relation_center_3, self.axis_scale)
            relation_center_3 = convert_to_axis(relation_center_3)
            relation_offset_3 = offsets[2][:, 0, :, :]
            relation_offset_3 = self.angle_scale(relation_offset_3, self.axis_scale)
            relation_offset_3 = convert_to_axis(relation_offset_3)
            new_query_center, new_query_offset = self.cone_proj(new_query_center.unsqueeze(1),
                                                                new_query_offset.unsqueeze(1), relation_center_3,
                                                                relation_offset_3 * 0.5)

            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset)
            score_center = new_query_center - tail

        elif qtype == '2-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 2, dim=0)
            tail_one_hot_chunk = torch.chunk(tail_one_hot, 2, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1
            # negation for 2
            query_one_hot2[query_one_hot2 == 1] = 2
            query_one_hot2[query_one_hot2 < 1] = 1
            query_one_hot2[query_one_hot2 == 2] = 0

            # intersection
            query_one_hot_res = query_one_hot1 + query_one_hot2
            query_one_hot_res[query_one_hot_res < 2] = 0
            query_one_hot_res[query_one_hot_res >= 2] = 1

            group_dist = F.relu(tail_one_hot_chunk[0] - query_one_hot_res)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 2, dim=0)
            offsets = torch.chunk(offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            # chain 1
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)
            # chain 2
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[1][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[1][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                            relation_offset * 0.5)
            # negation for 2
            query_center_2, query_offset_2 = self.cone_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack,
                                                                        query_center_1, query_offset_1, query_one_hot1,
                                                                        query_center_2, query_offset_2, query_one_hot2)
            tails = torch.chunk(tail, 2, dim=0)
            cur_tail = tails[0]
            cur_tail = self.angle_scale(cur_tail, self.axis_scale)
            cur_tail = convert_to_axis(cur_tail)
            score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - cur_tail

        elif qtype == '3-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 3, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            tail_one_hot_chunk = torch.chunk(tail_one_hot, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1

            # chain 3
            query_one_hot3 = query_one_hots[2]
            query_one_hot3 = torch.matmul(query_one_hot3, relation_matrix_chunk[2][:, 0, :, :])
            query_one_hot3[query_one_hot3 >= 1] = 1
            # negation for 3
            query_one_hot3[query_one_hot3 == 1] = 2
            query_one_hot3[query_one_hot3 < 1] = 1
            query_one_hot3[query_one_hot3 == 2] = 0

            # intersection
            query_one_hot_res = query_one_hot1 + query_one_hot2 + query_one_hot3
            query_one_hot_res[query_one_hot_res < 3] = 0
            query_one_hot_res[query_one_hot_res >= 3] = 1

            group_dist = F.relu(tail_one_hot_chunk[0] - query_one_hot_res)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)

            heads = torch.chunk(head, 3, dim=0)

            # chain 1
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)
            # chain 2
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[1][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[1][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                            relation_offset * 0.5)
            # chain 3
            query_center_3 = heads[2]
            query_center_3 = self.angle_scale(query_center_3, self.axis_scale)
            query_center_3 = convert_to_axis(query_center_3)
            query_offset_3 = torch.zeros_like(query_center_3).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_3, query_offset_3 = self.cone_proj(query_center_3, query_offset_3 * 0.5, relation_center,
                                                            relation_offset * 0.5)
            # negation
            query_center_3, query_offset_3 = self.cone_negation(query_center_3, query_offset_3)

            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_center_3 = query_center_3.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_offset_3 = query_offset_3.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2, query_center_3], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2, query_offset_3], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack,
                                                                        query_center_1, query_offset_1, query_one_hot1,
                                                                        query_center_2, query_offset_2, query_one_hot2,
                                                                        query_center_3, query_offset_3, query_one_hot3)

            tails = torch.chunk(tail, 3, dim=0)
            cur_tail = tails[0]
            cur_tail = self.angle_scale(cur_tail, self.axis_scale)
            cur_tail = convert_to_axis(cur_tail)
            score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - cur_tail

        elif qtype == 'chain-inter':
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[2][:, 0, :, :])

            query_one_hot1[query_one_hot1 >= 1] = 1
            query_one_hot2[query_one_hot2 >= 1] = 1
            query_one_hot_res = query_one_hot1 + query_one_hot2
            query_one_hot_res[query_one_hot_res < 2] = 0
            query_one_hot_res[query_one_hot_res >= 2] = 1

            group_dist = F.relu(tail_one_hot - query_one_hot_res)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            # chain
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1, relation_offset_1 * 0.5)
            for i in range(1, 2):
                # chain
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.axis_scale)
                relation_center = convert_to_axis(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.axis_scale)
                relation_offset = convert_to_axis(relation_offset)
                query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1, relation_center, relation_offset * 0.5)

            # chain
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.axis_scale)
            relation_center = convert_to_axis(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.axis_scale)
            relation_offset = convert_to_axis(relation_offset)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center, relation_offset * 0.5)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack, query_center_1, query_offset_1, query_one_hot1, query_center_2, query_offset_2, query_one_hot2)

            shifts = torch.chunk(shift_of_node, 2, dim=0)
            shift = (shifts[0] + shifts[1]) / 2
            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            shift = self.angle_scale(shift, self.axis_scale)
            shift = convert_to_axis(shift)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset, shift)
            score_center = new_query_center.unsqueeze(1) - tail - shift

        elif qtype == 'inter-chain' or qtype == 'disjoin-chain':
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])

            # intersection or difference
            if qtype == 'inter-chain':
                new_query_one_hot = query_one_hot1 + query_one_hot2
                new_query_one_hot[new_query_one_hot < 2] = 0
                new_query_one_hot[new_query_one_hot >= 2] = 1
            elif qtype == 'disjoin-chain':
                new_query_one_hot = query_one_hot1 - self.disjoin_weight_for_group_matrix * query_one_hot2
                new_query_one_hot[new_query_one_hot < 0] = 0

            new_query_one_hot = torch.matmul(new_query_one_hot, relation_matrix_chunk[2][:, 0, :, :])
            new_query_one_hot[new_query_one_hot >= self.disjoin_then_proj_threshold] = 1
            new_query_one_hot[new_query_one_hot < self.disjoin_then_proj_threshold] = 0

            group_dist = F.relu(tail_one_hot - new_query_one_hot)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            # chain
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1, relation_offset_1 * 0.5)

            # chain
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.axis_scale)
            relation_center_2 = convert_to_axis(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.axis_scale)
            relation_offset_2 = convert_to_axis(relation_offset_2)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center_2, relation_offset_2 * 0.5)

            if qtype == 'inter-chain':
                #intersection
                query_center_1 = query_center_1.squeeze(1)
                query_center_2 = query_center_2.squeeze(1)
                query_offset_1 = query_offset_1.squeeze(1)
                query_offset_2 = query_offset_2.squeeze(1)
                query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
                query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
                new_query_center, new_query_offset = self.cone_intersection(query_center_stack, query_offset_stack, query_center_1, query_offset_1, query_one_hot1, query_center_2, query_offset_2, query_one_hot2)

                #chain:
                relation_center_3 = relations[2][:, 0, :, :]
                relation_center_3 = self.angle_scale(relation_center_3, self.axis_scale)
                relation_center_3 = convert_to_axis(relation_center_3)
                relation_offset_3 = offsets[2][:, 0, :, :]
                relation_offset_3 = self.angle_scale(relation_offset_3, self.axis_scale)
                relation_offset_3 = convert_to_axis(relation_offset_3)
                new_query_center, new_query_offset = self.cone_proj(new_query_center.unsqueeze(1), new_query_offset.unsqueeze(1), relation_center_3, relation_offset_3 * 0.5)

            elif qtype == 'disjoin-chain':
                #difference
                disjoin_center, disjoin_offset = self.cone_difference(query_center_1, query_offset_1, query_center_2, query_offset_2)
                #chain
                relation_center_3 = relations[2][:, 0, :, :]
                relation_center_3 = self.angle_scale(relation_center_3, self.axis_scale)
                relation_center_3 = convert_to_axis(relation_center_3)
                relation_offset_3 = offsets[2][:, 0, :, :]
                relation_offset_3 = self.angle_scale(relation_offset_3, self.axis_scale)
                relation_offset_3 = convert_to_axis(relation_offset_3)
                new_query_center, new_query_offset = self.cone_proj(disjoin_center, disjoin_offset, relation_center_3, relation_offset_3 * 0.5)

            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            shift_of_node = self.angle_scale(shift_of_node, self.axis_scale)
            shift_of_node = convert_to_axis(shift_of_node)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset, shift_of_node)
            score_center = new_query_center - tail - shift_of_node

        elif qtype == 'union-chain':
            # transform 2u queries to two 1p queries
            # transform up queries to two 2p queries
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1
            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1

            new_query_one_hot = query_one_hot1 + query_one_hot2
            new_query_one_hot[new_query_one_hot < 1] = 0
            new_query_one_hot[new_query_one_hot >= 1] = 1

            new_query_one_hot = torch.matmul(new_query_one_hot, relation_matrix_chunk[2][:, 0, :, :])
            new_query_one_hot[new_query_one_hot >= 1] = 1
            new_query_one_hot[new_query_one_hot < 1] = 0

            group_dist = F.relu(tail_one_hot - new_query_one_hot)
            group_dist = torch.squeeze(group_dist, 2)
            group_dist = torch.norm(group_dist, p=1, dim=2)

            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            #chain
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.axis_scale)
            query_center_1 = convert_to_axis(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                            relation_offset_1 * 0.5)

            #chain
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.axis_scale)
            query_center_2 = convert_to_axis(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.axis_scale)
            relation_center_2 = convert_to_axis(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.axis_scale)
            relation_offset_2 = convert_to_axis(relation_offset_2)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2 * 0.5, relation_center_2,
                                                            relation_offset_2 * 0.5)

            #chain
            relation_center_3 = relations[2][:, 0, :, :]
            relation_center_3 = self.angle_scale(relation_center_3, self.axis_scale)
            relation_center_3 = convert_to_axis(relation_center_3)
            relation_offset_3 = offsets[2][:, 0, :, :]
            relation_offset_3 = self.angle_scale(relation_offset_3, self.axis_scale)
            relation_offset_3 = convert_to_axis(relation_offset_3)
            query_center_1, query_offset_1 = self.cone_proj(query_center_1, query_offset_1, relation_center_3, relation_offset_3 * 0.5)
            query_center_2, query_offset_2 = self.cone_proj(query_center_2, query_offset_2, relation_center_3, relation_offset_3 * 0.5)

            new_query_center = torch.stack([query_center_1, query_center_2], dim=0)
            new_query_offset = torch.stack([query_offset_1, query_offset_2], dim=0)

            tail = self.angle_scale(tail, self.axis_scale)
            tail = convert_to_axis(tail)
            shift_of_node = self.angle_scale(shift_of_node, self.axis_scale)
            shift_of_node = convert_to_axis(shift_of_node)

            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset, shift_of_node)
            score_center = new_query_center - tail - shift_of_node

        else:
            # group info
            query_one_hot = head_one_hot
            for rel in range(rel_len):
                rel_m = relation_matrix[:, rel, :, :]
                query_one_hot = torch.matmul(query_one_hot, rel_m)
            query_one_hot[query_one_hot >= 1] = 1
            if 'inter' not in qtype and 'union' not in qtype and 'disjoin' not in qtype:
                group_dist = F.relu(tail_one_hot - query_one_hot)
                group_dist = torch.squeeze(group_dist, 2)
                group_dist = torch.norm(group_dist, p=1, dim=2)
            else:
                rel_len_ = int(qtype.split('-')[0])
                new_query_one_hot_chunk = torch.chunk(query_one_hot, rel_len_, dim=0)
                tail_one_hot_chunk = torch.chunk(tail_one_hot, rel_len_, dim=0)
                if 'inter' in qtype:
                    new_query_one_hot_ = new_query_one_hot_chunk[0]
                    for i in range(1, rel_len_, 1):
                        new_query_one_hot_ = new_query_one_hot_ + new_query_one_hot_chunk[i]

                    new_query_one_hot_[new_query_one_hot_ < rel_len_] = 0
                    new_query_one_hot_[new_query_one_hot_ >= 1] = 1

                    group_dist = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                    group_dist = torch.squeeze(group_dist, 2)
                    group_dist = torch.norm(group_dist, p=1, dim=2)
                elif 'union' in qtype:
                    new_query_one_hot_ = new_query_one_hot_chunk[0]
                    for i in range(1, rel_len_, 1):
                        new_query_one_hot_ = new_query_one_hot_ + new_query_one_hot_chunk[i]

                    new_query_one_hot_[new_query_one_hot_ < 1] = 0
                    new_query_one_hot_[new_query_one_hot_ >= 1] = 1
                    group_dist = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                    group_dist = torch.squeeze(group_dist, 2)
                    group_dist = torch.norm(group_dist, p=1, dim=2)
                elif 'disjoin' in qtype:
                    if rel_len_ == 2:
                        new_query_one_hot_ = new_query_one_hot_chunk[0] - self.disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[1]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        group_dist = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                        group_dist = torch.squeeze(group_dist, 2)
                        group_dist = torch.norm(group_dist, p=1, dim=2)
                    elif rel_len_ == 3:
                        new_query_one_hot_ = new_query_one_hot_chunk[0] - self.disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[1]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        new_query_one_hot_[new_query_one_hot_ > 0] = 1
                        new_query_one_hot_ = new_query_one_hot_ - self.disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[2]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        group_dist = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                        group_dist = torch.squeeze(group_dist, 2)
                        group_dist = torch.norm(group_dist, p=1, dim=2)
            #chain
            query_center = head
            query_center = self.angle_scale(query_center, self.axis_scale)
            query_center = convert_to_axis(query_center)
            query_offset = torch.zeros_like(query_center).cuda()
            relation_center_1 = relation[:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.axis_scale)
            relation_center_1 = convert_to_axis(relation_center_1)
            relation_offset_1 = offset[:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.axis_scale)
            relation_offset_1 = convert_to_axis(relation_offset_1)
            query_center, query_offset = self.cone_proj(query_center, query_offset * 0.5, relation_center_1, relation_offset_1 * 0.5)
            for rel in range(1, rel_len):
                #chain
                relation_center = relation[:, rel, :, :]
                relation_center = self.angle_scale(relation_center, self.axis_scale)
                relation_center = convert_to_axis(relation_center)
                relation_offset = offset[:, rel, :, :]
                relation_offset = self.angle_scale(relation_offset, self.axis_scale)
                relation_offset = convert_to_axis(relation_offset)
                query_center, query_offset = self.cone_proj(query_center, query_offset, relation_center, relation_offset * 0.5)

            if 'inter' not in qtype and 'union' not in qtype and 'disjoin' not in qtype:
                tail = self.angle_scale(tail, self.axis_scale)
                tail = convert_to_axis(tail)
                shift_of_node = self.angle_scale(shift_of_node, self.axis_scale)
                shift_of_node = convert_to_axis(shift_of_node)
                score_offset, score_center_plus = distance_outside_inside(tail, query_center, query_offset, shift_of_node)
                score_center = query_center - tail - shift_of_node
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                queries_center = torch.chunk(query_center, rel_len, dim=0)
                queries_offset = torch.chunk(query_offset, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                shift = torch.chunk(shift_of_node, rel_len, dim=0)

                if 'inter' in qtype:
                    if rel_len == 2:
                        new_query_center, new_offset = self.cone_intersection(query_center, query_offset, queries_center[0].squeeze(1), queries_offset[0].squeeze(1), new_query_one_hot_chunk[0],
                                                            queries_center[1].squeeze(1), queries_offset[1].squeeze(1), new_query_one_hot_chunk[1])

                    elif rel_len == 3:
                        new_query_center, new_offset = self.cone_intersection(query_center, query_offset, queries_center[0].squeeze(1), queries_offset[0].squeeze(1), new_query_one_hot_chunk[0],
                                                            queries_center[1].squeeze(1), queries_offset[1].squeeze(1), new_query_one_hot_chunk[1],
                                                            queries_center[2].squeeze(1), queries_offset[2].squeeze(1), new_query_one_hot_chunk[2])

                    if rel_len == 2:
                        true_shift = shift[0] + shift[1]
                        true_shift = true_shift / 2
                    elif rel_len == 3:
                        true_shift = shift[0] + shift[1] + shift[2]
                        true_shift = true_shift / 3

                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.axis_scale)
                    cur_tail = convert_to_axis(cur_tail)
                    true_shift = self.angle_scale(true_shift, self.axis_scale)
                    true_shift = convert_to_axis(true_shift)
                    score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1), new_offset.unsqueeze(1), true_shift)
                    score_center = new_query_center.unsqueeze(1) - cur_tail - true_shift

                elif 'union' in qtype:
                    new_query_center = torch.stack(queries_center, dim=0)
                    new_query_offset = torch.stack(queries_offset, dim=0)
                    new_shift = torch.stack(shift, dim=0)
                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.axis_scale)
                    cur_tail = convert_to_axis(cur_tail)
                    new_shift = self.angle_scale(new_shift, self.axis_scale)
                    new_shift = convert_to_axis(new_shift)
                    score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center, new_query_offset,
                                                                              new_shift)
                    score_center = new_query_center.unsqueeze(1) - cur_tail - new_shift

                elif 'disjoin' in qtype:
                    if rel_len == 2:
                        new_query_center, new_query_offset = self.cone_difference(queries_center[0],queries_offset[0],
                                                                                         queries_center[1],queries_offset[1])
                    else:
                        new_query_center, new_query_offset = self.cone_difference(queries_center[0],queries_offset[0],
                                                                                         queries_center[1],queries_offset[1],
                                                                                         queries_center[2],queries_offset[2])

                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.axis_scale)
                    cur_tail = convert_to_axis(cur_tail)
                    score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center, new_query_offset)
                    score_center = new_query_center - cur_tail

                else:
                    assert False, 'qtype not exists: %s' % qtype

        score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.group_adj_weight * group_dist
        score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1) - self.group_adj_weight * group_dist
        score_center_plus = self.gamma.item() - (torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(
            score_center_plus, p=1, dim=-1)) * self.mudulus - self.group_adj_weight * group_dist

        if not is_train:
            if qtype == 'disjoin-chain' or qtype == '3-disjoin':
                group_dist = dis_group_multi_whole * 4
                score = score - self.group_adj_weight_multi * group_dist
                score_center = score_center - self.group_adj_weight_multi * group_dist
                score_center_plus = score_center_plus - self.group_adj_weight_multi * group_dist
            else:
                group_dist = dis_group_multi_whole
                score = score - self.group_adj_weight_multi * group_dist
                score_center = score_center - self.group_adj_weight_multi * group_dist
                score_center_plus = score_center_plus - self.group_adj_weight_multi * group_dist

        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
            score_center = torch.max(score_center, dim=0)[0]
            score_center_plus = torch.max(score_center_plus, dim=0)[0]
        return score, score_center, torch.mean(torch.norm(offset, p=2, dim=2).squeeze(1)), score_center_plus, None

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        if train_iterator.qtype == "inter-neg-chain" or train_iterator.qtype == 'chain-inter-neg' or train_iterator.qtype == 'chain-neg-inter':
            rel_len = 2
        else:
            rel_len = int(train_iterator.qtype.split('-')[0])
        qtype = train_iterator.qtype
        negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model(True,(positive_sample, negative_sample),
                                                                                                   rel_len, qtype, mode=mode, step=step)
        negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)

        positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, _ = model(True, positive_sample,
                                                                                                   rel_len, qtype, step=step)
        positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, test_ans, test_ans_hard, args):
        qtype = test_triples[0][-1]
        if qtype == 'chain-inter-neg' or qtype == 'chain-neg-inter' or qtype == 'inter-neg-chain':
            rel_len = 2
        elif qtype == 'chain-inter' or qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
            rel_len = 2
        else:
            rel_len = int(test_triples[0][-1].split('-')[0])

        model.eval()

        if qtype == '2-inter-neg' or qtype == '3-inter-neg':
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestInterDataset.collate_fn
            )
        elif qtype == 'chain-inter-neg' or qtype == 'chain-neg-inter':
            test_dataloader_tail = DataLoader(
                TestChainInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestChainInterDataset.collate_fn
            )
        elif qtype == 'inter-neg-chain':
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestInterChainDataset.collate_fn
            )
        elif qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif qtype == 'chain-inter':
            test_dataloader_tail = DataLoader(
                TestChainInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif 'inter' in qtype or 'union' in qtype or 'disjoin' in qtype:
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        else:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )

        test_dataset_list = [test_dataloader_tail]
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, query in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    assert batch_size == 1, batch_size

                    if 'inter' in qtype or 'disjoin' in qtype:
                        _, _, _, finalScore, _, _ = model(False, (positive_sample, negative_sample),
                                                                      rel_len,
                                                                      qtype, mode=mode)
                    else:
                        _, _, _, finalScore, _, _ = model(False, (positive_sample, negative_sample),
                                                                          rel_len,
                                                                          qtype, mode=mode)

                    score = finalScore

                    ans = test_ans[query]
                    hard_ans = test_ans_hard[query]
                    all_idx = set(range(args.nentity))
                    false_ans = all_idx - ans
                    hard_ans_list = list(hard_ans)
                    if len(hard_ans) >= 1848:
                        hard_ans_list = hard_ans_list[0:1500]

                    false_ans_list = list(false_ans)
                    ans_idxs = np.array(hard_ans_list)
                    vals = np.zeros((len(ans_idxs), args.nentity))
                    vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    axis2 = np.tile(false_ans_list, len(ans_idxs))
                    axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    vals[axis1, axis2] = 1
                    b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()

                    score -= (torch.min(score) - 1)
                    filter_score2 = b * score
                    argsort2 = torch.argsort(filter_score2, dim=1, descending=True)
                    ans_tensor = torch.LongTensor(hard_ans_list) if not args.cuda else torch.LongTensor(
                        hard_ans_list).cuda()
                    argsort2 = torch.transpose(torch.transpose(argsort2, 0, 1) - ans_tensor, 0, 1)
                    ranking2 = (argsort2 == 0).nonzero()
                    ranking2 = ranking2[:, 1]
                    ranking2 = ranking2 + 1

                    num_ans = len(hard_ans_list)
                    hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                    hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                    hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()
                    mrm_newd = torch.mean(ranking2.to(torch.float)).item()
                    mrrm_newd = torch.mean(1. / ranking2.to(torch.float)).item()

                    logs.append({
                        'MRRm_new': mrrm_newd,
                        'MRm_new': mrm_newd,
                        'HITS@1m_new': hits1m_newd,
                        'HITS@3m_new': hits3m_newd,
                        'HITS@10m_new': hits10m_newd,
                        'num_answer': num_ans
                    })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        num_answer = sum([log['num_answer'] for log in logs])
        for metric in logs[0].keys():
            if metric == 'num_answer':
                continue
            if 'm' in metric:
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            else:
                metrics[metric] = sum([log[metric] for log in logs]) / num_answer
        return metrics

