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

    def __call__(self, circle_point_embedding, scale=None):
        if scale is None:
            scale = pi
        return circle_point_embedding / self.embedding_range * scale

def arc_angle_range_regular(x):
    y = torch.tanh(x) * pi + pi
    return y

def half_arc_angle_range_regular(x):
    y = torch.tanh(x) * pi / 2 + pi / 2
    return y

def angle_range_regular(x):
    y = torch.tanh(x) * pi
    return y

def Identity(x):
    return x

class ArcNegation(nn.Module):
    def __init__(self, center_dim, arclength_dim, hidden_dim):
        super(ArcNegation, self).__init__()
        assert center_dim == arclength_dim
        self.center_dim = center_dim
        self.arclength_dim = arclength_dim
        self.hidden_dim = hidden_dim
        self.medium_center_layer = nn.Linear(self.center_dim, self.hidden_dim)
        self.medium_arclength_layer = nn.Linear(self.arclength_dim, self.hidden_dim)
        self.post_center_layer = nn.Linear(self.hidden_dim * 2, self.center_dim)
        self.post_arclength_layer = nn.Linear(self.hidden_dim * 2, self.arclength_dim)

        nn.init.xavier_uniform_(self.medium_center_layer.weight)
        nn.init.xavier_uniform_(self.medium_arclength_layer.weight)
        nn.init.xavier_uniform_(self.post_center_layer.weight)
        nn.init.xavier_uniform_(self.post_arclength_layer.weight)

    def forward(self, center_embedding, arclength_embedding):
        indicator_positive = center_embedding >= 0
        indicator_negative = center_embedding < 0

        center_embedding[indicator_positive] = center_embedding[indicator_positive] - pi
        center_embedding[indicator_negative] = center_embedding[indicator_negative] + pi

        arclength_embedding = pi - arclength_embedding

        logit_center = F.relu(self.medium_center_layer(center_embedding))
        logit_arclength = F.relu(self.medium_arclength_layer(arclength_embedding))
        logit = torch.cat([logit_center, logit_arclength], dim=-1)

        center_embedding = self.post_center_layer(logit)
        arclength_embedding = self.post_arclength_layer(logit)
        center_embedding = angle_range_regular(center_embedding)
        arclength_embedding = half_arc_angle_range_regular(arclength_embedding)

        return center_embedding, arclength_embedding

class ArcProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ArcProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_center0 = nn.Linear(self.entity_dim + self.entity_dim, self.hidden_dim)
        self.layer_center1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_center2 = nn.Linear(self.hidden_dim, self.entity_dim)
        nn.init.xavier_uniform_(self.layer_center0.weight)
        nn.init.xavier_uniform_(self.layer_center1.weight)
        nn.init.xavier_uniform_(self.layer_center2.weight)
        self.layer_arclength0 = nn.Linear(self.entity_dim + self.entity_dim, self.hidden_dim)
        self.layer_arclength1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_arclength2 = nn.Linear(self.hidden_dim, self.entity_dim)
        nn.init.xavier_uniform_(self.layer_arclength0.weight)
        nn.init.xavier_uniform_(self.layer_arclength1.weight)
        nn.init.xavier_uniform_(self.layer_arclength2.weight)

    def forward(self, source_embedding_center, source_embedding_arclength, r_embedding_center, r_embedding_arclength):
        temp_center = source_embedding_center + r_embedding_center
        temp_start_point = temp_center - r_embedding_arclength - source_embedding_arclength
        temp_end_point = temp_center + r_embedding_arclength + source_embedding_arclength

        x = torch.cat([temp_start_point, temp_end_point], dim=-1)
        for nl in range(0, self.num_layers):
            x = F.relu(getattr(self, "layer_center{}".format(nl))(x))
        x = self.layer_center2(x)
        center_embedding = angle_range_regular(x)

        y = torch.cat([temp_start_point, temp_end_point], dim=-1)
        for nl in range(0, self.num_layers):
            y = F.relu(getattr(self, "layer_arclength{}".format(nl))(y))
        y = self.layer_arclength2(y)
        arclength_embedding = half_arc_angle_range_regular(y)

        return center_embedding, arclength_embedding

class ArcIntersection(nn.Module):
    def __init__(self, dim, hidden_dim, drop):
        super(ArcIntersection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.layer_center1 = nn.Linear(self.entity_dim * 2, self.hidden_dim)
        self.layer_arclength1 = nn.Linear(self.entity_dim * 2, self.hidden_dim)
        self.layer_center2 = nn.Linear(self.hidden_dim, self.entity_dim)
        self.layer_arclength2 = nn.Linear(self.hidden_dim, self.entity_dim)

        nn.init.xavier_uniform_(self.layer_center1.weight)
        nn.init.xavier_uniform_(self.layer_arclength1.weight)
        nn.init.xavier_uniform_(self.layer_center2.weight)
        nn.init.xavier_uniform_(self.layer_arclength2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, center_embeddings, arclength_embeddings, center_embedding1, arclength_embedding1, multi_hop1, center_embedding2, arclength_embedding2, multi_hop2, center_embedding3 = [], arclength_embedding3 = [], multi_hop3 = []):
        # arclength-deepset
        logits = torch.cat([center_embeddings - arclength_embeddings, center_embeddings + arclength_embeddings], dim=-1)
        arclength_layer1_act = F.relu(self.layer_arclength1(logits))
        arclength_layer1_mean = torch.mean(arclength_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arclength2(arclength_layer1_mean))

        arclength_embeddings = self.drop(arclength_embeddings)
        arclength_embeddings, _ = torch.min(arclength_embeddings, dim=0)
        arclength_embeddings = arclength_embeddings * gate

        #center-attention
        if multi_hop1 != None:
            multi_hop1 = multi_hop1.squeeze(1).squeeze(1)
            multi_hop2 = multi_hop2.squeeze(1).squeeze(1)
            if multi_hop1.shape != multi_hop2.shape:
                print("multi_hop1/2:")
                print(multi_hop1.shape)
                print(multi_hop2.shape)
            if len(arclength_embedding3) > 0:
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

        logit1 = torch.cat([center_embedding1 - arclength_embedding1, center_embedding1 + arclength_embedding1], dim=-1)
        center_layer_act1 = self.layer_center2(F.relu(self.layer_center1(logit1)))
        logit2 = torch.cat([center_embedding2 - arclength_embedding2, center_embedding2 + arclength_embedding2], dim=-1)
        center_layer_act2 = self.layer_center2(F.relu(self.layer_center1(logit2)))
        center_attention = F.softmax(torch.stack([center_layer_act1 * similarWeight1, center_layer_act2 * similarWeight2]), dim=0)
        x_embeddings = center_attention[0] * torch.cos(center_embedding1) + center_attention[1] * torch.cos(center_embedding2)
        y_embeddings = center_attention[0] * torch.sin(center_embedding1) + center_attention[1] * torch.sin(center_embedding2)

        if len(center_embedding3) > 0:
            logit3 = torch.cat([center_embedding3 - arclength_embedding3, center_embedding3 + arclength_embedding3], dim=-1)
            center_layer_act3 = self.layer_center2(F.relu(self.layer_center1(logit3)))
            center_attention = F.softmax(torch.stack([center_layer_act1 * similarWeight1, center_layer_act2 * similarWeight2, center_layer_act3 * similarWeight3]), dim=0)
            x_embeddings = center_attention[0] * torch.cos(center_embedding1) + center_attention[1] * torch.cos(center_embedding2) + center_attention[2] * torch.cos(center_embedding3)
            y_embeddings = center_attention[0] * torch.sin(center_embedding1) + center_attention[1] * torch.sin(center_embedding2) + center_attention[2] * torch.sin(center_embedding1)

        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        center_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        center_embeddings[indicator_two] = center_embeddings[indicator_two] + pi
        center_embeddings[indicator_three] = center_embeddings[indicator_three] - pi

        return center_embeddings, arclength_embeddings

class ArcDifference3(nn.Module):
    def __init__(self, center_dim, offset_dim):
        super(ArcDifference3, self).__init__()
        assert center_dim == offset_dim
        self.center_dim = center_dim
        self.offset_dim = offset_dim
        self.attention_center_1 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.attention_center_2 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.attention_center_3 = nn.Parameter(torch.FloatTensor(1, center_dim * 2, 1))
        self.layer_deepset_inside = nn.Linear(self.center_dim * 2, self.center_dim)
        self.layer_deepset_outside = nn.Linear(self.center_dim, self.center_dim)

        nn.init.xavier_uniform(self.attention_center_1)
        nn.init.xavier_uniform(self.attention_center_2)
        nn.init.xavier_uniform(self.attention_center_3)
        nn.init.xavier_uniform_(self.layer_deepset_inside.weight)
        nn.init.xavier_uniform_(self.layer_deepset_outside.weight)

    def forward(self, center_embedding1, arclength_embedding1, center_embedding2, arclength_embedding2, center_embedding3 = None, arclength_embedding3 = None):
        #center:
        logit1 = torch.cat([center_embedding1 - arclength_embedding1, center_embedding1 + arclength_embedding1], dim=-1)
        center_weight1 = torch.matmul(logit1, self.attention_center_1)
        logit2 = torch.cat([center_embedding2 - arclength_embedding2, center_embedding2 + arclength_embedding2], dim=-1)
        center_weight2 = torch.matmul(logit2, self.attention_center_2)
        #arclength
        logit1_2 = torch.cat([torch.sin((center_embedding1 - center_embedding2) / 2), torch.sin((arclength_embedding1 - arclength_embedding2) / 2)], dim=-1)
        MLP_inside_input = torch.stack([logit1_2], dim=0)
        if center_embedding3 == None:
            #center
            combined1 = F.softmax(torch.cat([center_weight1, center_weight2], dim=1), dim=1)
            atten1 = (combined1[:, 0].view(logit1.size(0), 1)).unsqueeze(1)
            atten2 = (combined1[:, 1].view(logit2.size(0), 1)).unsqueeze(1)
            x_embedding = torch.cos(center_embedding1) * atten1 + torch.cos(center_embedding2) * atten2
            y_embedding = torch.sin(center_embedding1) * atten1 + torch.sin(center_embedding2) * atten2
            x_embedding[torch.abs(x_embedding) < 1e-3] = 1e-3
            center_embedding = torch.atan(y_embedding / x_embedding)

            indicator_x = x_embedding < 0
            indicator_y = y_embedding < 0
            indicator_two = indicator_x & torch.logical_not(indicator_y)
            indicator_three = indicator_x & indicator_y

            center_embedding[indicator_two] = center_embedding[indicator_two] + pi
            center_embedding[indicator_three] = center_embedding[indicator_three] - pi
        else:
            #arclength
            logit1_3 = torch.cat([torch.sin((center_embedding1 - center_embedding3) / 2), torch.sin((arclength_embedding1 - arclength_embedding3) / 2)], dim=-1)
            MLP_inside_input = torch.stack([logit1_2, logit1_3], dim=0)
            # center:
            logit3 = torch.cat([center_embedding3 - arclength_embedding3, center_embedding3 + arclength_embedding3], dim=-1)
            center_weight3 = torch.matmul(logit3, self.attention_center_3)
            combined1 = F.softmax(torch.cat([center_weight1, center_weight2, center_weight3], dim=1), dim=1)
            atten1 = (combined1[:, 0].view(center_embedding1.size(0), 1)).unsqueeze(1)
            atten2 = (combined1[:, 1].view(center_embedding2.size(0), 1)).unsqueeze(1)
            atten3 = (combined1[:, 2].view(center_embedding3.size(0), 1)).unsqueeze(1)
            x_embedding = torch.cos(center_embedding1) * atten1 + torch.cos(center_embedding2) * atten2 + torch.cos(
                center_embedding3) * atten3
            y_embedding = torch.sin(center_embedding1) * atten1 + torch.sin(center_embedding2) * atten2 + torch.sin(
                center_embedding3) * atten3
            x_embedding[torch.abs(x_embedding) < 1e-3] = 1e-3
            center_embedding = torch.atan(y_embedding / x_embedding)
            indicator_x = x_embedding < 0
            indicator_y = y_embedding < 0
            indicator_two = indicator_x & torch.logical_not(indicator_y)
            indicator_three = indicator_x & indicator_y
            center_embedding[indicator_two] = center_embedding[indicator_two] + pi
            center_embedding[indicator_three] = center_embedding[indicator_three] - pi

        deepsets_inside = F.relu(self.layer_deepset_inside(MLP_inside_input))
        deepsets_inside_mean = torch.mean(deepsets_inside, dim=0)
        gate = torch.sigmoid(self.layer_deepset_outside(deepsets_inside_mean))
        new_arclength_embedding = arclength_embedding1 * gate

        return center_embedding, new_arclength_embedding


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

        self.polar_angle_scale = 1.0
        self.angle_scale = AngleScale(self.embedding_range.item())
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_center_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.relation_center_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_arclength_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.relation_arclength_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arc_proj = ArcProjection(self.entity_dim, 800, 2)
        self.arc_intersection = ArcIntersection(self.entity_dim, hidden_dim, drop)
        self.arc_difference = ArcDifference3(self.entity_dim, self.relation_dim)
        self.arc_negation = ArcNegation(self.entity_dim, self.relation_dim, hidden_dim)
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

        if model_name not in ['HaLkArc']:
            raise ValueError('model %s not supported' % model_name)

    def forward(self, is_train, sample=None, rel_len=None, qtype=None, mode='single', step=-1):
        shift_of_node = None
        one_hot_head = None
        tail_one_hot = None
        if qtype == 'inter-neg-chain':
            if mode == 'single':
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)

                offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
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
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)

                relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)

                offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
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
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 3]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift1 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                shift2 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                shift_of_node = torch.cat([shift1, shift2], dim=0)

                offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
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
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

                shift1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1)
                shift2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1)
                shift_of_node = torch.cat([shift1, shift2], dim=0)

                offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
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
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 5]).unsqueeze(
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

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
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
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)
            relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            shift1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1)
            shift2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1)
            shift_of_node = torch.cat([shift1, shift2], dim=0)

            offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
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
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)


            offset_11 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            offset_12 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
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
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 5]).unsqueeze(
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

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
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
                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                relation = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)

                    offset2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset2], 1)

                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_center_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    offset3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

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
                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                relation = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                offset = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    offset2 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset, offset2], 1)

                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_center_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    offset3 = torch.index_select(self.relation_arclength_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
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

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
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
            'HaLkArc': self.HaLkArc
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

    def HaLkArc(self, is_train, head, relation, tail, mode, offset, rel_len, qtype, shift_of_node, head_one_hot,
                relation_matrix, tail_one_hot, head_one_hot_multi=None, relation_matrix_multi=None, tail_one_hot_multi=None):
        if not is_train:
            dis_group_multi_whole = run_multi_group(rel_len, qtype, self.group_times, head_one_hot_multi, relation_matrix_multi, tail_one_hot_multi,
                            self.disjoin_weight_for_group_matrix, self.disjoin_then_proj_threshold)

        def distance_outside_inside(entity_embedding, query_center_embedding, query_arclength_embedding, shift_embedding = None):
            if shift_embedding != None:
                delta1 = entity_embedding + shift_embedding - (query_center_embedding - query_arclength_embedding)
                delta2 = entity_embedding + shift_embedding - (query_center_embedding + query_arclength_embedding)
            else:
                delta1 = entity_embedding - (query_center_embedding - query_arclength_embedding)
                delta2 = entity_embedding - (query_center_embedding + query_arclength_embedding)
            distance2center = torch.abs(torch.sin((entity_embedding - query_center_embedding) / 2))
            distance_base = torch.abs(torch.sin(query_arclength_embedding / 2))
            # distance2center = torch.abs(torch.sin(entity_embedding - query_center_embedding) / 2)
            # distance_base = torch.abs(torch.sin(query_arclength_embedding * 2) / 2)
            indicator_in = distance2center < distance_base
            distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
            # distance_out = torch.min(torch.abs(torch.sin(delta1) / 2), torch.abs(torch.sin(delta2) / 2))
            distance_out[indicator_in] = 0.
            distance_in = torch.min(distance2center, distance_base)
            return distance_out, distance_in

        head_one_hot = torch.unsqueeze(head_one_hot, 2)
        tail_one_hot = torch.unsqueeze(tail_one_hot, 2)
        if qtype == 'chain-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # projection 2
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
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)
            for i in range(1, 2):
                # projection
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
                relation_center = angle_range_regular(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
                relation_offset = arc_angle_range_regular(relation_offset)
                query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1, relation_center,
                                                               relation_offset * 0.5)
            # projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                           relation_offset * 0.5)
            # negation for 2
            query_center_2, query_offset_2 = self.arc_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack,
                                                                       query_center_1, query_offset_1, query_one_hot1,
                                                                       query_center_2, query_offset_2, query_one_hot2)

            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - tail

        elif qtype == 'chain-neg-inter':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1
            # negation for 1
            query_one_hot1[query_one_hot1 == 1] = 2
            query_one_hot1[query_one_hot1 < 1] = 1
            query_one_hot1[query_one_hot1 == 2] = 0

            # projection 2
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
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)
            for i in range(1, 2):
                # projection
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
                relation_center = angle_range_regular(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
                relation_offset = arc_angle_range_regular(relation_offset)
                query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1, relation_center,
                                                               relation_offset * 0.5)

            # negation
            query_center_1, query_offset_1 = self.arc_negation(query_center_1, query_offset_1)

            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                           relation_offset * 0.5)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack,
                                                                       query_center_1, query_offset_1, query_one_hot1,
                                                                       query_center_2, query_offset_2, query_one_hot2)

            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
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
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)

            #projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.polar_angle_scale)
            relation_center_2 = angle_range_regular(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.polar_angle_scale)
            relation_offset_2 = arc_angle_range_regular(relation_offset_2)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center_2,
                                                           relation_offset_2 * 0.5)
            # negation
            query_center_2, query_offset_2 = self.arc_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack,
                                                                       query_center_1, query_offset_1,
                                                                       query_one_hot1, query_center_2,
                                                                       query_offset_2, query_one_hot2)

            # projection:
            relation_center_3 = relations[2][:, 0, :, :]
            relation_center_3 = self.angle_scale(relation_center_3, self.polar_angle_scale)
            relation_center_3 = angle_range_regular(relation_center_3)
            relation_offset_3 = offsets[2][:, 0, :, :]
            relation_offset_3 = self.angle_scale(relation_offset_3, self.polar_angle_scale)
            relation_offset_3 = arc_angle_range_regular(relation_offset_3)
            new_query_center, new_query_offset = self.arc_proj(new_query_center.unsqueeze(1),
                                                               new_query_offset.unsqueeze(1), relation_center_3,
                                                               relation_offset_3 * 0.5)

            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset)
            score_center = new_query_center - tail

        elif qtype == '2-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 2, dim=0)
            tail_one_hot_chunk = torch.chunk(tail_one_hot, 2, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # projection 2
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

            # projection 1
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)
            # projection 2
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[1][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[1][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                           relation_offset * 0.5)
            # negation for 2
            query_center_2, query_offset_2 = self.arc_negation(query_center_2, query_offset_2)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack,
                                                                       query_center_1, query_offset_1, query_one_hot1,
                                                                       query_center_2, query_offset_2, query_one_hot2)
            tails = torch.chunk(tail, 2, dim=0)
            cur_tail = tails[0]
            cur_tail = self.angle_scale(cur_tail, self.polar_angle_scale)
            cur_tail = angle_range_regular(cur_tail)
            score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - cur_tail

        elif qtype == '3-inter-neg':
            query_one_hots = torch.chunk(head_one_hot, 3, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            tail_one_hot_chunk = torch.chunk(tail_one_hot, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1

            # projection 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])
            query_one_hot2[query_one_hot2 >= 1] = 1

            # projection 3
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

            # projection 1
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)
            # projection 2
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[1][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[1][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center,
                                                           relation_offset * 0.5)
            # projection 3
            query_center_3 = heads[2]
            query_center_3 = self.angle_scale(query_center_3, self.polar_angle_scale)
            query_center_3 = angle_range_regular(query_center_3)
            query_offset_3 = torch.zeros_like(query_center_3).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_3, query_offset_3 = self.arc_proj(query_center_3, query_offset_3 * 0.5, relation_center,
                                                           relation_offset * 0.5)
            # negation
            query_center_3, query_offset_3 = self.arc_negation(query_center_3, query_offset_3)

            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_center_3 = query_center_3.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_offset_3 = query_offset_3.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2, query_center_3], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2, query_offset_3], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack,
                                                                       query_center_1, query_offset_1, query_one_hot1,
                                                                       query_center_2, query_offset_2, query_one_hot2,
                                                                       query_center_3, query_offset_3, query_one_hot3)

            tails = torch.chunk(tail, 3, dim=0)
            cur_tail = tails[0]
            cur_tail = self.angle_scale(cur_tail, self.polar_angle_scale)
            cur_tail = angle_range_regular(cur_tail)
            score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1),
                                                                      new_query_offset.unsqueeze(1))
            score_center = new_query_center.unsqueeze(1) - cur_tail

        elif qtype == 'chain-inter':
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            # projection 2
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

            # projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1, relation_offset_1 * 0.5)
            for i in range(1, 2):
                # projection
                relation_center = relations[i][:, 0, :, :]
                relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
                relation_center = angle_range_regular(relation_center)
                relation_offset = offsets[i][:, 0, :, :]
                relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
                relation_offset = arc_angle_range_regular(relation_offset)
                query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1, relation_center, relation_offset * 0.5)

            # projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center = relations[2][:, 0, :, :]
            relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
            relation_center = angle_range_regular(relation_center)
            relation_offset = offsets[2][:, 0, :, :]
            relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
            relation_offset = arc_angle_range_regular(relation_offset)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center, relation_offset * 0.5)

            # intersection
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            query_offset_1 = query_offset_1.squeeze(1)
            query_offset_2 = query_offset_2.squeeze(1)
            query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
            query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
            new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack, query_center_1, query_offset_1, query_one_hot1, query_center_2, query_offset_2, query_one_hot2)

            shifts = torch.chunk(shift_of_node, 2, dim=0)
            shift = (shifts[0] + shifts[1]) / 2
            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
            shift = self.angle_scale(shift, self.polar_angle_scale)
            shift = angle_range_regular(shift)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset, shift)
            score_center = new_query_center.unsqueeze(1) - tail - shift

        elif qtype == 'inter-chain' or qtype == 'disjoin-chain':
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            # projection 2
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

            # projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1, relation_offset_1 * 0.5)

            # projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.polar_angle_scale)
            relation_center_2 = angle_range_regular(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.polar_angle_scale)
            relation_offset_2 = arc_angle_range_regular(relation_offset_2)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center_2, relation_offset_2 * 0.5)

            if qtype == 'inter-chain':
                #intersection
                query_center_1 = query_center_1.squeeze(1)
                query_center_2 = query_center_2.squeeze(1)
                query_offset_1 = query_offset_1.squeeze(1)
                query_offset_2 = query_offset_2.squeeze(1)
                query_center_stack = torch.stack([query_center_1, query_center_2], dim=0)
                query_offset_stack = torch.stack([query_offset_1, query_offset_2], dim=0)
                new_query_center, new_query_offset = self.arc_intersection(query_center_stack, query_offset_stack, query_center_1, query_offset_1, query_one_hot1, query_center_2, query_offset_2, query_one_hot2)

                #projection:
                relation_center_3 = relations[2][:, 0, :, :]
                relation_center_3 = self.angle_scale(relation_center_3, self.polar_angle_scale)
                relation_center_3 = angle_range_regular(relation_center_3)
                relation_offset_3 = offsets[2][:, 0, :, :]
                relation_offset_3 = self.angle_scale(relation_offset_3, self.polar_angle_scale)
                relation_offset_3 = arc_angle_range_regular(relation_offset_3)
                new_query_center, new_query_offset = self.arc_proj(new_query_center.unsqueeze(1), new_query_offset.unsqueeze(1), relation_center_3, relation_offset_3 * 0.5)

            elif qtype == 'disjoin-chain':
                #difference
                disjoin_center, disjoin_offset = self.arc_difference(query_center_1, query_offset_1, query_center_2, query_offset_2)
                #projection
                relation_center_3 = relations[2][:, 0, :, :]
                relation_center_3 = self.angle_scale(relation_center_3, self.polar_angle_scale)
                relation_center_3 = angle_range_regular(relation_center_3)
                relation_offset_3 = offsets[2][:, 0, :, :]
                relation_offset_3 = self.angle_scale(relation_offset_3, self.polar_angle_scale)
                relation_offset_3 = arc_angle_range_regular(relation_offset_3)
                new_query_center, new_query_offset = self.arc_proj(disjoin_center, disjoin_offset, relation_center_3, relation_offset_3 * 0.5)

            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
            shift_of_node = self.angle_scale(shift_of_node, self.polar_angle_scale)
            shift_of_node = angle_range_regular(shift_of_node)
            score_offset, score_center_plus = distance_outside_inside(tail, new_query_center, new_query_offset, shift_of_node)
            score_center = new_query_center - tail - shift_of_node

        elif qtype == 'union-chain':
            # transform 2u queries to two 1p queries
            # transform up queries to two 2p queries
            # group info
            query_one_hots = torch.chunk(head_one_hot, 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix, 3, dim=0)
            # projection 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            query_one_hot1[query_one_hot1 >= 1] = 1
            # projection 2
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

            #projection
            query_center_1 = heads[0]
            query_center_1 = self.angle_scale(query_center_1, self.polar_angle_scale)
            query_center_1 = angle_range_regular(query_center_1)
            query_offset_1 = torch.zeros_like(query_center_1).cuda()
            relation_center_1 = relations[0][:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offsets[0][:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1 * 0.5, relation_center_1,
                                                           relation_offset_1 * 0.5)

            #projection
            query_center_2 = heads[1]
            query_center_2 = self.angle_scale(query_center_2, self.polar_angle_scale)
            query_center_2 = angle_range_regular(query_center_2)
            query_offset_2 = torch.zeros_like(query_center_2).cuda()
            relation_center_2 = relations[1][:, 0, :, :]
            relation_center_2 = self.angle_scale(relation_center_2, self.polar_angle_scale)
            relation_center_2 = angle_range_regular(relation_center_2)
            relation_offset_2 = offsets[1][:, 0, :, :]
            relation_offset_2 = self.angle_scale(relation_offset_2, self.polar_angle_scale)
            relation_offset_2 = arc_angle_range_regular(relation_offset_2)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2 * 0.5, relation_center_2,
                                                           relation_offset_2 * 0.5)

            #projection
            relation_center_3 = relations[2][:, 0, :, :]
            relation_center_3 = self.angle_scale(relation_center_3, self.polar_angle_scale)
            relation_center_3 = angle_range_regular(relation_center_3)
            relation_offset_3 = offsets[2][:, 0, :, :]
            relation_offset_3 = self.angle_scale(relation_offset_3, self.polar_angle_scale)
            relation_offset_3 = arc_angle_range_regular(relation_offset_3)
            query_center_1, query_offset_1 = self.arc_proj(query_center_1, query_offset_1, relation_center_3, relation_offset_3 * 0.5)
            query_center_2, query_offset_2 = self.arc_proj(query_center_2, query_offset_2, relation_center_3, relation_offset_3 * 0.5)

            new_query_center = torch.stack([query_center_1, query_center_2], dim=0)
            new_query_offset = torch.stack([query_offset_1, query_offset_2], dim=0)

            tail = self.angle_scale(tail, self.polar_angle_scale)
            tail = angle_range_regular(tail)
            shift_of_node = self.angle_scale(shift_of_node, self.polar_angle_scale)
            shift_of_node = angle_range_regular(shift_of_node)

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
            #projection
            query_center = head
            query_center = self.angle_scale(query_center, self.polar_angle_scale)
            query_center = angle_range_regular(query_center)
            query_offset = torch.zeros_like(query_center).cuda()
            relation_center_1 = relation[:, 0, :, :]
            relation_center_1 = self.angle_scale(relation_center_1, self.polar_angle_scale)
            relation_center_1 = angle_range_regular(relation_center_1)
            relation_offset_1 = offset[:, 0, :, :]
            relation_offset_1 = self.angle_scale(relation_offset_1, self.polar_angle_scale)
            relation_offset_1 = arc_angle_range_regular(relation_offset_1)
            query_center, query_offset = self.arc_proj(query_center, query_offset * 0.5, relation_center_1, relation_offset_1 * 0.5)
            for rel in range(1, rel_len):
                #projection
                relation_center = relation[:, rel, :, :]
                relation_center = self.angle_scale(relation_center, self.polar_angle_scale)
                relation_center = angle_range_regular(relation_center)
                relation_offset = offset[:, rel, :, :]
                relation_offset = self.angle_scale(relation_offset, self.polar_angle_scale)
                relation_offset = arc_angle_range_regular(relation_offset)
                query_center, query_offset = self.arc_proj(query_center, query_offset, relation_center, relation_offset * 0.5)

            if 'inter' not in qtype and 'union' not in qtype and 'disjoin' not in qtype:
                tail = self.angle_scale(tail, self.polar_angle_scale)
                tail = angle_range_regular(tail)
                shift_of_node = self.angle_scale(shift_of_node, self.polar_angle_scale)
                shift_of_node = angle_range_regular(shift_of_node)
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
                        new_query_center, new_offset = self.arc_intersection(query_center, query_offset, queries_center[0].squeeze(1), queries_offset[0].squeeze(1), new_query_one_hot_chunk[0],
                                                                             queries_center[1].squeeze(1), queries_offset[1].squeeze(1), new_query_one_hot_chunk[1])

                    elif rel_len == 3:
                        new_query_center, new_offset = self.arc_intersection(query_center, query_offset, queries_center[0].squeeze(1), queries_offset[0].squeeze(1), new_query_one_hot_chunk[0],
                                                                             queries_center[1].squeeze(1), queries_offset[1].squeeze(1), new_query_one_hot_chunk[1],
                                                                             queries_center[2].squeeze(1), queries_offset[2].squeeze(1), new_query_one_hot_chunk[2])

                    if rel_len == 2:
                        true_shift = shift[0] + shift[1]
                        true_shift = true_shift / 2
                    elif rel_len == 3:
                        true_shift = shift[0] + shift[1] + shift[2]
                        true_shift = true_shift / 3

                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.polar_angle_scale)
                    cur_tail = angle_range_regular(cur_tail)
                    true_shift = self.angle_scale(true_shift, self.polar_angle_scale)
                    true_shift = angle_range_regular(true_shift)
                    score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center.unsqueeze(1), new_offset.unsqueeze(1), true_shift)
                    score_center = new_query_center.unsqueeze(1) - cur_tail - true_shift

                elif 'union' in qtype:
                    new_query_center = torch.stack(queries_center, dim=0)
                    new_query_offset = torch.stack(queries_offset, dim=0)
                    new_shift = torch.stack(shift, dim=0)
                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.polar_angle_scale)
                    cur_tail = angle_range_regular(cur_tail)
                    new_shift = self.angle_scale(new_shift, self.polar_angle_scale)
                    new_shift = angle_range_regular(new_shift)
                    score_offset, score_center_plus = distance_outside_inside(cur_tail, new_query_center, new_query_offset,
                                                                              new_shift)
                    score_center = new_query_center.unsqueeze(1) - cur_tail - new_shift

                elif 'disjoin' in qtype:
                    if rel_len == 2:
                        new_query_center, new_query_offset = self.arc_difference(queries_center[0], queries_offset[0],
                                                                                 queries_center[1], queries_offset[1])
                    else:
                        new_query_center, new_query_offset = self.arc_difference(queries_center[0], queries_offset[0],
                                                                                 queries_center[1], queries_offset[1],
                                                                                 queries_center[2], queries_offset[2])

                    cur_tail = tails[0]
                    cur_tail = self.angle_scale(cur_tail, self.polar_angle_scale)
                    cur_tail = angle_range_regular(cur_tail)
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

