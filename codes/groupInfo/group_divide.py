import os
import pickle
import random
import numpy as np
import click

def get_kg_info(data_path):
    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    return nentity, nrelation

def read_original_triple_data(file_path):
    triples = []
    with open(file_path, 'r') as handle:
        for line in handle.readlines():
            words = line.strip().split("\t")
            words = [int(w.strip()) for w in words]
            triples.append(words)
    return triples


def read_original_id(data_path):
    with open('%s/ind2ent.pkl' % data_path, 'rb') as handle:
        ind2ent = pickle.load(handle)
    new_ind2ent = {}
    for k in ind2ent.keys():
        v = ind2ent[k]
        new_ind2ent[v] = k
    with open('%s/ind2rel.pkl' % data_path, 'rb') as handle:
        ind2rel = pickle.load(handle)
    new_ind2rel = {}
    for k in ind2rel.keys():
        v = ind2rel[k]
        new_ind2rel[v] = k
    return new_ind2ent, new_ind2rel, ind2ent, ind2rel

@click.command()
@click.option('--dataset', default="FB15k-237")
def main(dataset):
    if dataset not in ['FB15k', 'FB15k-237', 'NELL']:
        print("wrong dataset name, please select one of the following name:")
        print('FB15k, FB15k-237, NELL')
        exit(-1)
    nentity, nrelation = get_kg_info(r"../../data/%s" % dataset)
    org_ind2ent, org_ind2rel, ind2ent, ind2rel = read_original_id(r"../../data/%s" % dataset)
    original_triples = read_original_triple_data(r"../../data/%s/kg_triple.txt" % dataset)
    from_to_map_of_original_data = {}
    for t in original_triples:
        if t[0] not in ind2ent:
            continue
        if t[2] not in ind2ent:
            continue
        if t[1] not in ind2rel:
            continue
        x = t[0]
        y = t[1]
        z = t[2]
        if x not in from_to_map_of_original_data:
            from_to_map_of_original_data[x] = {}
        if y not in from_to_map_of_original_data[x]:
            from_to_map_of_original_data[x][y] = set()
        from_to_map_of_original_data[x][y].add(z)

    adj_map = from_to_map_of_original_data

    # print(nentity)
    # print(len(adj_map))

    group_size = 300
    node_id_2_group_id_map = {}
    node_id = list(range(nentity))
    random.shuffle(node_id)
    random.shuffle(node_id)
    random.shuffle(node_id)
    random.shuffle(node_id)
    random.shuffle(node_id)

    for i in range(len(node_id)):
        t_node_id = node_id[i]
        t_group_id = i % group_size
        node_id_2_group_id_map[t_node_id] = t_group_id

    # build group adj matrix
    adj_matrix = np.zeros((nrelation, group_size, group_size))
    for from_node in adj_map.keys():
        pred_map = adj_map[from_node]
        for pred in pred_map.keys():
            to_node_set = pred_map[pred]
            to_nodes = list(to_node_set)
            for t_to_node in to_nodes:
                from_node_group = node_id_2_group_id_map[from_node]
                to_node_group = node_id_2_group_id_map[t_to_node]
                adj_matrix[pred][from_node_group][to_node_group] = 1

    node_group_one_hot_vector = np.zeros((nentity, group_size))
    for key in node_id_2_group_id_map.keys():
        g_id = node_id_2_group_id_map[key]
        node_group_one_hot_vector[key][g_id] = 1

    with open("../../data/%s/group_adj_matrix.pkl" % dataset, "wb") as f:
        pickle.dump(adj_matrix, f)
    with open("../../data/%s/node_group_one_hot_vector.pkl" % dataset, "wb") as f:
        pickle.dump(node_group_one_hot_vector, f)


if __name__ == '__main__':
    main()

