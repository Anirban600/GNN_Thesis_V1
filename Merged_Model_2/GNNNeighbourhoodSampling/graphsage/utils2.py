from graphsage.aggregators import MeanAggregator
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from collections import defaultdict
import json
import pickle
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import random
from graphsage.lsh import train_lsh,get_nearest_neighbors
from graphsage.aggregators import MeanAggregator
from graphsage.planetoid import load_data
import pickle as pk
import time


def return_lsh_candidates(features, n_vectors=16, search_radius=3, num_lsh_neighbours=10, atleast=False, includeNeighbourhood=False, adj_list=None):
    model = train_lsh(features, n_vectors)
    query_vectors = np.copy(features)
    features_copy = np.copy(features)
    lsh_candidates_dic = {}
    for item_id in tqdm(range(features_copy.shape[0])):
        lsh_candidates_dic[item_id] = []
        query_vector = query_vectors[item_id]
        nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model, max_search_radius=search_radius, max_nn=num_lsh_neighbours)
        count = 0
        if atleast:
            if len(nearest_neighbors) < num_lsh_neighbours:
                radius = search_radius + 1
                while True:
                    nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model, max_search_radius=radius)
                    if (len(nearest_neighbors) > num_lsh_neighbours) or (radius >= n_vectors // 2): break
                    radius = radius + 1        
        lsh_candidates_dic[item_id] = nearest_neighbors
    return lsh_candidates_dic


def load_cora(lsh_helper, random_walk=False, root_folder='', embedding=None, augment_khop=False, planetoid=True, load_embeds=False, print_less=False):
    freq = {}
    dist_in_graph = {}
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    degrees = []
    train_mask = []
    test_mask = []
    val_mask = []
    cluster_labels = []
    distances = []
    lsh_neighbourlist_dic = {}
    lsh_cand_dic = {}
    
    if not print_less: print("Using Planetoid Split...")
    adj_lists, feat_data, labels, train_mask, val_mask, test_mask = load_data('cora')
    
    if load_embeds:
        feat_data = embedding.detach().cpu().numpy()
        if not print_less: print("Embedding Loaded with size : ", feat_data.shape)

    if augment_khop:
        print('Creating LSH...')
        lsh_cand_dic = return_lsh_candidates(np.array(feat_data),
                                             n_vectors=lsh_helper['n_vectors'],
                                             num_lsh_neighbours=lsh_helper['num_lsh_neighbours'],
                                             atleast=lsh_helper['atleast'],
                                             search_radius=lsh_helper['search_radius'])
        if not print_less: print('LSH Creation Done...')
        for key, value in adj_lists.items():
            node = int(key)
            if np.all((feat_data[node] == 0)): lsh_neighbourlist_dic[node] = []
            else: lsh_neighbourlist_dic[node] = lsh_cand_dic[node]


    data_loader_dic = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'train_mask': train_mask,
                       'test_mask': test_mask, 'val_mask': val_mask, 'distances': distances,
                       'cluster_labels': cluster_labels, 'freq': freq, 'dist_in_graph': dist_in_graph,
                       'centralityev': [], 'centralitybtw': [], 'centralityh': [],
                       'centralityd': degrees, 'lsh_neighbour_list': lsh_neighbourlist_dic}

    return data_loader_dic