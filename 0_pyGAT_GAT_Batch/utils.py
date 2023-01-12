import numpy as np
import scipy.sparse as sp
import torch
import sys

path = r"/content/drive/MyDrive/GAT/GNNNeighbourhoodSampling"
sys.path.append(path)
from graphsage.utils2 import load_wikics, load_ppi, load_cora, custom_load_pubmed, load_ogbn_arxiv


def run_data(str):
    lsh_helper = {'n_vectors': 16, 'search_radius': 2, 'num_lsh_neighbours': 10, 'atleast': False, 'includeNeighbourhood': False}
    print(f"Loading {str} dataset.")
    if str == 'wikics': out = load_wikics(lsh_helper=lsh_helper)
    if str == 'ppi': out = load_ppi(lsh_helper=lsh_helper)
    if str == 'cora': out = load_cora(lsh_helper=lsh_helper)
    if str == 'pubmed': out = custom_load_pubmed(lsh_helper=lsh_helper)
    if str == 'arxiv': out = load_ogbn_arxiv(lsh_helper=lsh_helper)
    
    if str in ['cora', 'pubmed', 'citesser']: features = normalize_features(out['feat_data'])
    else: features = out['feat_data']

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(out['labels'])
    adj_lists = out['adj_lists']
    for node in adj_lists: adj_lists[node].add(node)
    
    nodes = torch.arange(labels.shape[0])
    train_mask = nodes[out['train_mask']]
    test_mask = nodes[out['test_mask']]
    val_mask = nodes[out['val_mask']]

    idx_train = torch.LongTensor(train_mask)
    idx_val = torch.LongTensor(val_mask)
    idx_test = torch.LongTensor(test_mask)

    return features, adj_lists, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    zero_locs = (rowsum == 0.)
    rowsum[zero_locs] = [1.]
    zero_locs = zero_locs.flatten()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[zero_locs] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx