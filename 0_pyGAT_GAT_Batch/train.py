from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score

from utils import run_data, accuracy
from models import GAT


def train(epoch):
    t = time.time()
    model.train()
    nodes = idx_train.clone()
    for batch in range(0, nodes.shape[0], batch_size):
        optimizer.zero_grad()
        main_nodes = nodes[batch: batch + batch_size]
        all_nodes = set()
        if big_graph: 
            for node in main_nodes: 
                if len(adj_list[node.item()]) < 500: all_nodes = all_nodes.union(adj_list[node.item()])
                else: 
                    temp = np.array(list(adj_list[node.item()]))
                    np.random.shuffle(temp)
                    all_nodes = all_nodes.union(set(temp[:500]))
        else: 
            for node in main_nodes: all_nodes = all_nodes.union(adj_list[node.item()])
        aux_nodes = torch.tensor(list(all_nodes.difference(set(main_nodes.tolist()))), dtype=torch.int64)
        all_nodes = main_nodes.tolist() + aux_nodes.tolist()
        batch_feat = features[all_nodes]
        mapper = dict()
        for i, node in enumerate(all_nodes): mapper[node] = i
        batch_adj = np.zeros((len(all_nodes), len(all_nodes)))
        for node in all_nodes:
            for neigh in adj_list[node]:
                if neigh in mapper: batch_adj[mapper[node], mapper[neigh]] = 1
        batch_adj = torch.tensor(batch_adj)
        if args.cuda:
            batch_feat = batch_feat.cuda()
            batch_adj = batch_adj.cuda()
        output = model(batch_feat, batch_adj)
        loss_train = F.nll_loss(output[:main_nodes.shape[0]], labels[main_nodes])
        loss_train.backward()
        optimizer.step()

    if args.progress:
        loss_val = F.nll_loss(evaluate('val'), labels[idx_val])
        print('\rEpochs => {:04d}'.format(epoch+1), end='')

    else:
        train_output = evaluate('train')
        val_output = evaluate('val')
        loss_train = F.nll_loss(train_output, labels[idx_train])
        acc_train = accuracy(train_output, labels[idx_train])
        loss_val = F.nll_loss(val_output, labels[idx_val])
        acc_val = accuracy(val_output, labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'acc_train: {:.4f}'.format(acc_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'acc_val: {:.4f}'.format(acc_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


def evaluate(mode):
    model.eval()
    if mode == 'train': target = idx_train
    elif mode == 'val': target = idx_val
    else: target = idx_test
    nodes = target.clone()
    for batch in range(0, nodes.shape[0], batch_size):
        main_nodes = nodes[batch: batch + batch_size]
        all_nodes = set()
        if big_graph: 
            for node in main_nodes: 
                if len(adj_list[node.item()]) < 500: all_nodes = all_nodes.union(adj_list[node.item()])
                else: all_nodes = all_nodes.union(set(list(adj_list[node.item()])[:500]))
        else: 
            for node in main_nodes: all_nodes = all_nodes.union(adj_list[node.item()])
        aux_nodes = torch.tensor(list(all_nodes.difference(set(main_nodes.tolist()))), dtype=torch.int64)
        all_nodes = main_nodes.tolist() + aux_nodes.tolist()
        batch_feat = features[all_nodes]
        mapper = dict()
        for i, node in enumerate(all_nodes): mapper[node] = i
        batch_adj = np.zeros((len(all_nodes), len(all_nodes)))
        for node in all_nodes:
            for neigh in adj_list[node]:
                if neigh in mapper: batch_adj[mapper[node], mapper[neigh]] = 1
        batch_adj = torch.tensor(batch_adj)
        if args.cuda:
            batch_feat = batch_feat.cuda()
            batch_adj = batch_adj.cuda()
        output = model(batch_feat, batch_adj).detach()
        if batch == 0: final_output = output[:main_nodes.shape[0]]
        else: final_output = torch.cat((final_output, output[:main_nodes.shape[0]]), dim=0)
    return final_output

def compute_test():
    output = evaluate('test')
    loss_test = F.nll_loss(output, labels[idx_test])
    macro = f1_score(y_true=labels[idx_test].cpu(), y_pred=output.argmax(dim=1).cpu(), average='macro')
    micro = f1_score(y_true=labels[idx_test].cpu(), y_pred=output.argmax(dim=1).cpu(), average='micro')
    acc_test = accuracy(output, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "macro= {:.4f}".format(macro),
          "micro= {:.4f}".format(micro))


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Pass the gataset name')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--progress', action='store_true', default=False, help='Suppress Epoch details')
parser.add_argument('--special', action='store_true', default=False, help='Pubmed special')
parser.add_argument('--batch_size', type=int, default=256, help='No of nodes per batch.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
batch_size = args.batch_size
features, adj_list, labels, idx_train, idx_val, idx_test = run_data(args.dataset)
big_graph = True if features.shape[0] > 30000 else False
print(f"Train samples: {idx_train.shape[0]} | Val samples: {idx_val.shape[0]} | Test samples: {idx_test.shape[0]}")

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha,
            special=args.special)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model = model.cuda()
    labels = labels.cuda()

features, labels = Variable(features), Variable(labels.flatten())

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else: bad_counter += 1

    if bad_counter == args.patience: break

print("\nOptimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Restore best model
print('Loading {}th epoch'.format(best_epoch))
# Testing
compute_test()
