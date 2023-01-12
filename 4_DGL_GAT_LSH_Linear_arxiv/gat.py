#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
import sys
import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn.metrics import f1_score

path = r"/content/drive/MyDrive/HonsProject/GNNNeighbourhoodSampling"
sys.path.append(path)
from graphsage.utils2 import load_wikics, load_ppi, load_cora, custom_load_pubmed, load_ogbn_arxiv, create_graph_from_representation


from models import GAT

epsilon = 1 - math.log(2)

device = None

dataset = None
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


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


def load_data(dataset, aug_khop=False, lsh_add=False):
    global n_node_feats, n_classes

    lsh_helper = {'n_vectors': 16, 'search_radius': 2, 'num_lsh_neighbours': 10, 'atleast': False, 'includeNeighbourhood': False}
    print(f"Loading {dataset} dataset.")
    if dataset == 'wikics': out = load_wikics(lsh_helper=lsh_helper, augment_khop=aug_khop, load_embeds=aug_khop)
    if dataset == 'ppi': out = load_ppi(lsh_helper=lsh_helper, augment_khop=aug_khop, load_embeds=aug_khop)
    if dataset == 'cora': out = load_cora(lsh_helper=lsh_helper, planetoid=True, augment_khop=aug_khop, load_embeds=aug_khop)
    if dataset == 'pubmed': out = custom_load_pubmed(lsh_helper=lsh_helper, planetoid=True, augment_khop=aug_khop, load_embeds=aug_khop)
    if dataset == 'arxiv': out = load_ogbn_arxiv(lsh_helper=lsh_helper, augment_khop=aug_khop, load_embeds=aug_khop)
    if dataset == 'cifar10-mobnet':
        out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/output_complete.pickle",
                                               threshold=0.95,
                                               lsh_helper = lsh_helper,
                                               train_mask = [True] * 45000 + [False] * 15000,
                                               test_mask = [False] * 50000 + [True] * 10000,
                                               val_mask = [False] * 45000 + [True] * 5000 + [False] * 10000)
    if dataset == 'cifar10-gnet':
        out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/gnet_cifar10_emb_complete.pickle",
                                               threshold=0.95,
                                               lsh_helper = lsh_helper,
                                               train_mask = [True] * 45000 + [False] * 15000,
                                               test_mask = [False] * 50000 + [True] * 10000,
                                               val_mask = [False] * 45000 + [True] * 5000 + [False] * 10000)
    if dataset == 'fmnist':
        out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/fmnist_emb_complete.pickle",
                                               threshold=0.6,
                                               lsh_helper = lsh_helper,
                                               train_mask = [True] * 55000 + [False] * 15000,
                                               test_mask = [False] * 60000 + [True] * 10000,
                                               val_mask = [False] * 55000 + [True] * 5000 + [False] * 10000)


    features = torch.FloatTensor(out['feat_data'])
    labels = torch.LongTensor(out['labels']).reshape(-1, 1)
    nodes = torch.arange(labels.shape[0])

    train_mask = nodes[out['train_mask']]
    test_mask = nodes[out['test_mask']]
    val_mask = nodes[out['val_mask']]

    train_idx = torch.LongTensor(train_mask)
    val_idx = torch.LongTensor(val_mask)
    test_idx = torch.LongTensor(test_mask)

    # Real Graph
    adj_lists = out['adj_lists']
    edges = set()
    for node in adj_lists:
        for neigh in adj_lists[node]:
            edges.add((node, neigh))

    feat_g = None
    # Augmented Graph
    if aug_khop:
        feat_graph = out['lsh_neighbour_list']
        if not lsh_add:
            print("Creating LSH Stack Augmented Graph")
            feat_edges = set()
            for node in feat_graph:
                for neigh in feat_graph[node]:
                    feat_edges.add((node, neigh))
            feat_src, feat_dst = [], []
            for e in feat_edges:
                feat_src.append(e[0])
                feat_dst.append(e[1])
            feat_src = torch.tensor(feat_src)
            feat_dst = torch.tensor(feat_dst)

            feat_g = dgl.graph((feat_src, feat_dst))
            feat_g.ndata['feat'] = features
            feat_g = feat_g.int().to(device)

            # add self loop
            feat_g = dgl.remove_self_loop(feat_g)
            feat_g = dgl.add_self_loop(feat_g)
            feat_n_edges = feat_g.number_of_edges()

            # with open(r'/content/drive/MyDrive/GAT/GAT Embeddings/Emb_cora_khop-False_add-False.pickle', 'rb') as f:
            #     emb = pickle.load(f)
            # embedding = emb['embeddings'].detach()
        else:
            print("Creating LSH Add Augmented Graph")
            for node in feat_graph:
                for neigh in feat_graph[node]:
                    edges.add((node, neigh))
    
    src, dst = [], []
    for e in edges:
        src.append(e[0])
        dst.append(e[1])
    src = torch.tensor(src)
    dst = torch.tensor(dst)

    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g.ndata['feat'] = features
    g = g.int().to(device)

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    num_feats = n_node_feats = features.shape[1]
    n_classes = len(set(labels.flatten().tolist()))
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           len(train_mask),
           len(val_mask),
           len(test_mask)))

    if(aug_khop and not lsh_add):
        print("""----Augmented Data statistics------'
        #Edges %d
        #Classes %d""" %
            (feat_n_edges, n_classes))

    features = features.to(device)
    labels = labels.to(device)
    hot_labels = F.one_hot(labels).reshape(-1, n_classes).double()

    evaluator = None

    return g, feat_g, labels, train_idx, val_idx, test_idx, evaluator


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats
    types = [None] + ([args.agg_type] * (args.n_layers-1)) + [args.out_agg_type]
    model = GAT(
        n_node_feats_,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
        stacking=args.aug_khop and not args.lsh_add,
        types=types,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator):
    model.train()

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]
        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred, _ = model(graph, feat, feat_graph)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, _ = model(graph, feat, feat_graph)

    loss = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    if evaluator: return evaluator(pred[train_idx], labels[train_idx]), loss.item()
    else: return f1_score(labels[train_idx].flatten().cpu().numpy(), torch.argmax(pred[train_idx], dim=1).cpu().numpy(), average="micro"), loss.item()

@torch.no_grad()
def fake(args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]
        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_pred_idx = train_idx[mask]

    _, pred = model(graph, feat, feat_graph)

    return pred

@torch.no_grad()
def evaluate(args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred, _ = model(graph, feat, feat_graph)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, _ = model(graph, feat, feat_graph)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    if not evaluator:
        f1_train = f1_score(labels[train_idx].flatten().cpu().numpy(), torch.argmax(pred[train_idx], dim=1).cpu().numpy(), average="micro")
        f1_val = f1_score(labels[val_idx].flatten().cpu().numpy(), torch.argmax(pred[val_idx], dim=1).cpu().numpy(), average="micro")
        f1_test = f1_score(labels[test_idx].flatten().cpu().numpy(), torch.argmax(pred[test_idx], dim=1).cpu().numpy(), average="micro")

    return (
        evaluator(pred[train_idx], labels[train_idx]) if evaluator else f1_train,
        evaluator(pred[val_idx], labels[val_idx]) if evaluator else f1_val,
        evaluator(pred[test_idx], labels[test_idx]) if evaluator else f1_test,
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def run(args, graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    best_epoch = 0
    if evaluator:
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]
    else: evaluator_wrapper = None

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, count, best_val_loss = 0, 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator_wrapper)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            count = 0
            best_epoch = epoch
            torch.save(model.state_dict(), 'es_checkpoint.pt')
        else: 
            count += 1

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"{'-' * 40}\n"
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}  "
                f"Patience {count} / {args.patience}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}   "
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)
        
        if args.early_stopping and args.patience == count:
            break

    model.load_state_dict(torch.load('es_checkpoint.pt'))

    best_train_acc, best_val_acc, best_test_acc, train_loss, val_loss, test_loss, pred = evaluate(
        args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
    )

    print("*" * 50)
    print(f"Best val acc: {round(best_val_acc, 4)}, Best test acc: {round(final_test_acc, 4)}")
    print("*" * 50)

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")
    
    if n_running == args.n_runs:
        pred_emb = fake(args, model, graph, feat_graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator_wrapper)
        with open(f'Emb_{args.dataset}_lsh_add-{args.lsh_add}_linear.pickle', 'wb') as f:
            pickle.dump(pred_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Embeddings dumped successfully...")
    
    return best_val_acc, final_test_acc, best_epoch


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon, dataset

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--dataset", type=str, default="cora", help="Dataset to load")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--aug_khop", action="store_true", help="Perform Khop LSH")
    argparser.add_argument("--lsh_add", action="store_true", help="Perform LSH Add")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.0, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--early_stopping", action="store_true", help="")
    argparser.add_argument("--patience", type=int, default=100, help="")
    argparser.add_argument("--agg_type", type=str, default="add", help="")
    argparser.add_argument("--out_agg_type", type=str, default="add", help="")
    args = argparser.parse_args()

    dataset = args.dataset
    aug_khop = args.aug_khop
    lsh_add = args.lsh_add

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    if aug_khop:
        if lsh_add:
            graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, aug_khop=True, lsh_add=True)
        else:
            graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, aug_khop=True, lsh_add=False)
    else:
        graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )
    if feat_graph: feat_graph.to(device)

    # run
    val_accs, test_accs, epochs = [], [], []

    for i in range(args.n_runs):
        seed(args.seed + i)
        val_acc, test_acc, best_epoch = run(args, graph, feat_graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        epochs.append(best_epoch)
        val_accs.append(round(val_acc, 4))
        test_accs.append(round(test_acc, 4))

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print("best Epochs:", epochs)
    print(f"Average val accuracy: {round(np.mean(val_accs), 4)} ± {round(np.std(val_accs), 3)}")
    print(f"Average test accuracy: {round(np.mean(test_accs), 4)} ± {round(np.std(test_accs), 3)}")
    print(f"Average best epoch: {int(round(np.mean(epochs), 0))}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()