import argparse
import numpy as np
import networkx as nx
import time
import pickle
import torch
import torch.nn.functional as F
import dgl
import sys
from dgl.data import register_data_args

from gat import GAT, GATStack
from utils import EarlyStopping

path = r"Merged_Model/GNNNeighbourhoodSampling"
sys.path.append(path)
from graphsage.planetoid import load_data
from graphsage.utils2 import load_cora



def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


######################################################   Load Data   #############################################################

def main(args):
    for i in range(args.n_iter):
        print(f"Loading {args.dataset} dataset.")
        if args.dataset == 'cora': adj_lists, feat_data, labels, train_mask, val_mask, test_mask = load_data('cora')
        elif args.dataset == 'pubmed': adj_lists, feat_data, labels, train_mask, val_mask, test_mask = load_data('pubmed')
        # if dataset == 'cifar10-mobnet':
        #     out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/output_complete.pickle",
        #                                            threshold=0.95,
        #                                            lsh_helper = lsh_helper,
        #                                            train_mask = [True] * 45000 + [False] * 15000,
        #                                            test_mask = [False] * 50000 + [True] * 10000,
        #                                            val_mask = [False] * 45000 + [True] * 5000 + [False] * 10000)
        # if dataset == 'cifar10-gnet':
        #     out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/gnet_cifar10_emb_complete.pickle",
        #                                            threshold=0.95,
        #                                            lsh_helper = lsh_helper,
        #                                            train_mask = [True] * 45000 + [False] * 15000,
        #                                            test_mask = [False] * 50000 + [True] * 10000,
        #                                            val_mask = [False] * 45000 + [True] * 5000 + [False] * 10000)
        # if dataset == 'fmnist':
        #     out = create_graph_from_representation(path_file = "/content/drive/MyDrive/GNN_On_Image/fmnist_emb_complete.pickle",
        #                                            threshold=0.6,
        #                                            lsh_helper = lsh_helper,
        #                                            train_mask = [True] * 55000 + [False] * 15000,
        #                                            test_mask = [False] * 60000 + [True] * 10000,
        #                                            val_mask = [False] * 55000 + [True] * 5000 + [False] * 10000)

        features = torch.FloatTensor(feat_data)
        labels = torch.LongTensor(labels)
        nodes = torch.arange(labels.shape[0])
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        src, dst = [], []
        for node in adj_lists:
            src += [node] * len(adj_lists[node])
            dst += list(adj_lists[node])

        src = torch.tensor(src)
        dst = torch.tensor(dst)

        g = dgl.graph((src, dst))
        g.ndata['feat'] = features
        g = g.int().to(args.gpu)

        # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()

        num_feats = features.shape[1]
        n_classes = len(set(labels.flatten().tolist()))
        if not args.print_less :
            print("""----Data statistics------
              > Edges %d
              > Classes %d
              > Train samples %d
              > Val samples %d
              > Test samples %d\n""" %
                  (n_edges, n_classes,
                   train_mask.int().sum().item(),
                   val_mask.int().sum().item(),
                   test_mask.int().sum().item()))

        features = features.to(args.gpu)

        # Replacve label with noisy label
        if args.noisy_label:
            with open(args.noisy_label, 'rb') as f: noise_labels = pickle.load(f)
            labels = torch.LongTensor(noise_labels).to(args.gpu)


#################################################  Model First Phase  #########################################################

    # create model
        start = time.time()
        heads = ([args.num_heads_stage_1] * (args.num_layers_stage_1-1)) + [args.num_out_heads_stage_1]
        model = GAT(g,
                args.num_layers_stage_1,
                num_feats,
                args.num_hidden_stage_1,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
        if not args.print_less : print(model)

        if args.early_stop: stopper = EarlyStopping(patience=100)
        if cuda: model.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        for epoch in range(args.epochs_stage_1):
            model.train()
            if epoch >= 3:
                if cuda : torch.cuda.synchronize()
            
            logits, _ = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = accuracy(logits[train_mask], labels[train_mask])
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model, epoch, args.print_less): break

            if not args.print_less : print(f"Epoch: {epoch:3d} | Train loss : {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | ", end="")

        if not args.print_less : print()
        if args.early_stop: model.load_state_dict(torch.load('es_checkpoint.pt'))
        train_acc = evaluate(model, features, labels, train_mask)
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        time_taken = round(time.time() - start, 4)
        best_epoch = epoch - 100
        
        if not args.print_less : print('-' * 100)
        print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f} | Time Taken: {time_taken} | Epochs Taken: {best_epoch}")
        if not args.print_less : print('-' * 100)

        # Saving model embedding
        _, out_emb = model(features)
        data = {'embeddings': out_emb, 'labels': labels.cpu().numpy()}
        with open("Merged_Model/Embedding_dump.pickle", 'wb') as f: pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    

#####################################################   Create Augmented Graph   ####################################################################################

        lsh_helper = {'n_vectors': 16, 'search_radius': 2, 'num_lsh_neighbours': 10, 'atleast': False, 'includeNeighbourhood': False}
        if args.dataset == 'cora': out = load_cora(lsh_helper=lsh_helper, embedding=out_emb, augment_khop=True, load_embeds=True, print_less=args.print_less)
        elif args.dataset == 'pubmed': out = custom_load_pubmed(lsh_helper=lsh_helper, embedding=out_emb, planetoid=True, augment_khop=True, load_embeds=True, print_less=args.print_less)

        features = torch.FloatTensor(out['feat_data'])

        # Real Graph
        adj_lists = out['adj_lists']
        edges = set()
        for node in adj_lists:
            for neigh in adj_lists[node]:
                edges.add((node, neigh))

        feat_g = None
        # Augmented Graph
        feat_graph = out['lsh_neighbour_list']
        if not lsh_add:
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
            feat_g = feat_g.int().to(args.gpu)

            # add self loop
            feat_g = dgl.remove_self_loop(feat_g)
            feat_g = dgl.add_self_loop(feat_g)
            feat_n_edges = feat_g.number_of_edges()
        else:
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
        g.ndata['feat'] = features
        g = g.int().to(args.gpu)

        # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()

        num_feats = features.shape[1]
        n_classes = len(set(labels.flatten().tolist()))

        if not args.print_less and not lsh_add:
            print("""----Augmented Data statistics------
            > Edges %d
            > Classes %d\n""" %
                (feat_n_edges, n_classes))

        features = features.to(args.gpu)
    
    
#################################################  Model Second Phase  #########################################################


    # create model
        heads = ([args.num_heads_stage_2] * (args.num_layers_stage_2-1)) + [args.num_out_heads_stage_2]
        types = [None] + ([args.agg_type] * (args.num_layers_stage_2-1)) + [args.out_agg_type]
        start = time.time()
        if not lsh_add:
            model = GATStack(g,
                    feat_g, 
                    args.num_layers_stage_2,
                    num_feats,
                    args.num_hidden_stage_2,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual,
                    types)
        else:
            model = GAT(g,
                    args.num_layers_stage_2,
                    num_feats,
                    args.num_hidden_stage_2,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
        
        if not args.print_less : print(model)

        if args.early_stop: stopper = EarlyStopping(patience=100)
        if cuda: model.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        for epoch in range(args.epochs_stage_2):
            model.train()
            if epoch >= 3:
                if cuda : torch.cuda.synchronize()

            # forward
            logits, _ = model(features)
            loss = loss_fcn(logits[train_mask].cuda(), labels[train_mask].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = accuracy(logits[train_mask].cuda(), labels[train_mask].cuda())
            val_acc = evaluate(model, features, labels.cuda(), val_mask.cuda())
            if args.early_stop:
                if stopper.step(val_acc, model, epoch, args.print_less): break

            if not args.print_less : print(f"Epoch: {epoch:3d} | Train loss : {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | ", end="")

        if not args.print_less : print()
        if args.early_stop: model.load_state_dict(torch.load('es_checkpoint.pt'))
        train_acc = evaluate(model, features, labels.cuda(), train_mask.cuda())
        val_acc = evaluate(model, features, labels.cuda(), val_mask.cuda())
        test_acc = evaluate(model, features, labels.cuda(), test_mask.cuda())
        time_taken = round(time.time() - start, 4)
        best_epoch = epoch - 100

        if not args.print_less : print('-' * 100)
        print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f} | Time Taken: {time_taken} | Epochs Taken: {best_epoch}")
        print('=' * 125)
    
        
####################################################    Take Arguments    #########################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    
    # common parameters
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--n_iter", type=int, default=5, help="Number of Complete Iterations")
    parser.add_argument("--noisy_label", type=str, default="", help="Load noise labels generated by NRGNN")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2, help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False, help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False, help="skip re-evaluate the validation set")
    parser.add_argument('--print_less', action="store_true", default=False)
    
    # Stage 1 Parameters
    parser.add_argument("--epochs_stage_1", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--num_heads_stage_1", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads_stage_1", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers_stage_1", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden_stage_1", type=int, default=8, help="number of hidden units")
    
    # Stage 2 Parameters
    parser.add_argument("--epochs_stage_2", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--num_heads_stage_2", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads_stage_2", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers_stage_2", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden_stage_2", type=int, default=8, help="number of hidden units")
    parser.add_argument("--lsh_add", action="store_true", default=False, help="Perform LSH Add")
    parser.add_argument("--agg_type", type=str, default="cat", help="one of add, avg or cat")
    parser.add_argument("--out_agg_type", type=str, default="cat", help="one of add, avg or cat")

    # Depretiated Parameters
    # parser.add_argument('--more-train-data', action='store_true', default=False, help="")
    # parser.add_argument("--min_epochs", type=int, default=0, help="number of training epochs")
    
    args = parser.parse_args()
    cuda = args.gpu >= 0
    dataset = args.dataset
    lsh_add = args.lsh_add
    main(args)
