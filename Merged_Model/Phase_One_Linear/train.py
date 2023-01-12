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

from gat import GAT
from utils import EarlyStopping

path = r"GAT_Resources/GNNNeighbourhoodSampling"
sys.path.append(path)
from graphsage.utils2 import load_cora, custom_load_pubmed


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


#########################################################################################################################################

def main(args):
    print(f"Loading {args.dataset} dataset.")
    if args.dataset == 'cora': out = load_cora(lsh_helper={}, planetoid=True, augment_khop=False, load_embeds=False)
    elif args.dataset == 'pubmed': out = custom_load_pubmed(lsh_helper={}, planetoid=True, augment_khop=False, load_embeds=False)
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
    labels = torch.LongTensor(out['labels'])
    nodes = torch.arange(labels.shape[0])

    train_mask = torch.tensor(out['train_mask'])
    val_mask = torch.tensor(out['val_mask'])
    test_mask = torch.tensor(out['test_mask'])

    # Real Graph
    adj_lists = out['adj_lists']
    
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
    with open(args.noisy_label, 'rb') as f: noise_labels = pickle.load(f)
    labels = torch.LongTensor(noise_labels).to(args.gpu)


##################################################################################################################

    # create model
    all_train, all_val, all_test = [], [], []
    time_taken, epochs_taken = [], []
    best_test_acc = 0

    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    for i in range(args.n_iter):
        start = time.time()
        model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        for epoch in range(args.epochs):
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
        best_epoch = epoch - 100

        all_train.append(round(train_acc, 4))
        all_val.append(round(val_acc, 4))
        all_test.append(round(test_acc, 4))
        time_taken.append(round(time.time() - start, 4))
        epochs_taken.append(best_epoch)

        if not args.print_less : print('-' * 100)
        print(f"Iteration: {i+1}/{args.n_iter} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f} | Time Taken: {time_taken[-1]}")
        if not args.print_less : print('-' * 100)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')

    
    all_train, all_val, all_test, time_taken, epochs_taken = np.array(all_train), np.array(all_val), np.array(all_test), np.array(time_taken), np.array(epochs_taken)

    print()
    print("=" * 100)
    print(f"Train Accuracy :     \t{all_train}")
    print(f"Validation Accuracy :\t{all_val}")
    print(f"Test Accuracy :      \t{all_test}")
    print(f"Time Taken :         \t{time_taken}")
    print(f"Epochs Taken :       \t{epochs_taken}")
    print(f"Average Train Accuracy :     \t{all_train.mean():.4f} ± {all_train.std():.3f}")
    print(f"Average Validation Accuracy :\t{all_val.mean():.4f} ± {all_val.std():.3f}")
    print(f"Average Test Accuracy :      \t{all_test.mean():.4f} ± {all_test.std():.3f}")
    print(f"Average Time Taken :         \t{time_taken.mean():.4f}")
    print(f"Average Epochs Taken :       \t{epochs_taken.mean():.0f}")
    print("=" * 100)

    # Loading best model for embedding
    # model = GAT(g, args.num_layers, num_feats, args.num_hidden, n_classes, heads, F.elu, args.in_drop, args.attn_drop, args.negative_slope, args.residual)
    # if cuda: model.cuda()
    # model.load_state_dict(torch.load('best_model.pt'))
    _, output = model(features)
    data = {'embeddings': output, 'labels': labels.cpu().numpy()}
    with open(args.emb_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


########################################################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs")
    parser.add_argument('--more-train-data', action='store_true', default=False, help="")
    parser.add_argument("--min_epochs", type=int, default=0, help="number of training epochs")
    parser.add_argument("--n_iter", type=int, default=5, help="number of training iterations")
    parser.add_argument("--num-heads", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8, help="number of hidden units")
    parser.add_argument("--agg_type", type=str, default="cat", help="one of add, avg or cat")
    parser.add_argument("--noisy_label", type=str, default="", help="")
    parser.add_argument("--emb_path", type=str, default="", help="")
    parser.add_argument("--out_agg_type", type=str, default="cat", help="one of add, avg or cat")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2, help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False, help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False, help="skip re-evaluate the validation set")
    parser.add_argument('--print_less', action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    cuda = args.gpu >= 0
    dataset = args.dataset
    aug_khop = False
    lsh_add = False

    main(args)
