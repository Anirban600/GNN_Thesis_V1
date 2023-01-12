"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
from cgat import CGATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 graph_margin,
                 class_margin,
                 top_k,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(CGATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, graph_margin, class_margin, top_k, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(CGATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(CGATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, None))
        else:
            self.gat_layers.append(CGATConv(
                in_dim, num_classes, heads[0],
                feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, None))

    def forward(self, inputs, labels):
        h = inputs
        Lg_all = Lb_all = 0
        for l in range(self.num_layers):
            h, Lg, Lb = self.gat_layers[l](self.g, h, labels)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
            Lg_all += Lg
            Lb_all += Lb
        return h, Lg_all, Lb_all
