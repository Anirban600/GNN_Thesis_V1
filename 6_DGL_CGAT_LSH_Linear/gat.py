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
# from dgl.nn import GATConv
from cgat import CGATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 feat_g,
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
            #hidden Layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(CGATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(CGATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, None))
            self.gat_layers.append(nn.Linear(num_hidden * heads[-2], num_classes))
        else:
            self.gat_layers.append(CGATConv(
                in_dim, in_dim, heads[0],
                feat_drop, graph_margin, class_margin, top_k, negative_slope, residual, None))
            self.gat_layers.append(nn.Linear(in_dim, num_classes))

    def forward(self, inputs, label):
        h = inputs
        Lg_all, Lb_all = 0, 0
        for l in range(self.num_layers):
            h, Lg, Lb = self.gat_layers[l](self.g, h, label)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
            Lg_all += Lg
            Lb_all += Lb
        h_final = self.gat_layers[-1](h)
        return h_final, h, Lg_all, Lb_all



class GATStack(nn.Module):
    def __init__(self,
                 g,
                 feat_g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 types):
        super(GATStack, self).__init__()
        self.g = g
        self.feat_g = feat_g
        self.num_layers = num_layers
        self.activation = activation
        self.in_dim = in_dim
        self.types = types

        self.gat_layers = nn.ModuleList()
        self.s_gat_layers = nn.ModuleList()
        
        # input projection (no residual)
        self.gat_layers.append(CGATConv(
            in_dim, num_hidden, heads[0],
            feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
        
        self.s_gat_layers.append(CGATConv(
            in_dim, num_hidden, heads[0],
            feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))

        # middle layer is num_layers > 2
        for l in range(1, self.num_layers-1):
            if self.types[l] == 'avg':
                self.gat_layers.append(CGATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
                
                self.s_gat_layers.append(CGATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
            else:
                self.gat_layers.append(CGATConv(
                    num_hidden * heads[l-1] * 2, num_hidden, heads[l],
                    feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
                
                self.s_gat_layers.append(CGATConv(
                    num_hidden * heads[l-1] * 2, num_hidden, heads[l],
                    feat_drop=feat_drop, negative_slope=negative_slope, residual=False, activation=self.activation))

        # output Layer
        if self.types[-2] == 'avg':
            self.gat_layers.append(CGATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop=feat_drop, negative_slope=negative_slope, residual=residual, activation=None))
            
            self.s_gat_layers.append(CGATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop=feat_drop, negative_slope=negative_slope, residual=residual, activation=None))
        else:
            self.gat_layers.append(CGATConv(
                num_hidden * heads[-2] * 2, num_hidden * heads[-2], heads[-1],
                feat_drop=feat_drop, negative_slope=negative_slope, residual=residual, activation=None))
            
            self.s_gat_layers.append(CGATConv(
                num_hidden * heads[-2] * 2, num_hidden * heads[-2], heads[-1],
                feat_drop=feat_drop, negative_slope=negative_slope, residual=residual, activation=None))
        
        # Linear projection
        if self.types[-1] == 'avg':
            self.linear_layer = nn.Linear(num_hidden * heads[-2] * heads[-1], num_classes)
        else:
            self.linear_layer = nn.Linear(num_hidden * heads[-2] * heads[-1] * 2, num_classes)


    def forward(self, inputs, label):
        h = inputs
        Lg_all, Lb_all, s_Lg_all, s_Lb_all = 0, 0, 0, 0
        
        for l in range(self.num_layers):
            h_org, Lg, Lb = self.gat_layers[l](self.g, h, label)
            h_org = h_org.flatten(1)

            Lg_all += Lg
            Lb_all += Lb

            h_aug, Lg, Lb = self.s_gat_layers[l](self.feat_g, h, label)
            h_aug = h_aug.flatten(1)

            s_Lg_all += Lg
            s_Lb_all += Lb

            if self.types[l+1] == 'avg': h = (h_org + h_aug) / 2
            else: h = torch.cat((h_org, h_aug), 1)
        
        h_final = self.linear_layer(h)
        return h_final, h, Lg_all, Lb_all, s_Lg_all, s_Lb_all