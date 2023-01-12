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


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
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
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
            self.gat_layers.append(nn.Linear(num_hidden * heads[-2] * heads[-1], num_classes))
        else:
            self.gat_layers.append(GATConv(
                in_dim, in_dim, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))
            self.gat_layers.append(nn.Linear(in_dim, num_classes))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        h_final = self.gat_layers[-1](h)
        return h_final, h



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
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        
        self.s_gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # middle layer is num_layers > 2
        for l in range(1, self.num_layers-1):
            if self.types[l] in ['add', 'avg']:
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, False, self.activation))
                
                self.s_gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, False, self.activation))
            else:
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1] * 2, num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, False, self.activation))
                
                self.s_gat_layers.append(GATConv(
                    num_hidden * heads[l-1] * 2, num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, False, self.activation))

        # output Layer
        if self.types[-2] in ['add', 'avg']:
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
            
            self.s_gat_layers.append(GATConv(
                num_hidden * heads[-2], num_hidden * heads[-2], heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2] * 2, num_hidden * heads[-2], heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
            
            self.s_gat_layers.append(GATConv(
                num_hidden * heads[-2] * 2, num_hidden * heads[-2], heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        
        # Linear projection
        if self.types[-1] in ['add', 'avg']:
            self.linear_layer = nn.Linear(num_hidden * heads[-2] * heads[-1], num_classes)
        else:
            self.linear_layer = nn.Linear(num_hidden * heads[-2] * heads[-1] * 2, num_classes)


    def forward(self, inputs):
        h = inputs
        
        for l in range(self.num_layers):
            h_org = self.gat_layers[l](self.g, h)
            h_org = h_org.flatten(1)

            h_aug = self.s_gat_layers[l](self.feat_g, h)
            h_aug = h_aug.flatten(1)
        
            if self.types[l+1] == 'add': h = h_org + h_aug
            elif self.types[l+1] == 'avg': h = (h_org + h_aug) / 2
            else: h = torch.cat((h_org, h_aug), 1)
        
        h_final = self.linear_layer(h)
        return h_final, h