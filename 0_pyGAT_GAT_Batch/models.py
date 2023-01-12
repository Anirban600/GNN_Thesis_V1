import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, special):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.special = special

        self.attentions = [GraphAttentionLayer(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions): self.add_module('attention_{}'.format(i), attention)


        if self.special:
            self.out_att = [GraphAttentionLayer(nhid * nheads,
                                                nclass,
                                                dropout=dropout,
                                                alpha=alpha,
                                                concat=False) for _ in range(nheads)]
            for i, out in enumerate(self.out_att): self.add_module('out_att_{}'.format(i), out)

        else: self.out_att = GraphAttentionLayer(nhid * nheads,
                                                 nclass,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.special: 
            res = torch.zeros((x.shape[0], 3), dtype=torch.float, device=torch.device('cuda'))
            for att in self.out_att: res += att(x, adj)
            x = F.elu(res)
        else:
            x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


















