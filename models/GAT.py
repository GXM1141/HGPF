import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import GraphConv, GATConv
from functools import reduce


class GAT(nn.Module):
    def __init__(self,
                 g,
                 feat_dims,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop,
                 attn_drop,
                 heads,
                 negative_slope,
                 residual=False,
                 sparse_input=False):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.sparse_input = sparse_input
        if self.sparse_input:
            self.linear = nn.Linear(in_dim, num_hidden)
            in_dim = num_hidden
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        self.fc_list = nn.ModuleList([nn.Linear(feat_dim, in_dim, bias=True)
                                      for feat_dim in feat_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, X):
        x = []
        for feat, fc in zip(X, self.fc_list):
            x.append(fc(feat))
        inputs = torch.cat(x, dim=0)
        if self.sparse_input:
            h = self.linear(inputs)
        else:
            h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        logits = F.log_softmax(logits, dim=1)
        return logits