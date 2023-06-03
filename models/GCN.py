from audioop import bias
import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNConv(nn.Module):
    def __init__(self, in_feat, out_feat, bias = True):
        super(GCNConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.W = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.use_bias = bias
        if (bias):
            self.bias = nn.Parameter(torch.FloatTensor(out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, features, adj):
        support = torch.mm(features, self.W)
        out = torch.spmm(adj, support)
        if (self.use_bias):
            out = out + self.bias
        return out

class GCN(nn.Module):
    def __init__(self, layers, feat_dims, in_feat, out_feat, num_class, dropout, activation):
        super(GCN, self).__init__()
        self.layers = layers
        self.activation = activation
        assert layers >= 1
        if (layers == 1):
            self.Conv = [GCNConv(in_feat, out_feat)]
        else:
            self.Conv = [GCNConv(in_feat, 2*out_feat)]
            for i in range(layers - 2):
                self.Conv.append(GCNConv(2*out_feat, 2*out_feat))
            self.Conv.append(GCNConv(2*out_feat, out_feat))
        self.Conv = nn.ModuleList(self.Conv)
        self.fc_list = nn.ModuleList([nn.Linear(feat_dim, in_feat, bias=True)
                                      for feat_dim in feat_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.predict = nn.Linear(out_feat, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj):
        x = []
        for feat, fc in zip(X, self.fc_list):
            x.append(self.dropout(fc(feat)))
        x = torch.cat(x, dim=0)
        for conv in self.Conv:
            x = F.relu(conv(x, adj))
            #x = conv(x, adj)
        return self.predict(x)
