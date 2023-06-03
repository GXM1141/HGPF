import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
import math
import dgl.function as fn
from dgl.nn.pytorch import GATConv

class MLP2(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(MLP2, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(self.dropout(x))
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](self.dropout(h)))
            return self.linears[self.num_layers - 1](self.dropout(h))



class PLPConv(nn.Module):
    def __init__(
        self, 
        attn_drop,  
        graph, 
        activation, 
        device
    ):
        super(PLPConv, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.e = [nn.Parameter(torch.FloatTensor(size=(graph[i].num_edges(), 1))).to(device) for i in range(len(graph))]
        self.activation = activation
        self.leakyrelu = nn.LeakyReLU(0.6)
        self.g = graph
        gain = nn.init.calculate_gain('relu')
        for w in self.e:
            nn.init.xavier_normal_(w.data, gain=gain)

    def forward(self, i, soft_label):
        graph = self.g[i]
       
        cog_label = soft_label
        graph.srcdata.update({'ft': cog_label})
        graph.edata['e'] = self.e[i]
        e = graph.edata.pop('e')

        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        att = graph.edata['a'].squeeze()
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        if self.activation:
            rst = self.activation(rst)
        return rst, graph

class CPF(nn.Module):
    def __init__(
        self, 
        g, 
        graph, 
        byte_idx_train, 
        labels_one_hot, 
        labels_init, 
        node_num, 
        features, 
        hidden_dim, 
        num_classes, 
        layers, 
        mlp_layers, 
        feat_dims, 
        attn_drop, 
        dropout, 
        activation, 
        device
    ):
        super(CPF, self).__init__()
        self.g = g
        self.graph = graph
        self.byte_idx_train = byte_idx_train
        self.labels_one_hot = labels_one_hot
        self.num_classes = num_classes
        self.label_init = labels_init
        self.masked = self.byte_idx_train #n*k的矩阵，其中已知label的train——idx的每一行全是1
        self.masked_label = (~self.masked).float()
        self.masked_labels_one_hot = torch.mul(self.labels_one_hot, self.masked)#train_idx的label，其他行是0
        self.alpha = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))
        self.attention = nn.Parameter(torch.FloatTensor(size = (node_num, len(g), 1)))
        self.mlp = MLP2(mlp_layers, features[0].shape[1], hidden_dim, num_classes, dropout)
        self.layers = layers
        self.plp_layers = nn.ModuleList()
        for _ in range(layers):
            self.plp_layers.append(PLPConv(attn_drop, g, activation, device))
        
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attention.data, gain=gain)
        #nn.init.kaiming_normal_(self.attention.data, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.alpha.data, gain=gain)
        
    def forward(self, features):
        h_prime = [torch.zeros_like(self.label_init) for _ in range(len(self.g))]
        gt = []
        for i in range(len(self.g)):
            h = self.label_init
            for plpConv in self.plp_layers:
                h, g= plpConv(i, h)
                # = torch.sigmoid(self.alpha) * h + torch.sigmoid(-self.alpha) * self.mlp(features[0])
                h = torch.mul(h, self.masked_label) + self.masked_labels_one_hot#带标签的点标签不变，其他点+h
            gt.append(g)
            h_prime[i] = h
        
        h_prime = [hp.unsqueeze(dim=-1) for hp in h_prime]
        x = torch.cat(h_prime, dim=-1)

        logits = torch.matmul(x, F.softmax(self.attention, dim = 1)).squeeze(2)
        logits = torch.sigmoid(self.alpha) * logits +torch.sigmoid(-self.alpha) * self.mlp(features[0])

        return logits, 1, 1, 1, 1