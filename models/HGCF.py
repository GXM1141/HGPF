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

class NS_ec(nn.Module):
    def __init__(
        self, 
        graph, 
        hidden_dim, 
        feat_dims, 
        dropout, 
        mlp_layers, 
        num_classes, 
        node_num
    ):
        super(NS_ec, self).__init__()
        self.graph = graph
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feat_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        #self.sc = SC_aggr(mlp_layers, hidden_dim, num_classes, dropout, node_num)    
        self.sc = NS_aggr(graph, mlp_layers, hidden_dim, num_classes, dropout, node_num)    

    def forward(self, features):  # p a s
        h_all = []
        for i in range(0, len(features)):
            h_all.append(self.feat_drop(self.fc_list[i](features[i])))
        h_all = torch.cat(h_all, dim=0)
        logits, att = self.sc(self.graph, h_all)
        return logits, att

class NS_aggr(nn.Module):
    def __init__(
        self, 
        graph, 
        mlp_layers, 
        hidden_dim, 
        num_classes, 
        dropout, 
        node_num
    ):
        super(NS_aggr, self).__init__()
        self.g = graph
        self.mlp = MLP2(mlp_layers, hidden_dim, hidden_dim, num_classes, dropout)
        self.alpha = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))
        self.node_num = node_num
        self.e = nn.Parameter(torch.ones(graph.num_edges(), device=graph.device))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.alpha.data, gain=gain)
        self.act = None

    def forward(self, graph, h):
        with graph.local_scope():
            h_prime = []
            
            graph.srcdata.update({'ft': torch.softmax(self.mlp(h), dim=-1)})
            graph.edata['e'] = self.e
            e = graph.edata.pop('e')
            graph.edata['a'] = edge_softmax(graph, e)
            
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                            fn.sum('m', 'ft'))
            h_prime.append(graph.ndata['ft'])
            nei_cls = torch.cat(h_prime, dim=1)[:self.node_num]
        self_cls = torch.softmax(self.mlp(h[:self.node_num]), dim=-1)
        out = torch.sigmoid(self.alpha) * self_cls + torch.sigmoid(-self.alpha) * nei_cls
        if self.act is not None:
            logits = self.act(out)
        else:
            logits = out
        return logits, self.alpha
class SC_aggr(nn.Module):
    def __init__(
        self, 
        mlp_layers, 
        hidden_dim, 
        num_classes, 
        dropout, 
        node_num
    ):
        super(SC_aggr, self).__init__()
        self.mlp = MLP2(mlp_layers, hidden_dim, hidden_dim, num_classes, dropout)
        self.alpha = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))
        self.node_num = node_num
        gain = nn.init.calculate_gain('relu')
        self.attention = nn.Parameter(torch.FloatTensor(size = (self.node_num, 5, 1)))
        nn.init.xavier_normal_(self.alpha.data, gain=gain)
        nn.init.xavier_normal_(self.attention.data, gain=gain)
        self.act = None

    def forward(self, sc_inst, feats):
        sc_preds = []
        for sc in sc_inst:
            sc_nei = torch.cat(sc, dim=0).cuda()
            sc_emb = F.embedding(sc_nei, feats)
            sc_cls = F.log_softmax(self.mlp(sc_emb), dim = -1)
            sc_cls = torch.exp(sc_cls)
            sc_cls = torch.mean(sc_cls, dim = 1)
            sc_preds.append(sc_cls)
        sc_preds = [sc_pred.unsqueeze(dim=-1) for sc_pred in sc_preds]
        x = torch.cat(sc_preds, dim=-1)
        scheme_pred = torch.matmul(x, F.softmax(self.attention, dim = 1)).squeeze(2)
        self_pred = F.log_softmax(self.mlp(feats[:self.node_num]), dim = -1)
        self_pred = torch.exp(self_pred)
        logits = torch.sigmoid(self.alpha) * self_pred + torch.sigmoid(-self.alpha) * scheme_pred
        #logits = 0.5 * self_pred + 0.5 * scheme_pred
        return logits
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

class HGCF(nn.Module):
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
        super(HGCF, self).__init__()
        self.g = g
        self.graph = graph
        self.byte_idx_train = byte_idx_train
        self.labels_one_hot = labels_one_hot
        self.num_classes = num_classes
        self.label_init = labels_init
        self.masked = self.byte_idx_train #n*k的矩阵，其中已知label的train——idx的每一行全是1
        self.masked_label = (~self.masked).float()
        self.mlp = MLP2(mlp_layers, features[0].shape[1], hidden_dim, num_classes, dropout)
        self.masked_labels_one_hot = torch.mul(self.labels_one_hot, self.masked)#train_idx的label，其他行是0
        self.alpha = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))
        self.attention = nn.Parameter(torch.FloatTensor(size = (node_num, len(g), 1)))
        self.NS_ec = NS_ec(graph, hidden_dim, feat_dims, dropout, mlp_layers, num_classes, node_num)
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
                h = torch.mul(h, self.masked_label) + self.masked_labels_one_hot#带标签的点标签不变，其他点+h
            gt.append(g)
            h_prime[i] = h
        
        h_prime = [hp.unsqueeze(dim=-1) for hp in h_prime]
        x = torch.cat(h_prime, dim=-1)

        logits_lp = torch.matmul(x, F.softmax(self.attention, dim = 1)).squeeze(2)
        #logits_ns, att_NS =  self.NS_ec(features)
        logits_ns = self.mlp(features[0])
        logits = torch.sigmoid(self.alpha) * logits_lp + torch.sigmoid(-self.alpha) * logits_ns
        #att_target = torch.sigmoid(att_NS)
        #att_context = torch.sigmoid(-att_NS)
        att_global = torch.sigmoid(self.alpha)
        att_local = torch.sigmoid(-self.alpha)
        att_mp = F.softmax(self.attention, dim = 1)
        """
        att_local_mean = torch.mean(att_local, dim = 0)
        att_local_max = torch.topk(att_local, 10, dim = 0)
        att_global_mean = torch.mean(att_global, dim = 0)
        att_global_max = torch.topk(att_global, 10, dim = 0)
        att_target_mean = torch.mean(att_target, dim = 0)
        att_target_max = torch.topk(att_target, 30, dim = 0)
        att_context_mean = torch.mean(att_context, dim = 0)
        att_context_max = torch.topk(att_context, 30, dim = 0)
        att_mp_mean = torch.mean(att_mp, dim = 0)
        att_mp_std = torch.std_mean(att_mp, dim = 0)
        att_mp_max = torch.topk(att_mp, 10, dim = 0)
        """
        return logits, logits_lp, logits_ns, (0, 0), gt