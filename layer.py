import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
import math

class PLPConv(nn.Module):
    def __init__(self, opt, graph):
        super(PLPConv, self).__init__()
        self.attn_drop = nn.Dropout(opt['LP_attn_drop'])
        self.e = [nn.Parameter(torch.FloatTensor(size=(graph[i].num_edges(), 1))).to(opt['device']) for i in range(len(graph))]
        self.activation = opt['activation']
        self.leakyrelu = nn.LeakyReLU(0.5)
        self.g = graph
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)
        for w in self.e:
            reset(w)

    def forward(self, i, feat, soft_label):
        graph = self.g[i]
        with graph.local_scope():
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
            return rst, att


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

class NS_aggr(nn.Module):
    def __init__(self, opt):
        super(NS_aggr, self).__init__()
        self.sample_rate = opt['sample_rate']
        self.nei_num = opt['nei_num']
        self.mlp = MLP2(opt['mlp_layers'], opt['stu_hid'], opt['stu_hid'], opt['num_class'], opt['NS_drop'])
        self.alpha = nn.Parameter(torch.FloatTensor(size=(opt['node_num'], 1)))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.alpha.data, gain=gain)

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei.cpu(), sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei.cpu(), sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()#Sampling neighbors of the i-th type of point
            one_type_emb = F.embedding(sele_nei, nei_h[i + 1]) #node_num * (sample_num * feature_dim)
            one_type_cls = F.log_softmax(self.mlp(one_type_emb), dim = -1)
            one_type_cls = torch.exp(one_type_cls)
            one_type_cls = torch.mean(one_type_cls, dim = 1)
            embeds.append(one_type_cls)
        nei_cls = 0
        for embed in embeds:
            nei_cls += embed / len(embeds)
        node_cls = F.log_softmax(self.mlp(nei_h[0]), dim = -1)
        node_cls = torch.exp(node_cls)
        logits = torch.sigmoid(self.alpha) * node_cls + torch.sigmoid(-self.alpha) * nei_cls
        return logits