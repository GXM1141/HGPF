import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import PLPConv, MLP2
from dgl.nn.pytorch import GATConv
from layer import NS_aggr

class PLP(nn.Module):
    def __init__(
        self, 
        g, 
        byte_idx_train, 
        labels_one_hot, 
        labels_init, 
        nei_idx, 
        node_num, 
        features, 
        hidden_dim, 
        layers, 

    ):
        super(PLP, self).__init__()
        self.g = g
        self.opt = opt
        self.byte_idx_train = opt['byte_idx_train']
        self.labels_one_hot = opt['labels_one_hot']
        self.label_init = opt['label_init']
        self.nei_idx = opt['nei_idx']
        self.masked = self.byte_idx_train #n*k的矩阵，其中已知label的train——idx的每一行全是1
        self.masked_label = (~self.masked).float()
        self.masked_labels_one_hot = torch.mul(self.labels_one_hot, self.masked)#train_idx的label，其他行是0
        self.plp_layer = PLPConv(opt, self.g)
        self.alpha = nn.Parameter(torch.FloatTensor(size=(opt['node_num'], 1)))
        self.attention = nn.Parameter(torch.FloatTensor(size = (opt['node_num'], len(g), 1)))
        self.NS_ec = NS_ec(opt)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attention.data, gain=gain)
        nn.init.xavier_normal_(self.alpha.data, gain=gain)

    def forward(self, feats):
        h_prime = [torch.zeros_like(self.label_init) for _ in range(len(self.g))]
        for i in range(len(self.g)):
            h = self.label_init
            for l in range(self.opt['LP_layers']):
                h, att= self.plp_layer(i, feats, h)
                h = torch.mul(h, self.masked_label) + self.masked_labels_one_hot#带标签的点标签不变，其他点+h
            h_prime[i] = h
        
        h_prime = [hp.unsqueeze(dim=-1) for hp in h_prime]
        x = torch.cat(h_prime, dim=-1)
        logits_lp = torch.matmul(x, F.softmax(self.attention, dim = 1)).squeeze(2)
        logits_ns =  self.NS_ec(self.opt['feats'])
        logits = torch.sigmoid(self.alpha) * logits_lp + torch.sigmoid(-self.alpha) * logits_ns
        return logits, logits_lp, logits_ns

class NS_ec(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        nei_idx, 
        features, 
        dropout
    ):
        super(NS_ec, self).__init__()
        self.hidden_dim = opt['stu_hid']
        self.nei_idx = opt['nei_idx']
        self.features = opt['feats']
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.hidden_dim, bias=True)
                                      for feats_dim in opt['feats_dim_list']])

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if opt['feat_drop'] > 0:
            self.feat_drop = nn.Dropout(opt['feat_drop'])
        else:
            self.feat_drop = lambda x: x
        self.sc = NS_aggr(opt)    

    def forward(self, feats):  # p a s
        h_all = []
        for i in range(0, len(self.features)):
            h_all.append(self.feat_drop(self.fc_list[i](self.features[i])))
        logits = self.sc(h_all, self.nei_idx)
        return logits

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, opt, G):
        super(HAN, self).__init__()

        self.G = G 
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(opt['num_meta_paths'], opt['in_features'], opt['tea_hid'], opt['num_heads'][0], opt['tea_drop']))
        for l in range(1, len(opt['num_heads'])):
            self.layers.append(HANLayer(opt['num_meta_paths'], opt['tea_hid'] * opt['num_heads'][l-1],
                                        opt['tea_hid'], opt['num_heads'][l], opt['tea_drop']))
        self.predict = nn.Linear(opt['tea_hid'] * opt['num_heads'][-1], opt['num_class'])

    def forward(self, feats):
        for gnn in self.layers:
            feats = gnn(self.G, feats)
        
        return self.predict(feats)