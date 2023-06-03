import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

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
    def __init__(
        self, 
        num_paths,
        in_dim, 
        hid_dim, 
        num_heads, 
        dropout, 
        num_classes,  
        g
    ):
        super(HAN, self).__init__()
        self.g = g 
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_paths, in_dim, hid_dim, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_paths, hid_dim * num_heads[l-1],
                                        hid_dim, num_heads[l], dropout))
        self.predict = nn.Linear(hid_dim * num_heads[-1], num_classes)

    def forward(self, features):
        feats = features[0]
        for gnn in self.layers:
            feats = gnn(self.g, feats)
        
        return self.predict(feats), 1, 2, 3, 4

class HAN_AC(nn.Module):
    def __init__(
        self, 
        num_paths,
        in_dim, 
        hid_dim, 
        num_heads, 
        dropout, 
        num_classes,  
        g
    ):
        super(HAN, self).__init__()
        self.g = g 
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_paths, in_dim, hid_dim, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_paths, hid_dim * num_heads[l-1],
                                        hid_dim, num_heads[l], dropout))
        self.predict = nn.Linear(hid_dim * num_heads[-1], num_classes)

    def forward(self, features):
        feats = features[0]
        for gnn in self.layers:
            feats = gnn(self.g, feats)
        
        return self.predict(feats), 1, 2, 3, 4