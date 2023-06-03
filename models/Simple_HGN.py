import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.GATConv import myGATConv

import torch.nn.functional as F




class SimpleHGN(nn.Module):

    def __init__(
        self,
        graph, 
        edge_dim,
        num_etypes,
        in_dim,
        feats_dim_list,
        num_hidden,
        num_classes,
        num_layers,
        heads,
        feat_drop,
        attn_drop,
        negative_slope,#0.05
        residual,
        alpha
    ):
        super(SimpleHGN, self).__init__()
        self.g = graph

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, in_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.dropout = nn.Dropout(feat_drop)
        self.gat_layers.append(
            myGATConv(
                edge_dim,
                num_etypes,
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        # hidden layers
        for l in range(1, num_layers):  # noqa E741
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                myGATConv(
                    edge_dim,
                    num_etypes,
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                )
            )
        # output projection
        self.gat_layers.append(
            myGATConv(
                edge_dim,
                num_etypes,
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                alpha=alpha,
            )
        )
        self.epsilon = torch.FloatTensor([1e-12])


    def forward(self, X):
        h = []
        for fc, feature in zip(self.fc_list, X):
            h.append(self.dropout(fc(feature)))
        h = torch.cat(h, dim = 0)
        res_attn = None
        for l in range(self.num_layers):  # noqa E741
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        # logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return logits, 1, 2, 3


