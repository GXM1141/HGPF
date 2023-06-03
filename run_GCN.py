from data.utils import normalize_adj_1
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import networkx as nx
from scipy import sparse
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from load_data import load_acm, load_dblp, load_imdb
import time
from models.GCN import GCN
from models.GAT import GAT

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)
epoches = 100
device = 'cpu'
dataset = 'imdb'
ratio = '50'
lr = 0.005
wd = 0.0000
layers = 3
in_feat = 64
hid_feat = 64
out_feat = 64
dropout = 0.5
activation = 'relu'
#GCN,acm20:0.005/0/2/256/128/0.5/relu
#GCN,acm50:0.005/0/2/128/64/0.6/relu
#GCN,dblp20:0.005/0/3/128/64/0.5/relu
#GCN,dblp50:0.005/0/3/128/64/0.5/relu
#GCN,imdb20:0.005/0/3/128/64/0.5/relu
#GCN,imdb50:0.005/0/3/128/64/0.5/relu
if dataset == 'acm':
    adj, feats, feat_dims, idx_train, idx_val, idx_test, num_classes, labels, g = load_acm(device, ratio)
if dataset == 'dblp':
    adj, feats, feat_dims, idx_train, idx_val, idx_test, num_classes, labels, g = load_dblp(device, ratio)
if dataset == 'imdb':
    adj, feats, feat_dims, idx_train, idx_val, idx_test, num_classes, labels, g = load_imdb(device, ratio)

#model = GCN(layers, feat_dims, in_feat, out_feat, num_classes, dropout, activation).to(device)
heads = [4]*2 + [4]*2 + [1]
slope = 0.05
model = GAT(
            g=g,
            feat_dims = feat_dims,
            in_dim=64,
            num_hidden=64,
            num_classes=num_classes,
            num_layers=3,
            activation=F.elu,
            feat_drop=0.7,
            attn_drop=0.3,
            heads=heads,
            negative_slope=slope,
            residual=False,
            sparse_input=True)

#GAT,acm20:0.005/0/3/128/128/0.7/0.3/elu
#GAT,acm50:0.005/0/3/64/64/0.7/0.3/elu
#GCN,dblp20:0.001/0/3/128/128/0.7/0.3/elu
#GCN,dblp50:0.001/0/3/128/64/0.5/relu
#GCN,imdb20:0.005/0/3/128/64/0.5/relu
#GCN,imdb50:0.005/0/3/128/64/0.5/relu
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def evaluate(idx_val, idx_test, target, feats):
    model.eval()
    with torch.no_grad():
        logits = model(feats)
    logp = torch.softmax(logits, dim=-1)
    loss_val = loss_fcn(logp[idx_val], target[idx_val])
    _, val_indices = torch.max(logits[idx_val], dim=1)
    val_prediction = val_indices.long().cpu().numpy()
    val_logp = logp[idx_val].float().cpu().numpy()
    t = target.long().cpu().numpy()
    auc_val = roc_auc_score(t[idx_val.long().cpu().numpy()], val_logp, multi_class='ovo')
    val_micro_f1 = f1_score(t[idx_val.long().cpu().numpy()], val_prediction, average='micro')
    val_macro_f1 = f1_score(t[idx_val.long().cpu().numpy()], val_prediction, average='macro')
    _, test_indices = torch.max(logits[idx_test], dim=1)
    test_prediction = test_indices.long().cpu().numpy()
    test_logp = logp[idx_test].float().cpu().numpy()
    auc_test = roc_auc_score(t[idx_test.long().cpu().numpy()], test_logp, multi_class='ovo')
    test_micro_f1 = f1_score(t[idx_test.long().cpu().numpy()], test_prediction, average='micro')
    test_macro_f1 = f1_score(t[idx_test.long().cpu().numpy()], test_prediction, average='macro')
    print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(epoch + 1, loss_val.item(), val_micro_f1, val_macro_f1, auc_val.item()))

    return logits, [(auc_val.item(), auc_test.item())], [(val_micro_f1.item(), test_micro_f1.item())], [(val_macro_f1.item(), test_macro_f1.item())]

best = [0, 0.0, 0.0, 0.0, 0.0]
best_test = [0, 0.0, 0.0, 0.0, 0.0]
loss_fcn = torch.nn.CrossEntropyLoss()
feats = [feat.to(device) for feat in feats]
for epoch in range(epoches):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()    
    logits = model(feats)
    #logp = torch.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim = -1)
    logp = torch.exp(logp)
    loss = loss_fcn(logp[idx_train], labels[idx_train])
    #-torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
    loss.backward(retain_graph=True)
    optimizer.step()
    #print (loss)
    val_logits, auc, micro_f1, macro_f1 = evaluate(idx_val, idx_test, labels, feats)
    acc_v, acc_t = micro_f1[0]
    macro_v, macro_t = macro_f1[0]
    auc_v, auc_t = auc[0]
    if acc_v > best[2]:
        best = [epoch + 1, loss, acc_v, macro_v, auc_v]
        best_test = [epoch+1, loss, acc_t, macro_t, auc_t]
        outputs = val_logits

print('Epoch {:d} | Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(best[0], best[1], \
        best[2], best[3], best[4]))

print('Epoch {:d} | Loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test AUC {:.4f}'.format(best_test[0], best_test[1], \
        best_test[2], best_test[3], best_test[4]))
logits = F.log_softmax(outputs, dim=-1)
logits = torch.exp(logits)
preds = logits.detach().cpu().numpy()

