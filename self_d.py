import argparse
from networkx.exception import PowerIterationFailedConvergence
import numpy as np
import copy
import torch
import time
import torch.nn.functional as F
from data.utils import load_data
from model import PLP, HAN, NS_ec, Feat_Ts
from trainer import Trainer
import copy
from utils import my_loss
from sklearn.metrics import roc_auc_score
from models.MAGNN import MAGNN_nc
from sklearn.metrics import f1_score
import dgl


parser = argparse.ArgumentParser('Main')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--model', type=str, default='MAGNN')

parser.add_argument('--label_ratio', type=int, default=20, 
                    help='train labels per class [20, 50]')

parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate')#0.01

parser.add_argument('--wd', type=float, default=0.0005, 
                    help='weight decay')#0.0005

parser.add_argument('--optimizer', type=str, default='adam', 
                    help='Optimizer.')

parser.add_argument('--num_heads', type=list, default=[8])
parser.add_argument('--dropout', type=float, default=0.6)#teacher dropout0.6
parser.add_argument('--hid_dim',type=int, default=64)#teacher hidden dim 12
parser.add_argument('--epoch', type=int, default=200)

args = parser.parse_args()


opt = vars(args)
epoches = opt['epoch']
device = opt['device']
dataset = opt['dataset']
ratio = opt['label_ratio']
lr = opt['lr']
wd = opt['wd']
num_layers = 2
if dataset == 'ACM':
    type_num = [4019, 7167, 60]
    sample_rate = [3, 1]
    nei_num = 2

if dataset =='DBLP':
    type_num = [4057, 14328, 7723, 20]
    sample_rate = [4]
    nei_num = 1

if dataset =='IMDB':
    type_num = [4278, 2081, 5257]
    sample_rate = [1, 3]
    nei_num = 2
    etypes_lists = [[[0, 1], [2, 3]]]


nx_G_lists, edge_metapath_indices_lists, G, nei_idx, feats, adjs, labels, idx_train, idx_val, idx_test, labels_one_hot, labels_init, type_mask \
    = load_data(dataset, ratio, type_num, device)

feats_dim_list = [i.shape[1] for i in feats]
num_classes = labels_one_hot.shape[1]
features = feats[0].to(device)
edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]

in_dims = [feat.shape[1] for feat in feats]
target_node_indices = np.where(type_mask == 0)[0]

opt['feats_dim_list'] = feats_dim_list
opt['num_meta_paths'] = len(G)
opt['in_features'] = features.shape[1]
opt['num_class'] = num_classes
opt['activation']=F.relu
opt['node_num'] = features.shape[0]
opt['labels_one_hot'] = labels_one_hot

g_lists = []
for nx_G_list in nx_G_lists:
    g_lists.append([])
    for nx_G in nx_G_list:
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(nx_G.number_of_nodes())
        g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
        g = g.to(device)
        g_lists[-1].append(g)

num_heads = 1
attn_vec_dim = 128

loss_fcn = torch.nn.CrossEntropyLoss()
#model = HAN(opt, G)
model = MAGNN_nc(num_layers, [2], 4, etypes_lists, in_dims, opt['hid_dim'], opt['num_class'], num_heads, attn_vec_dim,
                       rnn_type = 'RotatE0', dropout_rate = 0.5)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

pretrain_outputs = np.load('outputs/predsMAGNN'+ str(dataset) + str(ratio) + '.npy')
soft_target = torch.FloatTensor(pretrain_outputs).to(device)
idx_no_train = torch.LongTensor(np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(device)
def evaluate(idx_val, idx_test, target, feats):
    model.eval()
    with torch.no_grad():
        logits, _ = model((g_lists, feats, type_mask, edge_metapath_indices_lists), target_node_indices)
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
    #print('Epoch {:d} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC {:.4f}'.format(epoch + 1, loss_val.item(), val_micro_f1, val_macro_f1, auc_val.item()))

    return logits, [(auc_val.item(), auc_test.item())], [(val_micro_f1.item(), test_micro_f1.item())], [(val_macro_f1.item(), test_macro_f1.item())]

best = [0, 0.0, 0.0, 0.0, 0.0]
best_test = [0, 0.0, 0.0, 0.0, 0.0]

for epoch in range(epoches):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    #logits = model(features)
    logits, _ = model((g_lists, feats, type_mask, edge_metapath_indices_lists), target_node_indices)
    #logp = torch.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim = -1)
    logp = torch.exp(logp)
    loss = F.kl_div(logp[idx_no_train], soft_target[idx_no_train], reduction='batchmean') + loss_fcn(logp[idx_train], labels[idx_train])
    #-torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
    loss.backward(retain_graph=True)
    optimizer.step()
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
print(preds)
