import argparse
import random
from networkx.exception import PowerIterationFailedConvergence
import numpy as np
import copy
import torch
import time
import torch.nn.functional as F
from data.process_acm import load_acm, load_acm_simplehgn, load_acm_ycm, load_acm_magnn, load_acm_hgt
from data.process_dblp import load_dblp, load_dblp_simplehgn, load_dblp_ycm, load_dblp_magnn, load_dblp_hgt
from data.process_imdb import load_imdb, load_imdb_simplehgn, load_imdb_ycm, load_imdb_magnn, load_imdb_hgt
from models.Simple_HGN import SimpleHGN
from models.HGCF import HGCF
from models.HAN import HAN
from models.HGT import HGT
from trainer import Trainer
import copy
from utils import my_loss
from sklearn.metrics import roc_auc_score
from models.MAGNN import MAGNN_nc, MAGNN_nc_mb
from models.GTN import GTN
from models.Simple_HGN import SimpleHGN
from utils import index_generator, parse_minibatch
from sklearn.metrics import f1_score
import dgl
import scipy.sparse as sp

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser('Main')
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--dataset', type=str, default='ACM')
parser.add_argument('--model', type=str, default='HAN')
parser.add_argument('--label_ratio', type=int, default=20, 
                    help='train labels per class [20, 50]')

parser.add_argument('--lr', type=float, default=0.001, #simple[0.001,0]
                    help='learning rate')#0.01

parser.add_argument('--wd', type=float, default=0.0000, 
                    help='weight decay')#0.0005

parser.add_argument('--optimizer', type=str, default='adam', 
                    help='Optimizer.')
parser.add_argument('--num_layers', type=int, default=1, 
                    help='number of teacher model layers')
parser.add_argument('--lr_teacher', type=float, default=0.005, 
                    help='teacher learning rate')#0.01
parser.add_argument('--tea_hid',type=int, default=32)
parser.add_argument('--wd_teacher', type=float, default=0.0005, 
                    help='teacher weight decay')#0.001
parser.add_argument('--tea_drop', type=float, default=0.4)
parser.add_argument('--num_heads', type=list, default=[2])
parser.add_argument('--dropout', type=float, default=0.5)#teacher dropout0.6
parser.add_argument('--epoch', type=int, default=150)

args = parser.parse_args()


opt = vars(args)
setup_seed(opt['seed'])
epoches = opt['epoch']
device = opt['device']
dataset = opt['dataset']
ratio = opt['label_ratio']
lr = opt['lr']
wd = opt['wd']
num_layers = 1

if dataset == 'ACM':
    type_num = [4019, 7167, 60]
    sample_rate = [3, 1]
    nei_num = 2
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_acm(ratio, type_num, device)

if dataset =='DBLP':
    type_num = [4057, 14328, 7723, 20]
    sample_rate = [3]
    nei_num = 1
    opt['etypes_lists'] = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
    opt['metapath_list'] = 3
    opt['edge_type'] = 6
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_dblp(ratio, type_num, device)

if dataset =='IMDB':
    type_num = [4278, 2081, 5257]
    sample_rate = [1, 2]
    nei_num = 2
    etypes_lists = [[[0, 1], [2, 3]]]
    metapath_list = [2]
    edge_type = 4
    feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_imdb(ratio, type_num, device)

opt['nei_num'] = nei_num
opt['sample_rate'] = sample_rate

feats = [feat.to(device) for feat in feats]
#G = [g.to(device) for g in G]
labels = labels.to(device)


if  1:
    if opt['model'] == 'HAN':
        if dataset == 'ACM':
            g, graph, nei_idx, labels_one_hot, labels_init = load_acm_ycm(ratio, device)
        if dataset == 'DBLP':
            g, graph, nei_idx, labels_one_hot, labels_init = load_dblp_ycm(ratio, device)
        if dataset == 'IMDB':
            g, nei_idx, labels_one_hot, labels_init = load_imdb_ycm(ratio, device)

        model = HAN(num_paths=len(g), in_dim=feats[0].shape[1], hid_dim=64, num_heads=[8], dropout=0.6, num_classes=num_classes,  g=g)

    if opt['model'] == 'SimpleHGN':
        if dataset == 'ACM':
            gs = load_acm_simplehgn(device)
        if dataset == 'DBLP':
            gs = load_dblp_simplehgn(device)
        if dataset == 'IMDB':
            gs= load_imdb_simplehgn(device)
        model = SimpleHGN(graph=gs, edge_dim=64, num_etypes=5, in_dim=128, feats_dim_list=feat_dims, num_hidden=128, num_classes=num_classes,\
        num_layers=1, heads=[8, 1], feat_drop=0.6, attn_drop=0.6, negative_slope=0.05, residual=True, alpha=0.05)

    if opt['model'] == 'MAGNN':
        if dataset == 'ACM':
            adjlists, edge_metapath_indices_list, type_mask = load_acm_magnn(device)
        if dataset == 'DBLP':
            adjlists, edge_metapath_indices_list, type_mask = load_dblp_magnn(device)
        if dataset == 'IMDB':
            g_lists, edge_metapath_indices_lists, type_mask, target_node_indices = load_imdb_magnn(device)
            model = MAGNN_nc(num_layers=1, num_metapaths_list=[2], num_edge_type=4, etypes_lists=etypes_lists, \
                feats_dim_list=feat_dims, hidden_dim=64, out_dim=num_classes, num_heads=8, attn_vec_dim=64, rnn_type='RotatE0', dropout_rate=0.5)
    if opt['model'] == 'HGT':
        if dataset == 'ACM':
            G_hgt, node_dict, edge_dict = load_acm_hgt(device)
            model = HGT(G_hgt,
            node_dict, edge_dict,
            n_inp=[feat_dims[1], feat_dims[0], feat_dims[2]],
            n_hid=256,
            n_out=num_classes,
            n_layers=2,
            n_heads=8,
            feat_drop=0.6,
            dropout=0.8,
            use_norm = True).to(device)
        if dataset == 'DBLP':
            G_hgt, node_dict, edge_dict = load_dblp_hgt(device)
            model = HGT(G_hgt,
            node_dict, edge_dict,
            n_inp=[feat_dims[0], feat_dims[3], feat_dims[1], feat_dims[2]],
            n_hid=256,
            n_out=num_classes,
            n_layers=2,
            n_heads=4,
            feat_drop=0.3, 
            dropout = 0.6, 
            use_norm = True).to(device)
        if dataset == 'IMDB':
            G_hgt, node_dict, edge_dict = load_imdb_hgt(device)
            model = HGT(G_hgt,
            node_dict, edge_dict,
            n_inp=[feat_dims[2], feat_dims[1], feat_dims[0]],
            n_hid=256,
            n_out=num_classes,
            n_layers=2,
            n_heads=4,
            feat_drop=0.6, 
            dropout = 0.5, 
            use_norm = True).to(device)
    

loss_fcn = torch.nn.CrossEntropyLoss()
#num_nodes = edges[0].shape[0]
num_nodes = 4057 + 14328 + 7723 + 20
print (num_nodes)



model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def evaluate(idx_val, idx_test, target, feats):
    model.eval()
    with torch.no_grad():
        if opt['model'] == 'MAGNN':
            logits, _, _ = model((g_lists, feats, type_mask, edge_metapath_indices_lists), target_node_indices)
        else:
            logits, _, _, _, _ = model(feats)
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
feats = [feat.to(device) for feat in feats]
for epoch in range(epoches):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    #logits = model(features)
    if opt['model'] == 'MAGNN':
        logits, _, _ = model((g_lists, feats, type_mask, edge_metapath_indices_lists), target_node_indices)
    else:
        logits, _, _, _, _ = model(feats)
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
print(preds)
np.save('outputs/preds'+ str(opt['model']) + str(dataset) + str(ratio) + '.npy', preds)
