import scipy.io
import urllib.request
import dgl
import math
import torch
import numpy as np
from model import *
from models.HGT import HGT, HeteroRGCN
import argparse
from data.process_acm import load_acm, load_acm_simplehgn, load_acm_ycm, load_acm_magnn
torch.manual_seed(0)
path = "data/acm/"

pa = np.loadtxt(path + "pa.txt")
ps = np.loadtxt(path + "ps.txt")



parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')
feats, labels, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims = load_acm(20, [4019, 7167, 60], 'cuda:1')

parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0) 
parser.add_argument('--max_lr',  type=float, default=1e-3) 

args = parser.parse_args()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
label = labels.cpu()
def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'paper')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, 'paper')
            pred   = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == label[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == label[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == label[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'], 
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))

device = torch.device("cuda:1")

type_num = [4019, 7167, 60]
num_nodes = 4019 + 7167 + 60

pa_u = []
pa_v = []

for edge in pa:
    pa_u.append(int(edge[0]))
    pa_v.append(int(edge[1]))

pa_u = torch.from_numpy(np.array(pa_u))
pa_v = torch.from_numpy(np.array(pa_v))


ps_u = []
ps_v = []

for edge in ps:
    ps_u.append(int(edge[0]))
    ps_v.append(int(edge[1]))

ps_u = torch.from_numpy(np.array(ps_u))
ps_v = torch.from_numpy(np.array(ps_v))

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : (pa_u, pa_v),
        ('author', 'writing', 'paper') : (pa_v, pa_u),
#        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
#        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : (ps_u, ps_v),
        ('subject', 'has', 'paper') : (ps_v, ps_u),
    })
print(G)


#pvc = data['PvsC'].tocsr()
#p_selected = pvc.tocoo()
# generate labels
#labels = pvc.indices
#labels = torch.tensor(labels).long()

# generate train/val/test split
#pid = p_selected.row
#shuffle = np.random.permutation(pid)
#train_idx = torch.tensor(shuffle[0:800]).long()
#val_idx = torch.tensor(shuffle[800:900]).long()
#test_idx = torch.tensor(shuffle[900:]).long()
"""
train_idx = idx_train.to(device)
val_idx = idx_val.to(device)
test_idx = idx_test.to(device)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
feats = [feats[1], feats[0], feats[2]]
feat_dims = [feat_dims[1], feat_dims[0], feat_dims[2]]
#     Random initialize input feature
for ntype in G.ntypes:
    print (ntype)
    print (G.number_of_nodes(ntype))

    if ntype == 'paper':
        emb = feats[1]
    if ntype == 'author':
        emb = feats[0]
    if ntype == 'subject':
        emb = feats[2]
    G.nodes[ntype].data['inp'] = emb

    
G = G.to(device)

model = HGT(G,
            node_dict, edge_dict,
            n_inp=feat_dims,
            n_hid=args.n_hid,
            n_out=labels.max().item()+1,
            n_layers=2,
            n_heads=4,
            use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training HGT with #param: %d' % (get_n_params(model)))
train(model, G)




model = HeteroRGCN(G,
                   in_size=args.n_inp,
                   hidden_size=args.n_hid,
                   out_size=labels.max().item()+1).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training RGCN with #param: %d' % (get_n_params(model)))
train(model, G)



model = HGT(G,
            node_dict, edge_dict,
            n_inp=args.n_inp,
            n_hid=args.n_hid,
            n_out=labels.max().item()+1,
            n_layers=0,
            n_heads=4).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training MLP with #param: %d' % (get_n_params(model)))
train(model, G)
"""