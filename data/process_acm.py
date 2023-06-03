import torch
import time
import dgl
import numpy as np
import scipy.sparse as sp
import scipy.sparse
import pickle
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
from data.utils import encode_onehot, preprocess_adj, initialize_label, preprocess_features_1

def load_acm(ratio, type_num, device):#
    path = "data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")

    feat_p = sp.csr_matrix(feat_p)
    feat_p = torch.from_numpy(feat_p.todense()).float()

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a))
    feat_s = torch.FloatTensor(preprocess_features_1(feat_s))

    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    node_num = label.shape[0]
    num_classes = labels_one_hot.shape[1]
    feat_dims = [feat_p.shape[1], feat_a.shape[1], feat_s.shape[1]]
    return [feat_p, feat_a, feat_s], label, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims

def load_acm_ycm(ratio, device):#
    path = "data/acm/"
    pa = np.loadtxt(path + "pa.txt")
    ps = np.loadtxt(path + "ps.txt")
    type_num = [4019, 7167, 60]
    num_nodes = 4019 + 7167 + 60
    sc_inst = np.load(path + 'sc_instance.npy', allow_pickle=True)
    num_edges = [pa.shape[0], ps.shape[0], pa.shape[0], ps.shape[0]]


    pa_u = []
    pa_v = []

    for edge in pa:
        pa_u.append(int(edge[0]))
        pa_v.append(int(edge[1]+4019))

    pa_u = torch.from_numpy(np.array(pa_u))
    pa_v = torch.from_numpy(np.array(pa_v))


    ps_u = []
    ps_v = []

    for edge in ps:
        ps_u.append(int(edge[0]))
        ps_v.append(int(edge[1]+4019 + 7167))

    ps_u = torch.from_numpy(np.array(ps_u))
    ps_v = torch.from_numpy(np.array(ps_v))

    self_u = torch.arange(num_nodes).int()

    u = torch.cat([pa_u, ps_u, pa_v, ps_v, self_u], dim = -1)
    v = torch.cat([pa_v, ps_v, pa_u, ps_u, self_u], dim = -1)

    g = dgl.graph((u, v))
    graph = g.to(device)
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")

    nei_a = [torch.LongTensor(i).to(device) for i in nei_a]
    nei_s = [torch.LongTensor(i).to(device) for i in nei_s]
    sc_in = []
    for sc in sc_inst:
        sc = [torch.LongTensor(i).unsqueeze(dim=0).to(device) for i in sc]
        sc_in.append(sc)
    
    pap = preprocess_adj(pap)
    psp = preprocess_adj(psp)
    pap_g = dgl.from_scipy(pap)
    psp_g = dgl.from_scipy(psp)
    g = [pap_g, psp_g]
    g[0] = g[0].to(device)
    g[1] = g[1].to(device)
    train = np.load(path + "train_" + str(ratio) + ".npy")
    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    label_init = initialize_label(idx_train, labels_one_hot).to(device)

    return g, graph, [nei_a, nei_s], labels_one_hot, label_init
    #return [adj00, adj01], [idx0, idx1], g, [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, idx_train, idx_val, idx_test, labels_one_hot, label_init, type_mask



def load_acm_magnn(device):#
    path = "data/acm/"

    idx0 = np.load(path + '/idx0.npy', allow_pickle=True)
    idx1 = np.load(path + '/idx1.npy', allow_pickle=True)

    type_mask = np.load(path + '/node_types.npy')


    idx00 = {}
    adj0 = []

    for i in range(4019):
        inst = []
        a = b = 0
        adj = [i]
        for idx in idx0[i]:
            if idx[2] == a and idx[1] == b:
                continue
            else: 
                idx = list(idx)
                idx.reverse()
                idx[1] = idx[1] + 4019
                inst.append(np.array(idx))
                a = idx[0]
                b = idx[1]
                adj.append(int(a))
        inst = np.array(inst)
        idx00[i] = inst
        adj0.append(adj)


    idx01 = {}
    adj1 = []
    for i in range(4019):
        inst = []
        a = b = 0
        adj = [i]
        for idx in idx1[i]:
            if idx[2] == a and idx[1] == b:
                continue
            else: 
                idx = list(idx)
                idx.reverse()
                idx[1] = idx[1] + 4019 + 7167
                inst.append(np.array(idx))
                a = idx[0]
                b = idx[1]
                adj.append(int(a))
        inst = np.array(inst)
        idx01[i] = inst
        adj1.append(adj)
    return [adj0, adj1], [idx00, idx01], type_mask
    

def load_acm_simplehgn(device):#
    path = "data/acm/"
    pa = np.loadtxt(path + "pa.txt")
    ps = np.loadtxt(path + "ps.txt")
    type_num = [4019, 7167, 60]
    num_nodes = 4019 + 7167 + 60

    num_edges = [pa.shape[0], ps.shape[0], pa.shape[0], ps.shape[0]]
    

    pa_u = []
    pa_v = []

    for edge in pa:
        pa_u.append(int(edge[0]))
        pa_v.append(int(edge[1]+4019))

    pa_u = torch.from_numpy(np.array(pa_u))
    pa_v = torch.from_numpy(np.array(pa_v))


    ps_u = []
    ps_v = []

    for edge in ps:
        ps_u.append(int(edge[0]))
        ps_v.append(int(edge[1]+4019 + 7167))

    ps_u = torch.from_numpy(np.array(ps_u))
    ps_v = torch.from_numpy(np.array(ps_v))

    self_u = torch.arange(num_nodes).int()

    u = torch.cat([pa_u, ps_u, pa_v, ps_v, self_u], dim = -1)
    v = torch.cat([pa_v, ps_v, pa_u, ps_u, self_u], dim = -1)

    types = []

    for i in range(len(num_edges)):
        t = torch.zeros(num_edges[i])
        t = t + i
        types.append(t.int())
    t = torch.zeros(num_nodes)
    t = t + 4
    types.append(t.int())

    types = torch.cat(types, dim = -1)
    g = dgl.graph((u, v))
    g.edata['t'] = types
    gs = g.to(device)
    return gs

def load_acm_hgt(device):#
    path = "data/acm/"
    pa = np.loadtxt(path + "pa.txt")
    ps = np.loadtxt(path + "ps.txt")
    type_num = [4019, 7167, 60]
    num_nodes = 4019 + 7167 + 60
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    feat_p = sp.csr_matrix(feat_p)
    feat_p = torch.from_numpy(feat_p.todense()).float()

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a))
    feat_s = torch.FloatTensor(preprocess_features_1(feat_s))
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
    for ntype in G.ntypes:
        print (ntype)
        print (G.number_of_nodes(ntype))
        if ntype == 'paper':
            emb = feat_p
        if ntype == 'author':
            emb = feat_a
        if ntype == 'subject':
            emb = feat_s
        G.nodes[ntype].data['inp'] = emb
    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    G = G.to(device)
    return G, node_dict, edge_dict

