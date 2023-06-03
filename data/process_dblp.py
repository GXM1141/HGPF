import torch
import time
import dgl
import numpy as np
import scipy.sparse as sp
import scipy.sparse
import pickle
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
from dgl.data.utils import save_graphs, load_graphs
from data.utils import encode_onehot, preprocess_adj, initialize_label, preprocess_features_1

def load_dblp(ratio, type_num, device):
    path = "data/dblp/"
    
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label).to(device)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    feat_t = sp.eye(type_num[2])
    feat_c = sp.eye(20)

    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")
    
    feat_a = sp.csr_matrix(feat_a)
    feat_a = torch.from_numpy(feat_a.todense()).float().to(device)
    feat_p = torch.FloatTensor(preprocess_features_1(feat_p)).to(device)
    feat_t = torch.FloatTensor(preprocess_features_1(feat_t)).to(device)
    feat_c = torch.FloatTensor(preprocess_features_1(feat_c)).to(device)

    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    node_num = label.shape[0]
    num_classes = labels_one_hot.shape[1]
    feat_dims = [feat_a.shape[1], feat_p.shape[1], feat_t.shape[1], feat_c.shape[1]]
    return  [feat_a, feat_p, feat_t, feat_c], label, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims

def load_dblp_ycm(ratio, device):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "data/dblp/"

    pa = np.loadtxt(path + "pa.txt")
    pc = np.loadtxt(path + "pc.txt")
    pt = np.loadtxt(path + "pt.txt")
    type_num = [4057, 14328, 7723, 20]
    num_nodes = 4057 + 14328 + 7723 + 20
    #            a(tar)  p      t      c

    num_edges = [pa.shape[0], pt.shape[0], pc.shape[0], pa.shape[0], pt.shape[0], pc.shape[0]]
    g_list, _ = load_graphs('data/g1.bin')
    g = g_list[0]
    graph = g.to(device)
    
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    
    nei_p = [torch.LongTensor(i).to(device) for i in nei_p]
    apa = preprocess_adj(apa)
    apcpa = preprocess_adj(apcpa)
    aptpa = preprocess_adj(aptpa)
    
    apa_g = dgl.from_scipy(apa)
    apcpa_g = dgl.from_scipy(apcpa)
    aptpa_g = dgl.from_scipy(aptpa)

    g = [apa_g, apcpa_g, aptpa_g]
    g[0] = g[0].to(device)
    g[1] = g[1].to(device)
    g[2] = g[2].to(device)
    train = np.load(path + "train_" + str(ratio) + ".npy")
    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    label_init = initialize_label(idx_train, labels_one_hot).to(device)

    return  g, graph, [nei_p], labels_one_hot, label_init

def load_dblp_simplehgn(device):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "data/dblp/"

    pa = np.loadtxt(path + "pa.txt")
    pc = np.loadtxt(path + "pc.txt")
    pt = np.loadtxt(path + "pt.txt")
    type_num = [4057, 14328, 7723, 20]
    num_nodes = 4057 + 14328 + 7723 + 20
    #            a(tar)  p      t      c

    num_edges = [pa.shape[0], pt.shape[0], pc.shape[0], pa.shape[0], pt.shape[0], pc.shape[0]]

    pa_u = []
    pa_v = []

    for edge in pa:
        pa_u.append(int(edge[0] + 4057))
        pa_v.append(int(edge[1]))

    pa_u = torch.from_numpy(np.array(pa_u))
    pa_v = torch.from_numpy(np.array(pa_v))

    pt_u = []
    pt_v = []
    for edge in pt:
        pt_u.append(int(edge[0] + 4057))
        pt_v.append(int(edge[1] + 4057 + 14328))

    pt_u = torch.from_numpy(np.array(pt_u))
    pt_v = torch.from_numpy(np.array(pt_v))
    pc_u = []
    pc_v = []

    for edge in pc:
        pc_u.append(int(edge[0] + 4057))
        pc_v.append(int(edge[1] + 4057 + 14328 + 7723))

    pc_u = torch.from_numpy(np.array(pc_u))
    pc_v = torch.from_numpy(np.array(pc_v))
    self_u = torch.arange(num_nodes).int()

    u = torch.cat([pa_u, pt_u, pc_u, pa_v, pt_v, pc_v], dim = -1)
    v = torch.cat([pa_v, pt_v, pc_v, pa_u, pt_u, pc_u], dim = -1)

    types = []

    for i in range(len(num_edges)):
        t = torch.zeros(num_edges[i])
        t = t + i
        types.append(t.int())
 
    types = torch.cat(types, dim = -1)

    g = dgl.graph((u, v))

    g.edata['t'] = types
    g_s = g.to(device)
    return g_s


def load_dblp_magnn(device):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "data/dblp/"
    
    in_file = open(path + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]#节点v基于该类元路径的所有邻居组成的list(shape:4057 * num_neighbors(不确定))
    in_file.close()
    in_file = open(path + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(path + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(path + '/0/0-1-0_idx.pickle', 'rb')
    idx0 = pickle.load(in_file)#dic[key:节点v val:array([u, k, v元路径实例]...)]
    in_file.close()
    in_file = open(path + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx1 = pickle.load(in_file)
    in_file.close()
    in_file = open(path + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx2 = pickle.load(in_file)
    in_file.close()
    
    type_mask = np.load(path + '/node_types.npy')

    return  [adjlist00, adjlist01, adjlist02], [idx0, idx1, idx2], type_mask


def load_dblp_hgt(device):#
    path = "data/dblp/"
    pa = np.loadtxt(path + "pa.txt")
    pc = np.loadtxt(path + "pc.txt")
    pt = np.loadtxt(path + "pt.txt")
    type_num = [4057, 14328, 7723, 20]
    num_nodes = 4057 + 14328 + 7723 + 20
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    feat_t = sp.eye(type_num[2])
    feat_c = sp.eye(20)

    feat_a = sp.csr_matrix(feat_a)
    feat_a = torch.from_numpy(feat_a.todense()).float()
    feat_p = torch.FloatTensor(preprocess_features_1(feat_p))
    feat_t = torch.FloatTensor(preprocess_features_1(feat_t))
    feat_c = torch.FloatTensor(preprocess_features_1(feat_c))

    pa_u = []
    pa_v = []

    for edge in pa:
        pa_u.append(int(edge[0]))
        pa_v.append(int(edge[1]))

    pa_u = torch.from_numpy(np.array(pa_u))
    pa_v = torch.from_numpy(np.array(pa_v))


    pt_u = []
    pt_v = []

    for edge in pt:
        pt_u.append(int(edge[0]))
        pt_v.append(int(edge[1]))

    pt_u = torch.from_numpy(np.array(pt_u))
    pt_v = torch.from_numpy(np.array(pt_v))

    pc_u = []
    pc_v = []

    for edge in pc:
        pc_u.append(int(edge[0]))
        pc_v.append(int(edge[1]))

    pc_u = torch.from_numpy(np.array(pc_u))
    pc_v = torch.from_numpy(np.array(pc_v))

    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : (pa_u, pa_v),
            ('author', 'writing', 'paper') : (pa_v, pa_u),
            ('paper', 'b', 'c') : (pc_u, pc_v),
            ('c', 'd', 'paper') : (pc_v, pc_u),
            ('paper', 'e', 't') : (pt_u, pt_v),
            ('t', 'f', 'paper') : (pt_v, pt_u),
        })
    for ntype in G.ntypes:
        if ntype == 'paper':
            emb = feat_p
        if ntype == 'author':
            emb = feat_a
        if ntype == 'c':
            emb = feat_c
        if ntype == 't':
            emb = feat_t
        G.nodes[ntype].data['inp'] = emb
    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    G = G.to(device)
    print(G )
    return G, node_dict, edge_dict

