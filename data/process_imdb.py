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

def load_imdb(ratio, type_num, device):
    path = "data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float()
    label = label.nonzero()[:, 1]
    feat_m = sp.load_npz(path + "m_feat.npz")
    feat_d= sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")
    feat_m = torch.FloatTensor(feat_m.todense()).to(device)

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a)).to(device)
    feat_d = torch.FloatTensor(preprocess_features_1(feat_d)).to(device)

    idx_train = torch.LongTensor(train).squeeze(0)
    idx_val = torch.LongTensor(val).squeeze(0)
    idx_test = torch.LongTensor(test).squeeze(0)
    node_num = label.shape[0]
    num_classes = labels_one_hot.shape[1]
    feat_dims = [feat_m.shape[1], feat_d.shape[1], feat_a.shape[1]]

    return [feat_m, feat_d, feat_a], label, idx_train, idx_val, idx_test, node_num, num_classes, feat_dims

def load_imdb_ycm(ratio, device):
    path = "data/imdb/"
    """
    adj = sp.load_npz(path + 'adjM.npz')

    type_num = [4278, 2081, 5257]

    ma_u, ma_v, md_u, md_v = [], [], [], []

    ma_num = 0
    md_num = 0

    for i in range(4278):
        for j in range(4278, 4278 + 2081):
            if (adj[i, j] != 0):
                md_u.append(i)
                md_v.append(j)
                md_num = md_num + 1

        #ma, am
    for i in range(4278):
        for j in range(4278 + 2081, 4278 + 2081 + 5257):
            if (adj[i, j] != 0):
                ma_u.append(i)
                ma_v.append(j)
                ma_num = ma_num + 1

    edges_num = [ma_num, md_num, ma_num, md_num]

    ma_u = torch.from_numpy(np.array(ma_u))
    ma_v = torch.from_numpy(np.array(ma_v))
    md_u = torch.from_numpy(np.array(md_u))
    md_v = torch.from_numpy(np.array(md_v))

    u = [ma_u, md_u, ma_v, md_v]
    v = [ma_v, md_v, ma_u, md_u]

    u = torch.cat(u, dim = -1)
    v = torch.cat(v, dim = -1)

    g = dgl.graph((u, v), num_nodes = (4278 + 2081 + 5257))
    graph = g.to(device)
    """
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    G00 = nx.read_adjlist(path + '/IMDB_processed/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(path + '/IMDB_processed/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = [torch.LongTensor(i).to(device) for i in nei_a]
    nei_d = [torch.LongTensor(i).to(device) for i in nei_d]
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")

    mam = preprocess_adj(mam)
    mdm = preprocess_adj(mdm)
    #mam_g = dgl.from_scipy(mam)
    #mdm_g = dgl.from_scipy(mdm)
    nx_G_list = [G00, G01]
    g = []
    for nx_G in nx_G_list:
        gr = dgl.DGLGraph(multigraph=True)
        #gr = dgl.grpah()
        gr.add_nodes(nx_G.number_of_nodes())
        gr.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
        g.append(gr)
    g[0] = g[0].to(device)
    g[1] = g[1].to(device)
    train = np.load(path + "train_" + str(ratio) + ".npy")
    idx_train = torch.LongTensor(train).squeeze(0)
    label_init = initialize_label(idx_train, labels_one_hot)
    graph = 1
    return g, graph, [nei_d, nei_a], labels_one_hot, label_init

def load_imdb_simplehgn(device):
    path = "data/imdb/"

    adj = sp.load_npz(path + 'adjM.npz')

    type_num = [4278, 2081, 5257]

    ma_u, ma_v, md_u, md_v = [], [], [], []

    ma_num = 0
    md_num = 0

    for i in range(4278):
        for j in range(4278, 4278 + 2081):
            if (adj[i, j] != 0):
                md_u.append(i)
                md_v.append(j)
                md_num = md_num + 1

        #ma, am
    for i in range(4278):
        for j in range(4278 + 2081, 4278 + 2081 + 5257):
            if (adj[i, j] != 0):
                ma_u.append(i)
                ma_v.append(j)
                ma_num = ma_num + 1

    edges_num = [ma_num, md_num, ma_num, md_num]

    ma_u = torch.from_numpy(np.array(ma_u))
    ma_v = torch.from_numpy(np.array(ma_v))
    md_u = torch.from_numpy(np.array(md_u))
    md_v = torch.from_numpy(np.array(md_v))

    u = [ma_u, md_u, ma_v, md_v]
    v = [ma_v, md_v, ma_u, md_u]

    u = torch.cat(u, dim = -1)
    v = torch.cat(v, dim = -1)
    types = []

    for i in range(len(edges_num)):
        t = torch.zeros(edges_num[i], dtype = torch.int)
        t = t + i
        types.append(t.int())

    types = torch.cat(types, dim = -1)
    g = dgl.graph((u, v), num_nodes = (4278 + 2081 + 5257))
    g.edata['t'] = types
    g_s = g.to(device)

    return g_s

def load_imdb_magnn(device):
    path = "data/imdb/"
    G00 = nx.read_adjlist(path + '/IMDB_processed/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(path + '/IMDB_processed/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(path + '/IMDB_processed/0/0-1-0_idx.npy')
    idx01 = np.load(path + '/IMDB_processed/0/0-2-0_idx.npy')
   
    type_mask = np.load(path + '/node_types.npy')
    edge_metapath_indices_lists = [[idx00, idx01]]
    nx_G_lists = [[G00, G01]]
    target_node_indices = np.where(type_mask == 0)[0]
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g = g.to(device)
            g_lists[-1].append(g)
    return g_lists, edge_metapath_indices_lists, type_mask, target_node_indices


def load_imdb_hgt(device):#
    path = "data/imdb/"
    ma = np.loadtxt(path + "ma.txt")
    md = np.loadtxt(path + "md.txt")
    type_num = [4278, 2081, 5257]
    feat_m = sp.load_npz(path + "m_feat.npz")
    feat_d= sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])

    feat_m = torch.FloatTensor(feat_m.todense())

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a))
    feat_d = torch.FloatTensor(preprocess_features_1(feat_d))
    ma_u = []
    ma_v = []

    for edge in ma:
        ma_u.append(int(edge[0]))
        ma_v.append(int(edge[1]-4278-2081))

    ma_u = torch.from_numpy(np.array(ma_u))
    ma_v = torch.from_numpy(np.array(ma_v))


    md_u = []
    md_v = []

    for edge in md:
        md_u.append(int(edge[0]))
        md_v.append(int(edge[1]-4278))

    md_u = torch.from_numpy(np.array(md_u))
    md_v = torch.from_numpy(np.array(md_v))

    G = dgl.heterograph({
            ('movie', 'a', 'actor') : (ma_u, ma_v),
            ('actor', 'b', 'movie') : (ma_v, ma_u),
    #        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
    #        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('movie', 'c', 'director') : (md_u, md_v),
            ('director', 'd', 'movie') : (md_v, md_u),
        })
    for ntype in G.ntypes:
        print (ntype)
        print (G.number_of_nodes(ntype))
        if ntype == 'movie':
            emb = feat_m
        if ntype == 'director':
            emb = feat_d
        if ntype == 'actor':
            emb = feat_a
        G.nodes[ntype].data['inp'] = emb
    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    G = G.to(device)
    print (G)
    return G, node_dict, edge_dict





