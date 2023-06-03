from data.process_dblp import load_dblp_magnn
import numpy as np
import scipy
import networkx as nx
import dgl
import scipy.sparse as sp
import pickle
import torch
if 1:
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
    at_u = []
    at_v = []
    for edge in pt:
        for edge1 in pa:
            if edge[0] == edge1[0]:
                at_u.append(int(edge1[1]))
                at_v.append(int(edge[1]))
    at_u = torch.from_numpy(np.array(at_u))
    at_v = torch.from_numpy(np.array(at_v))
    ac_u = []
    ac_v = []
    for edge in pc:
        for edge1 in pa:
            if edge[0] == edge1[0]:
                ac_u.append(int(edge1[1]))
                ac_v.append(int(edge[1]))
    ac_u = torch.from_numpy(np.array(ac_u))
    ac_v = torch.from_numpy(np.array(ac_v))


    u = torch.cat([pa_u, at_u, ac_u, pa_v, at_v, ac_v], dim = -1)
    v = torch.cat([pa_v, at_v, ac_v, pa_u, at_u, ac_u], dim = -1)


    g = dgl.graph((u, v))

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