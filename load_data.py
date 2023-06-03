from operator import index
from data.utils import normalize_adj_1
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from data.utils import encode_onehot, preprocess_adj, initialize_label, preprocess_features_1
import dgl
def normalize_adjacency_matrix(A, I):
   
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat
def load_acm(device,ratio):
    path = "data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    pa = np.loadtxt(path + "pa.txt").astype('int32')
    ps = np.loadtxt(path + "ps.txt").astype('int32')
    type_num = [4019, 7167, 60]
    num_nodes = 4019 + 7167 + 60
    index_1 = [edge[0] for edge in ps] + [(edge[1]+4019+7167) for edge in ps]+[edge[0] for edge in pa] + [(edge[1]+4019) for edge in pa]
    index_2 = [(edge[1]+4019+7167) for edge in ps] + [edge[0] for edge in ps]+[(edge[1]+4019) for edge in pa] + [edge[0] for edge in pa]
    num_edges = [pa.shape[0], ps.shape[0], pa.shape[0], ps.shape[0]]
    types = []
    for i in range(len(num_edges)):
        t = torch.zeros(num_edges[i])
        t = t + i
        types.append(t.int())
    t = torch.zeros(num_nodes)
    t = t + 4
    #types.append(t.int())
    types = torch.cat(types, dim = -1)

    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((np.array(values)/2, (np.array(index_1), np.array(index_2))), shape=(node_count, node_count), dtype=np.float32)
    I = sparse.eye(A.shape[0])
    adj = normalize_adjacency_matrix(A, I)
    adj = A + I
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.FloatTensor(adj.data.astype(np.float))).to(device)
    feat_p = sparse.load_npz(path + "p_feat.npz")
    feat_a = sparse.eye(type_num[1])
    feat_s = sparse.eye(type_num[2])
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")

    feat_p = sparse.csr_matrix(feat_p)
    feat_p = torch.from_numpy(feat_p.todense()).float().to(device)

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a)).to(device)
    feat_s = torch.FloatTensor(preprocess_features_1(feat_s)).to(device)

    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    node_num = label.shape[0]
    num_classes = labels_one_hot.shape[1]
    feat_dims = [feat_p.shape[1], feat_a.shape[1], feat_s.shape[1]]
    #self_u = np.array(range(node_count)).tolist()
    #index_1 = index_1 + self_u
    #index_2 = index_2 + self_u
    u = torch.LongTensor(np.array(index_1)).to(device)
    v = torch.LongTensor(np.array(index_2)).to(device)
    edge_index = [index_1, index_2]
    g = dgl.graph((u, v))
    g.edata['t'] = types

    return adj, [feat_p, feat_a, feat_s], feat_dims, idx_train, idx_val, idx_test, num_classes, label, g

def load_dblp(device,ratio):
    path = "data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    pa = np.loadtxt(path + "pa.txt").astype('int32')
    pc = np.loadtxt(path + "pc.txt").astype('int32')
    pt = np.loadtxt(path + "pt.txt").astype('int32')
    type_num = [4057, 14328, 7723, 20]
    num_nodes = 4057 + 14328 + 7723 + 20
    num_edges = [pa.shape[0], pc.shape[0], pt.shape[0], pa.shape[0], pc.shape[0], pt.shape[0]]
    types = []
    for i in range(len(num_edges)):
        t = torch.zeros(num_edges[i])
        t = t + i
        types.append(t.int())
    t = torch.zeros(num_nodes)
    t = t + 6
    types.append(t.int())
    types = torch.cat(types, dim = -1)

    index_1 = [edge[1] for edge in pa] + [(edge[0]+4057) for edge in pa]+[edge[0]+4057 for edge in pt] + [(edge[1]+4057+14328) for edge in pt]+[edge[0]+4057 for edge in pc] + [(edge[1]+4057+14328+7723) for edge in pc]
    index_2 = [(edge[0]+4057) for edge in pa] + [edge[1] for edge in pa]+[(edge[1]+4057+14328) for edge in pt] + [edge[0]+4057 for edge in pt]+[(edge[1]+4057+14328+7723) for edge in pc] + [edge[0]+4057 for edge in pc]
    values = [1 for edge in index_1]
    node_count =  max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((np.array(values)/2, (np.array(index_1), np.array(index_2))), shape=(node_count, node_count), dtype=np.float32)
    I = sparse.eye(A.shape[0])
    adj = normalize_adjacency_matrix(A, I)
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.FloatTensor(adj.data.astype(np.float))).to(device)
    
    print (adj.shape)
    feat_a = sparse.load_npz(path + "a_feat.npz")
    feat_p = sparse.eye(type_num[1])
    feat_t = sparse.eye(type_num[2])
    feat_c = sparse.eye(20)
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")

    feat_a = sparse.csr_matrix(feat_a)
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
    u = torch.LongTensor(np.array(index_1)).to(device)
    v = torch.LongTensor(np.array(index_2)).to(device)
    edge_index = [index_1, index_2]
    g = dgl.graph((u, v))
    #g.edata['t'] = types

    return adj, [feat_a, feat_p, feat_t, feat_c], feat_dims, idx_train, idx_val, idx_test, num_classes, label, g

def load_imdb(device,ratio):
    path = "data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    type_num = [4278, 2081, 5257]
    pa = np.loadtxt(path + "md.txt").astype('int32')
    ps = np.loadtxt(path + "ma.txt").astype('int32')
    num_nodes = 4278+2081+5257
    index_1 = [edge[0] for edge in ps] + [(edge[1]) for edge in ps]+[edge[0] for edge in pa] + [(edge[1]) for edge in pa]
    index_2 = [(edge[1]) for edge in ps] + [edge[0] for edge in ps]+[(edge[1]) for edge in pa] + [edge[0] for edge in pa]

    num_edges = [pa.shape[0], ps.shape[0], pa.shape[0], ps.shape[0]]
    types = []
    for i in range(len(num_edges)):
        t = torch.zeros(num_edges[i])
        t = t + i
        types.append(t.int())
    t = torch.zeros(num_nodes)
    t = t + 4
    #types.append(t.int())
    types = torch.cat(types, dim = -1)
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((np.array(values)/2, (np.array(index_1), np.array(index_2))), shape=(node_count, node_count), dtype=np.float32)
    I = sparse.eye(A.shape[0])
    adj = normalize_adjacency_matrix(A, I)
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.FloatTensor(adj.data.astype(np.float))).to(device)
    print(adj.shape)
    feat_p = sparse.load_npz(path + "m_feat.npz")
    feat_a = sparse.eye(type_num[1])
    feat_s = sparse.eye(type_num[2])
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")

    feat_p = sparse.csr_matrix(feat_p)
    feat_p = torch.from_numpy(feat_p.todense()).float().to(device)

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a)).to(device)
    feat_s = torch.FloatTensor(preprocess_features_1(feat_s)).to(device)

    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    node_num = label.shape[0]
    num_classes = labels_one_hot.shape[1]
    feat_dims = [feat_p.shape[1], feat_a.shape[1], feat_s.shape[1]]
    u = torch.LongTensor(np.array(index_1)).to(device)
    v = torch.LongTensor(np.array(index_2)).to(device)
    edge_index = [index_1, index_2]
    g = dgl.graph((u, v))
    #g.edata['t'] = types

    return adj, [feat_p, feat_a, feat_s], feat_dims, idx_train, idx_val, idx_test, num_classes, label, g
