import torch
import time
import dgl
import numpy as np
import scipy.sparse as sp
import scipy.sparse
import pickle
from sklearn.preprocessing import OneHotEncoder
import networkx as nx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def quick_matrix_pow(m, n):
    t = time.time()
    E = torch.eye(len(m))
    while n:
        if n % 2 != 0:
            E = torch.matmul(E, m)
        m = torch.matmul(m, m)
        n >>= 1
    print(time.time() - t)
    return E

def row_normalize(data):
    return (data.t() / torch.sum(data.t(), dim=0)).t()

def np_normalize(matrix):
    from sklearn.preprocessing import normalize
    """Normalize the matrix so that the rows sum up to 1."""
    matrix[np.isnan(matrix)] = 0
    return normalize(matrix, norm='l1', axis=1)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))#度数向量d
    r_inv = np.power(rowsum, -1).flatten()#d的逆
    r_inv[np.isinf(r_inv)] = 0.#去掉无效值
    r_mat_inv = sp.diags(r_inv)#以对角线生成稀疏矩阵度数矩阵的逆D-1
    mx = r_mat_inv.dot(mx)#乘以A得到归一化的邻接矩阵
    return mx

def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))#归一化的邻接矩阵加自环
    return adj

def normalize_features(features):
    features = normalize(features)
    return features

def initialize_label(idx_train, labels_one_hot):
    labels_init = torch.ones_like(labels_one_hot) / len(labels_one_hot[0])
    labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init

def preprocess_adj(adj):
    return normalize_adj(adj)

def preprocess_features_1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def preprocess_features(features):
    return features

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def normalize_adj_1(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_imdb(ratio, type_num, device):
    path = "data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float()
    label = label.nonzero()[:, 1]
    G00 = nx.read_adjlist(path + '/IMDB_processed/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(path + '/IMDB_processed/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(path + '/IMDB_processed/0/0-1-0_idx.npy')
    idx01 = np.load(path + '/IMDB_processed/0/0-2-0_idx.npy')
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = [torch.LongTensor(i).to(device) for i in nei_a]
    nei_d = [torch.LongTensor(i).to(device) for i in nei_d]
    feat_m = sp.load_npz(path + "m_feat.npz")
    feat_d= sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")
    with open('data/imdb'+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    #feat_m = sp.csr_matrix(feat_m)
    feat_m = torch.FloatTensor(feat_m.todense())

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a))
    feat_d = torch.FloatTensor(preprocess_features_1(feat_d))

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
    type_mask = np.load(path + '/node_types.npy')
    #g = [mam_g, mdm_g]
    g[0] = g[0]
    g[1] = g[1]
    idx_train = torch.LongTensor(train).squeeze(0)
    idx_val = torch.LongTensor(val).squeeze(0)
    idx_test = torch.LongTensor(test).squeeze(0)
    label_init = initialize_label(idx_train, labels_one_hot)
    return [[G00, G01]], [[idx00, idx01]], g, [nei_d, nei_a], [feat_m, feat_d, feat_a], [mam, mdm], label, idx_train, idx_val, idx_test, labels_one_hot, label_init, type_mask, edges

def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))#以空格分割，转换为整数后，变为list，数据中第一个节点是当前节点的index
        nodes.add(row_parsed[0])#当前节点，
        if len(row_parsed) > 1:#采样邻居
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)#去重复邻居，返回count是每个重复数的重复次数
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()#概率归一化，重复次数越多采样概率越大
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g = g.to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list

def load_dblp(ratio, type_num, device):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "data/dblp/"
    
    in_file = open(path + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
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
    idx0 = pickle.load(in_file)
    in_file.close()
    in_file = open(path + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx1 = pickle.load(in_file)
    in_file.close()
    in_file = open(path + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx2 = pickle.load(in_file)
    in_file.close()
    
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label).to(device)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)

    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    feat_t = sp.eye(type_num[2])
    feat_c = sp.eye(20)
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")
    
    nei_p = [torch.LongTensor(i).to(device) for i in nei_p]
    feat_a = sp.csr_matrix(feat_a)
    feat_a = torch.from_numpy(feat_a.todense()).float().to(device)
    feat_p = torch.FloatTensor(preprocess_features_1(feat_p)).to(device)
    feat_t = torch.FloatTensor(preprocess_features_1(feat_t)).to(device)
    feat_c = torch.FloatTensor(preprocess_features_1(feat_c)).to(device)
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
    type_mask = np.load(path + '/node_types.npy')
    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    label_init = initialize_label(idx_train, labels_one_hot).to(device)

    return  [adjlist00, adjlist01, adjlist02], \
            [idx0, idx1, idx2], g, [nei_p], [feat_a, feat_p, feat_t, feat_c], [apa, apcpa, aptpa], \
            label, idx_train, idx_val, idx_test, labels_one_hot, label_init, type_mask, 0

def load_acm(ratio, type_num, device):
    path = "data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.from_numpy(label)
    labels_one_hot = label.float().to(device)
    label = label.nonzero()[:, 1].to(device)
    G00 = nx.read_adjlist(path + '/G_pap.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(path + '/G_psp.adjlist', create_using=nx.MultiDiGraph)
    adj00 = np.load(path + '/adj00.npy', allow_pickle=True)
    adj01 = np.load(path + '/adj01.npy', allow_pickle=True)
    idx00 = np.load(path + '/idx00.npy')
    idx01 = np.load(path + '/idx01.npy')
    idx0 = np.load(path + '/idx0.npy', allow_pickle=True)
    idx1 = np.load(path + '/idx1.npy', allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    train = np.load(path + "train_" + str(ratio) + ".npy")
    test = np.load(path + "test_" + str(ratio) + ".npy")
    val = np.load(path + "val_" + str(ratio) + ".npy")
    with open('data/acm'+'/edges.pkl','rb') as f:
        edges = pickle.load(f)

    nei_a = [torch.LongTensor(i).to(device) for i in nei_a]
    nei_s = [torch.LongTensor(i).to(device) for i in nei_s]
    feat_p = sp.csr_matrix(feat_p)
    feat_p = torch.from_numpy(feat_p.todense()).float()

    feat_a = torch.FloatTensor(preprocess_features_1(feat_a))
    feat_s = torch.FloatTensor(preprocess_features_1(feat_s))

    pap = preprocess_adj(pap)
    psp = preprocess_adj(psp)
    pap_g = dgl.from_scipy(pap)
    psp_g = dgl.from_scipy(psp)
    g = [pap_g, psp_g]
    """
    g[0] = g[0].to(device)
    g[1] = g[1].to(device)
    """
    idx_train = torch.LongTensor(train).squeeze(0).to(device)
    idx_val = torch.LongTensor(val).squeeze(0).to(device)
    idx_test = torch.LongTensor(test).squeeze(0).to(device)
    label_init = initialize_label(idx_train, labels_one_hot).to(device)
    type_mask = np.load(path + '/node_types.npy')
    return [[G00, G01]], [[idx00, idx01]], g, [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, idx_train, idx_val, idx_test, labels_one_hot, label_init, type_mask, edges
    #return [adj00, adj01], [idx0, idx1], g, [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, idx_train, idx_val, idx_test, labels_one_hot, label_init, type_mask

def load_data(dataset, ratio, type_num, device):
    if dataset == 'ACM':
        return load_acm(ratio, type_num, device)
    if dataset == 'DBLP':
        return load_dblp(ratio, type_num, device)
    if dataset == 'IMDB':
        return load_imdb(ratio, type_num, device)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))
