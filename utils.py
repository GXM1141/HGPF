import torch
import numpy as np
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC


def accuracy(output, labels, details=False, hop_idx=None):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()
    if details:
        hop_num = np.bincount(hop_idx, minlength=7)
        true_idx = np.array((correct > 0).nonzero().squeeze(dim=1).cpu())
        true_hop = np.bincount(hop_idx[true_idx], minlength=7)/hop_num
        return result / len(labels), true_hop
    return result / len(labels)

def eucli_dist(output, target):
    return torch.sqrt(torch.sum(torch.pow((output - target), 2)))

def my_loss(output, target, mode=0):
    if mode == 0:
        return eucli_dist(torch.exp(output), target)
    elif mode == 1:
        # Distilling the Knowledge in a Neural Network
        return torch.nn.BCELoss()(torch.exp(output), target)
    elif mode == 2:
        # Exploring Knowledge Distillation of Deep Neural Networks for Efficient Hardware Solutions
        return torch.nn.KLDivLoss()(output, target)
    # output = F.log_softmax(output, dim=1)
    # return torch.mean(-torch.sum(torch.mul(target, output), dim=1))
    # Cross Entropy Error Function without mean
    # return -torch.sum(torch.mul(target, output))
    # return torch.sum(F.pairwise_distance(torch.exp(output), target, p=2))

def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        #row_parsed = list(map(int, row.split(' ')))#以空格分割，转换为整数后，变为list，数据中第一个节点是当前节点的index
        row_parsed = row
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
        a = [adjlist[i] for i in idx_batch]
        b = [indices[i] for i in idx_batch]
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            a, b, samples)

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


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size)) #ceil向上取整

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter #剩余的batch数

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
