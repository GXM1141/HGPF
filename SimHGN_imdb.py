from data.utils import load_data
import numpy as np
import scipy
import networkx as nx
import dgl
import scipy.sparse as sp
import pickle
import torch


path = "data/imdb/"

adj = sp.load_npz(path + 'adjM.npz')

type_num = [4278, 2081, 5257]

ma_u, ma_v, md_u, md_v = [], [], [], []

ma_num = 0
md_num = 0

#md, dm

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
print (edges_num)

ma_u = torch.from_numpy(np.array(ma_u))
ma_v = torch.from_numpy(np.array(ma_v))
md_u = torch.from_numpy(np.array(md_u))
md_v = torch.from_numpy(np.array(md_v))

u = [ma_u, md_u, ma_v, md_v]
v = [ma_v, md_v, ma_u, md_u]

u = torch.cat(u, dim = -1)
v = torch.cat(v, dim = -1)
print (u.shape)

types = []

for i in range(len(edges_num)):
    t = torch.zeros(edges_num[i], dtype = torch.int)
    t = t + i
    types.append(t)

types = torch.cat(types, dim = -1)
print (types.shape)

g = dgl.graph((u, v), num_nodes = (4278 + 2081 + 5257))

g.edata['t'] = types

print (g)