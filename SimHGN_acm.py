from data.utils import load_data
import numpy as np
import scipy
import networkx as nx
import dgl
import scipy.sparse as sp
import pickle
import torch


path = "data/acm/"

pa = np.loadtxt(path + "pa.txt")
ps = np.loadtxt(path + "ps.txt")
type_num = [4019, 7167, 60]
num_nodes = 4019 + 7167 + 60

print (pa)

num_edges = [pa.shape[0], ps.shape[0], pa.shape[0], ps.shape[0]]

pa_u = []
pa_v = []

for edge in pa:
    pa_u.append(int(edge[0]))
    pa_v.append(int(edge[1] + 4019))

pa_u = torch.from_numpy(np.array(pa_u))
pa_v = torch.from_numpy(np.array(pa_v))


ps_u = []
ps_v = []

for edge in ps:
    ps_u.append(int(edge[0]))
    ps_v.append(int(edge[1] + 4019 + 7167))

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

print (types)

g = dgl.graph((u, v))

g.edata['t'] = types

print (g)


