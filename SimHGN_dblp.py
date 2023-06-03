from data.utils import load_data
import numpy as np
import scipy
import networkx as nx
import dgl
import scipy.sparse as sp
import pickle
import torch


path = "data/dblp/"

pa = np.loadtxt(path + "pa.txt")
pc = np.loadtxt(path + "pc.txt")
pt = np.loadtxt(path + "pt.txt")
type_num = [4057, 14328, 7723, 20]
num_nodes = 4057 + 14328 + 7723 + 20
#            a(tar)  p      t      c
print (pa.shape)#19645
print (pc.shape)#14328
print (pt.shape)#85810

num_edges = [pa.shape[0], pc.shape[0], pt.shape[0], pa.shape[0], pc.shape[0], pt.shape[0]]

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


u = torch.cat([pa_u, pt_u, pc_u, pa_v, pt_v, pc_v, self_u], dim = -1)
v = torch.cat([pa_v, pt_v, pc_v, pa_u, pt_u, pc_u, self_u], dim = -1)

types = []

for i in range(len(num_edges)):
    t = torch.zeros(num_edges[i])
    t = t + i
    types.append(t.int())
t = torch.zeros(num_nodes)
t = t + 6
types.append(t.int())
types = torch.cat(types, dim = -1)

print (types)

g = dgl.graph((u, v))

g.edata['t'] = types

print (g)


