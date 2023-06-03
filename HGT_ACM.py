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

for edge in pa:
    edge[1] = edge[1] + 4019
for edge in ps:
    edge[1] = edge[1] + 4019 + 7167

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : pa.nonzero(),
        ('author', 'writing', 'paper') : pa.transpose().nonzero(),
#        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
#        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : ps.nonzero(),
        ('subject', 'has', 'paper') : ps.transpose().nonzero(),
    })
print(G)

