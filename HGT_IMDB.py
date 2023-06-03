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

ma_num = 0
md_num = 0
md = []
ma = []
for i in range(4278):
    for j in range(4278, 4278 + 2081):
        if (adj[i, j] != 0):
            md.append([i, j])
            md_num = md_num + 1

#ma, am
for i in range(4278):
    for j in range(4278 + 2081, 4278 + 2081 + 5257):
        if (adj[i, j] != 0):
            ma.append([i, j])
            ma_num = ma_num + 1

np.savetxt(path + "ma.txt", ma)
np.savetxt(path + "md.txt", md)

