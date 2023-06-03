from data.utils import load_acm
import numpy as np
import networkx as nx
import scipy

path = "data/acm/"
nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
l = 5
sc_inst = []
for i in range(l):
    sc_inst.append([])

for i in range(nei_a.shape[0]):
    if len(nei_a[i]) >= l:
        select_a = np.random.choice(nei_a[i], l, replace=False) + 4019
    else:
        select_a = np.random.choice(nei_a[i], l, replace=True) + 4019
    if len(nei_s[i]) >= l:
        select_s = np.random.choice(nei_s[i], l, replace=False) + 4019 + 7167
    else:
        select_s = np.random.choice(nei_s[i], l, replace=True) + 4019 + 7167
    sc = []
    for i in range (l):
        sc_inst[i].append([select_a[i], select_s[i]])

print (sc_inst)

np.save(path + 'sc_instance.npy', sc_inst)







