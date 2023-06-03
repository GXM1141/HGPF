from models.CPF import PLP
from data.utils import normalize_adj_1
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import networkx as nx
from scipy import sparse
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from load_data import load_acm, load_dblp, load_imdb
import time

