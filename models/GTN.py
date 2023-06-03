import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, in_dims, out_dim, num_class,num_layers,norm,device):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.device = device
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weights = []
        for in_dim in in_dims:    
            self.weights.append(nn.Parameter(torch.Tensor(in_dim, out_dim)).to(device))
        self.w = nn.Parameter(torch.Tensor(out_dim, out_dim)).to(device)
        self.bias = nn.Parameter(torch.Tensor(out_dim)).to(device)
        self.loss = nn.CrossEntropyLoss().to(device)
        self.linear1 = nn.Linear(self.out_dim*self.num_channels, self.out_dim).to(device)
        self.linear2 = nn.Linear(self.out_dim, self.num_class).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, feats_list, H):
        for i in range(len(feats_list)):
            if i == 0:
                X = torch.mm(feats_list[i], self.weights[i])
            else:
                X = torch.cat([X, torch.mm(feats_list[i], self.weights[i])], dim = 0)
        X = X.to(self.device)
        X = torch.mm(X, self.w)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        H = H.to(self.device)
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).to(self.device))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).to(self.device)) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(self.device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).to(self.device)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_)
        return y

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
