import numpy as np
from numpy import load

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
import dgl.data
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

import random

import time

# GCN configurations

GCN_structure = ['', '']
window_size = 3
train_split = 0.8
lead_time = 1
loss_function = 'MSE'
optimizer = 'SGD'
learning_rate = 0.005
momentum = 0.9
batch_size = 16 # Crashed at 32
num_sample = 1677 # max: node_features.shape[1]-window_size-lead_time+1
num_train_epoch = 20

# Transform the graphs into the DGL forms.

data_path = 'data/'
models_path = 'models/'

node_features = load(data_path + 'node_features.npy')
edge_features = load(data_path + 'edge_features.npy')
y = load(data_path + 'y.npy')

node_num = node_features.shape[0]

# DGL Node feature matrix structure
u = []
v = []
for i in range(node_num):
  for j in range(node_num):
    if j == i:
      pass
    else:
      u.append(i)
      v.append(j)
u = torch.tensor(u)
v = torch.tensor(v)
graph = dgl.graph((u, v))

# DGL Node feature matrix
graph.ndata["feat"] = torch.tensor(node_features)

print("DGL node feature matrix:")
print(graph.ndata["feat"])
print("Shape of DGL node feature matrix:")
print(graph.ndata['feat'].shape)
print("----------")
print()

# DGL Edge feature matrix
w = []
for i in range(node_num):
  for j in range(node_num):
    if i != j:
      w.append(edge_features[i][j])

graph.edata['w'] = torch.tensor(w)

print("DGL edge feature matrix:")
print(graph.edata['w'])
print("Shape of DGL edge feature matrix:")
print(graph.edata['w'].shape)

print("----------")
print()

save_graphs(data_path + 'graph.bin', graph)

print("Save the graph in a BIN file.")
print("--------------------")
print()

graph_1 = load_graphs(data_path + 'graph.bin')
print(graph_1)