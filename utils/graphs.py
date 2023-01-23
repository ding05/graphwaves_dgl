import numpy as np
from numpy import load

import torch

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs

data_path = 'data/'
models_path = 'out/'
out_path = 'out/'

window_size = 5
lead_time = 1
num_sample = 1680-window_size-lead_time+1

# Transform the graphs into the DGL forms.

node_features = load(data_path + 'node_features.npy')
edge_features = load(data_path + 'edge_features.npy')
y = load(data_path + 'y.npy')

# Normalize the data.

node_features = node_features / np.linalg.norm (node_features)
edge_features = edge_features / np.linalg.norm (edge_features)
y = y / np.linalg.norm (y)

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
graph.ndata['feat'] = torch.tensor(node_features)

print('DGL node feature matrix:')
print(graph.ndata['feat'])
print('Shape of DGL node feature matrix:')
print(graph.ndata['feat'].shape)
print('----------')
print()

# DGL Edge feature matrix
w = []
for i in range(node_num):
  for j in range(node_num):
    if i != j:
      w.append(edge_features[i][j])

graph.edata['w'] = torch.tensor(w)

print('DGL edge feature matrix:')
print(graph.edata['w'])
print('Shape of DGL edge feature matrix:')
print(graph.edata['w'].shape)

print('----------')
print()

save_graphs(data_path + 'graph.bin', graph)

print('Save the graph in a BIN file.')
print('--------------------')
print()

class SSTAGraphDataset(DGLDataset):
    """
    Create a DGL graph dataset.
    """
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
    
        self.graphs = []
        self.ys = []
        
        for i in range(num_sample):
          graph_temp = dgl.graph((u, v))
          graph_temp.ndata['feat'] = torch.tensor(node_features[:, i:i+window_size])
          graph_temp.edata['w'] = graph.edata['w']
          self.graphs.append(graph_temp)
          
          y_temp = y[i+window_size+lead_time-1]
          self.ys.append(y_temp)

    def __getitem__(self, i):
        return self.graphs[i], self.ys[i]

    def __len__(self):
        return len(self.graphs)

class SSTAGraphDataset_NodeLabels(DGLDataset):
    """
    Create a DGL graph dataset.
    """
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
    
        self.graphs = []
        
        for i in range(num_sample):
          graph_temp = dgl.graph((u, v))
          graph_temp.ndata['feat'] = torch.tensor(node_features[:, i:i+window_size])
          graph_temp.ndata['label'] =  torch.tensor(node_features[:, i+window_size+lead_time-1])
          graph_temp.edata['w'] = graph.edata['w']
          self.graphs.append(graph_temp)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)