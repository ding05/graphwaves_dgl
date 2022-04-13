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
from dgl import save_graphs
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
num_sample = 200 # max: node_features.shape[1]-window_size-lead_time+1
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
print("--------------------")
print()

# Create a DGL graph dataset.

class SSTAGraphDataset(DGLDataset):
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
          
          y_temp = y[i+window_size-lead_time+1]
          self.ys.append(y_temp)

    def __getitem__(self, i):
        return self.graphs[i], self.ys[i]

    def __len__(self):
        return len(self.graphs)
    
    #def save(self):
    #    save_graphs(data_path + 'graphs_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_trainsplit_' + str(train_split) + '.bin', self.graphs, self.labels)

dataset = SSTAGraphDataset()

print('Create a DGL dataset: SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_trainsplit_' + str(train_split))
print("The last graph and its label:")
print(dataset[-1])
print("--------------------")
print()

# Create data loaders.

num_examples = len(dataset)
num_train = int(num_examples * train_split)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

#train_sampler = Sampler(torch.arange(num_train))
#test_sampler = Sampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

it = iter(train_dataloader)
batch = next(it)
print("A batch:")
print(batch)
print("----------")
print()

batched_graph, y = batch
print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())
print("----------")
print()

graphs = dgl.unbatch(batched_graph)
print('The original graphs in the minibatch:')
print(graphs)
print("--------------------")
print()

# Set up a GCN.

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)
        self.double()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# Train the GCN.

model = GCN(window_size, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_f = nn.MSELoss()

print("Start training.")
print("----------")
print()

# Start time
start = time.time()

for epoch in range(num_train_epoch):
    print("Epoch " + str(epoch))
    print("----------")
    print()
    for batched_graph, y in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'])
        #print('Predicted y:', pred.cpu().detach().numpy())
        #print('Observed y:', y.cpu().detach().numpy())
        loss = loss_f(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print("----------")
    #print()

torch.save(model.state_dict(), models_path + 'model_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.pt')

# End time
stop = time.time()

print(f'Complete training. Time spent: {stop - start} seconds.')
print("----------")
print()


print("Save the model in a PT file.")
print("----------")
print()

sum_mse = 0
num_tests = 0
for batched_graph, y in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'])
    sum_mse += (pred - y) ** 2
    num_tests += 1

print('Test MSE:', sum_mse / num_tests)
print("--------------------")
print()