import numpy as np
from numpy import load
import random

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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import time

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GCN configurations

GCN_structure = ['', '']
window_size = 3
train_split = 0.8
lead_time = 1
loss_function = 'MSE'
optimizer = 'SGD' # Adam
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.0001
batch_size = 64
num_sample = 1677 # max: node_features.shape[1]-window_size-lead_time+1
num_train_epoch = 100

data_path = 'data/'
models_path = 'out/'
out_path = 'out/'

# Transform the graphs into the DGL forms.

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

dataset = SSTAGraphDataset()

print('Create a DGL dataset: SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_trainsplit_' + str(train_split))
print("The last graph and its label:")
print(dataset[-1])
print("--------------------")
print()

# Create data loaders.

num_examples = len(dataset)
num_train = int(num_examples * train_split)

# Random sampler
#train_sampler = SubsetRandomSampler(torch.arange(num_train))
#test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

# Sequential sampler
train_sampler = SequentialSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=1, drop_last=False)

it = iter(train_dataloader)
batch = next(it)
print("A batch in the traning set:")
print(batch)
print("----------")
print()

it = iter(test_dataloader)
batch = next(it)
print("A batch in the test set:")
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
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, out_feats)
        self.double()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# Train the GCN.

model = GCN(window_size, 200, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
loss_f = nn.MSELoss()

print("Start training.")
print("----------")
print()

# Start time
start = time.time()

for epoch in range(num_train_epoch):
    print("Epoch " + str(epoch))
    print()

    losses = []
    for batched_graph, y in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'])
        #print('Predicted y:', pred.cpu().detach().numpy())
        #print('Observed y:', y.cpu().detach().numpy())
        loss = loss_f(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    print('Training loss:', sum(losses) / len(losses))

    preds = []
    ys = []
    for batched_graph, y in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'])
        preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
        ys.append(y.cpu().detach().numpy().squeeze(axis=0))
    val_rmse = mean_squared_error(np.array(preds), np.array(ys), squared=True)
    print('Validation RMSE:', val_rmse)

    print("----------")
    print()

# End time
stop = time.time()

print(f'Complete training. Time spent: {stop - start} seconds.')
print("----------")
print()

#torch.save(model.state_dict(), models_path + 'model_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.pt')

#print("Save the model in a PT file.")
#print("----------")
#print()

torch.save({
            'epoch': num_train_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, models_path + 'checkpoint_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.tar')

print("Save the checkpoint in a TAR file.")
print("----------")
print()

preds = []
ys = []
for batched_graph, y in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'])
    print('Observed:', y.cpu().detach().numpy().squeeze(axis=0), '; predicted:', pred.cpu().detach().numpy().squeeze(axis=0))
    preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
    ys.append(y.cpu().detach().numpy().squeeze(axis=0))

test_mse = mean_squared_error(np.array(preds), np.array(ys), squared=False)
test_rmse = mean_squared_error(np.array(preds), np.array(ys), squared=True)

print("----------")
print()

print('Final validation / test RMSE:', test_rmse)
print("----------")
print()

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel('Month')
plt.ylabel('SSTA')
plt.title('GNN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch), fontsize=12)
blue_patch = mpatches.Patch(color='blue', label='Predicted')
red_patch = mpatches.Patch(color='red', label='Observed')
ax.legend(handles=[blue_patch, red_patch])
month = np.arange(0, len(ys), 1, dtype=int)
ax.plot(month, np.array(preds), 'o', color='blue')
ax.plot(month, np.array(ys), 'o', color='red')
plt.savefig(out_path + 'plot_GNN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.png')

print("Save the observed vs. predicted plot.")
print("--------------------")
print()