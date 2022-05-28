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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

window_size = 3

data_path = 'data/'
models_path = 'out/'
out_path = 'out/'

window_size = 3
train_split = 0.8
lead_time = 1
optimizer = 'SGD'
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.0001
batch_size = 64
num_sample = 1677

# Load the model.

# Define the model before loading.
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

model = GCN(window_size, 200, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

checkpoint = torch.load(models_path + 'checkpoint_GCN_SSTAGraphDataset_windowsize_3_leadtime_1_numsample_1677_trainsplit_0.8_numepoch_100.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Define the graph and create the dataloaders again.

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

# DGL Edge feature matrix
w = []
for i in range(node_num):
  for j in range(node_num):
    if i != j:
      w.append(edge_features[i][j])

graph.edata['w'] = torch.tensor(w)

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
          
          y_temp = y[i+window_size+lead_time-1]
          self.ys.append(y_temp)

    def __getitem__(self, i):
        return self.graphs[i], self.ys[i]

    def __len__(self):
        return len(self.graphs)

dataset = SSTAGraphDataset()

# Create data loaders.
num_examples = len(dataset)
num_train = int(num_examples * train_split)

train_dataloader = GraphDataLoader(dataset, sampler=torch.arange(num_train), batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(dataset, sampler=torch.arange(num_train, num_examples), batch_size=1, drop_last=False)

# Test the model.

preds = []
ys = []
for batched_graph, y in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'])
    print('Observed:', y.cpu().detach().numpy().squeeze(axis=0), '; predicted:', pred.cpu().detach().numpy().squeeze(axis=0))
    preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
    ys.append(y.cpu().detach().numpy().squeeze(axis=0))

test_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
test_rmse = mean_squared_error(np.array(ys), np.array(preds), squared=False)

print("----------")
print()

print('Final validation / test MSE:', test_mse)
print("----------")
print()

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel('Month')
plt.ylabel('SSTA')
plt.title('GCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '_MSE_' + str(round(test_mse, 4)), fontsize=12)
blue_patch = mpatches.Patch(color='blue', label='Predicted')
red_patch = mpatches.Patch(color='red', label='Observed')
ax.legend(handles=[blue_patch, red_patch])
month = np.arange(0, len(ys), 1, dtype=int)
ax.plot(month, np.array(preds), 'o', color='blue')
ax.plot(month, np.array(ys), 'o', color='red')
plt.savefig(out_path + 'plot_GCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(epoch) + '.png')

print("Save the observed vs. predicted plot.")
print("--------------------")
print()