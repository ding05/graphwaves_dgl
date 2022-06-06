# The script is not working yet.

import numpy as np
from numpy import load
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import dgl
import dgl.nn as dglnn
import dgl.data
import dgl.function as fn
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl import save_graphs
from dgl.dataloading import GraphDataLoader

import time

from sklearn.metrics import mean_squared_error

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

for lead_time in [1]: # [1, 2, 3, 6, 12, 23]

    # GNN configurations
    
    net_class = 'GCN' # 'GAT'
    num_layer = '3'
    num_hid_feat = 200
    num_out_feat = 100
    num_head = 10
    window_size = 5
    train_split = 0.8
    #lead_time = 1
    loss_function = 'Huber' # 'MSE', 'MAE', 'Huber'
    activiation = 'lrelu' # 'relu', 'tanh' 
    optimizer = 'SGD' # Adam
    learning_rate = 0.02 # 0.05, 0.02, 0.01
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
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
              
              y_temp = y[i+window_size+lead_time-1]
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
    
    # Sequential sampler
    train_dataloader = GraphDataLoader(dataset, sampler=torch.arange(num_train), batch_size=batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(dataset, sampler=torch.arange(num_train, num_examples), batch_size=1, drop_last=False)
    
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
    
    # Set up GATs, ref: https://www.dgl.ai/blog/2019/02/17/gat.html
    
    class GATLayer(nn.Module):
        def __init__(self, g, in_dim, out_dim):
            super(GATLayer, self).__init__()
            self.g = g
            # equation (1)
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
            # equation (2)
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        
        def edge_attention(self, edges):
            # edge UDF for equation (2)
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)
            return {'e' : F.leaky_relu(a)}
        
        def message_func(self, edges):
            # message UDF for equation (3) & (4)
            return {'z' : edges.src['z'], 'e' : edges.data['e']}
        
        def reduce_func(self, nodes):
            # reduce UDF for equation (3) & (4)
            # equation (3)
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            # equation (4)
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            return {'h' : h}

    class MultiHeadGATLayer(nn.Module):
        def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
            super(MultiHeadGATLayer, self).__init__()
            self.heads = nn.ModuleList()
            for i in range(num_heads):
                self.heads.append(GATLayer(g, in_dim, out_dim))
            self.merge = merge
        
        def forward(self, h):
            head_outs = [attn_head(h) for attn_head in self.heads]
            if self.merge == 'cat':
                # concat on the output feature dimension (dim=1)
                return torch.cat(head_outs, dim=1)
            else:
                # merge using average
                return torch.mean(torch.stack(head_outs))

    class GAT(nn.Module):
        def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
            super(GAT, self).__init__()
            self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
            # Be aware that the input dimension is hidden_dim*num_heads since
            #   multiple head outputs are concatenated together. Also, only
            #   one attention head in the output layer.
            self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, hidden_dim, num_heads)
            self.layer3 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        
        def forward(self, h):
            h = self.layer1(h)
            h = act_f(h)
            h = self.layer2(h)
            h = act_f(h)
            h = self.layer3(h)
            return h
    
    # Train the GAT.
    
    model = GAT(dataset, window_size, num_hid_feat, num_out_feat, num_head)
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    if loss_function == 'MAE':
        loss_f = nn.L1Loss()
    elif loss_function == 'Huber':
        loss_f = nn.HuberLoss()
    else:
        loss_f = nn.MSELoss()
    
    if activiation == 'lrelu':
        act_f = nn.LeakyReLU(0.1)
    elif activiation == 'tanh':
        act_f = nn.Tanh()
    else:
        act_f = nn.ReLu()
    
    print("Start training.")
    print("----------")
    print()
    
    # Start time
    start = time.time()
    
    all_loss = []
    all_eval = []
    
    for epoch in range(num_train_epoch):
        print("Epoch " + str(epoch+1))
        print()
    
        losses = []
        for batched_graph, y in train_dataloader:
            pred = model(batched_graph.ndata['feat'])
            loss = loss_f(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.cpu().detach().numpy())
        print('Training loss:', sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
    
        preds = []
        ys = []
        for batched_graph, y in test_dataloader:
            pred = model(batched_graph.ndata['feat'])
            preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
            ys.append(y.cpu().detach().numpy().squeeze(axis=0))
        val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
        print('Validation MSE:', val_mse)
        all_eval.append(val_mse)
    
        print("----------")
        print()
    
    # End time
    stop = time.time()
    
    print(f'Complete training. Time spent: {stop - start} seconds.')
    print("----------")
    print()

    # Save the model.
    
    torch.save({
                'epoch': num_train_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
                }, models_path + 'checkpoint_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activiation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.tar')
    
    print("Save the checkpoint in a TAR file.")
    print("----------")
    print()