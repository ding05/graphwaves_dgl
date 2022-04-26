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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

for lead_time in [1, 2, 3, 6, 12, 23]:

    # GCN configurations
    
    GCN_structure = ['', '']
    window_size = 3
    train_split = 0.8
    #lead_time = 1
    loss_function = 'MSE'
    optimizer = 'SGD' # Adam
    learning_rate = 0.01 # 0.05
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
    num_train_epoch = 20
    
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
    
    # Set up GCNs.
    
    # GCN without applying edge features
    class GCN(nn.Module):
        def __init__(self, in_feats, h_feats, out_feats):
            super(GCN, self).__init__()
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, out_feats)
            self.double()
    
        def forward(self, g, in_feat, edge_feat=None):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            g.ndata['h'] = h
            return dgl.mean_nodes(g, 'h')
    
    # GCN that uses edge features, ref: https://discuss.dgl.ai/t/using-edge-features-for-gcn-in-dgl/427/
    class GNNLayer(nn.Module):
        def __init__(self, ndim_in, edims, ndim_out, activation):
            super(GNNLayer, self).__init__()
            self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
            self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
            self.activation = activation
            self.double()
    
        def message_func(self, edges):
            #print("edges.src['h']:", edges.src['h'].shape) ###
            #print(edges.src['h']) ###
            #print("edges.data['h']:", torch.unsqueeze(edges.data['h'], dim=1).shape) ###
            #print(torch.unsqueeze(edges.data['h'], dim=1)) ###
            return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], torch.unsqueeze(edges.data['h'], dim=1)], 1)))}
    
        def forward(self, g_dgl, nfeats, efeats):
            with g_dgl.local_scope():
                g = g_dgl
                g.ndata['h'] = nfeats
                g.edata['h'] = efeats
                g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
                #print("g.ndata['h']:", g.ndata['h'].shape) ###
                #print(g.ndata['h']) ###
                #print("g.ndata['h_neigh']:", g.ndata['h_neigh'].shape) ###
                #print(g.ndata['h_neigh']) ###
                g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1)))
                return g.ndata['h']
    
    class GCN2(nn.Module):
        def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
            super(GCN2, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GNNLayer(ndim_in, edim, 32, activation))
            self.layers.append(GNNLayer(32, edim, 32, activation))
            self.layers.append(GNNLayer(32, edim, ndim_out, activation))
            self.dropout = nn.Dropout(p=dropout)
    
        def forward(self, g, nfeats, efeats):
            for i, layer in enumerate(self.layers):
                if i != 0:
                    nfeats = self.dropout(nfeats)
                nfeats = layer(g, nfeats, efeats)
            return nfeats.sum(1)
    
    """
    if __name__ == '__main__':
        model = GCN2(3, 1, 1, F.relu, 0.5)
        g = dgl.DGLGraph([[0, 2], [2, 3]])
        nfeats = torch.randn((g.number_of_nodes(), 3, 3))
        efeats = torch.randn((g.number_of_edges(), 3, 3))
        model(g, nfeats, efeats)
    """
    
    # Train the GCN.
    
    model = GCN(window_size, 200, 1)
    #model = GCN2(window_size, 1, 1, F.relu, 0.5)
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
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w'])
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
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w']) ###
            #print('pred:', pred) ###
            preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
            ys.append(y.cpu().detach().numpy().squeeze(axis=0))
        val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
        print('Validation MSE:', val_mse)
    
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
                }, models_path + 'checkpoint_GCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.tar')
    
    print("Save the checkpoint in a TAR file.")
    print("----------")
    print()
    
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
    plt.savefig(out_path + 'plot_GCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.png')
    
    print("Save the observed vs. predicted plot.")
    print("--------------------")
    print()