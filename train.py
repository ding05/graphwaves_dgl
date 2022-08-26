from utilities.losses import *
from utilities.graphs import *
from utilities.gnns import *

import numpy as np
from numpy import load
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.autograd import Variable

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

for loss_function in ['MSE', 'MAE', 'Huber', 'WMSE', 'WMAE', 'WHuber', 'WFMSE', 'WFMAE', 'BMSE']:

    # GNN configurations
    
    net_class = 'GCN' # 'GAT'
    num_layer = 3
    num_hid_feat = 200
    num_out_feat = 100
    window_size = 5
    train_split = 0.8
    lead_time = 1
    #loss_function = 'BMSE' # 'MSE', 'MAE', 'Huber', 'WMSE', 'WMAE', 'WHuber', 'WFMSE', 'WFMAE', 'BMSE
    activation = 'lrelu' # 'relu', 'tanh' 
    optimizer = 'SGD' # Adam
    learning_rate = 0.02 # 0.05, 0.02, 0.01
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
    num_train_epoch = 20
    
    data_path = 'data/'
    models_path = 'out/'
    out_path = 'out/'
    
    # Create a DGL graph dataset.
    
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
    
    # Train the GCN.
    
    model = GCN(window_size, num_hid_feat, num_out_feat)
    #model = GCN2(window_size, 1, 1, F.relu, 0.5)
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    
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
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w'])
            # Set a loss function.
            if loss_function == 'MSE':
                loss = mse(pred, y)
            elif loss_function == 'MAE':
                loss = mae(pred, y)
            elif loss_function == 'Huber':
                loss = huber(pred, y)
            elif loss_function == 'WMSE':
                loss = weighted_mse(pred, y)
            elif loss_function == 'WMAE':
                loss = weighted_mae(pred, y)
            elif loss_function == 'WHuber':
                loss = weighted_huber(pred, y)                
            elif loss_function == 'WFMSE':
                loss = weighted_focal_mse(pred, y)  
            elif loss_function == 'WFMAE':
                loss = weighted_focal_mae(pred, y)              
            elif loss_function == 'BMSE':
                loss = balanced_mse(pred, y)
            else:
                pass
            
            """
            # Add L1 regularization.
            l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            target = Variable(torch.from_numpy(np.zeros((200,100))))
            for param in model.parameters():
              reg_loss += l1_crit(param, target)
            factor = 0.0005
            loss += factor * reg_loss
            """
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.cpu().detach().numpy())
        print('Training loss:', sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
    
        preds = []
        ys = []
        for batched_graph, y in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w']) ###
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
                }, models_path + 'checkpoint_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.tar')
    
    print("Save the checkpoint in a TAR file.")
    print("----------")
    print()

    # Test the model.
    
    preds = []
    ys = []
    for batched_graph, y in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'])
        preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
        ys.append(y.cpu().detach().numpy().squeeze(axis=0))
    
    test_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
    test_rmse = mean_squared_error(np.array(ys), np.array(preds), squared=False)
    
    print('Final validation / test MSE:', test_mse)
    print("----------")
    print() 
    
    # Show the results.

    all_loss = np.array(all_loss)
    all_eval = np.array(all_eval)
    all_epoch = np.array(list(range(1, num_train_epoch+1)))

    all_perform_dict = {
      'training_time': str(stop-start),
      'all_loss': all_loss.tolist(),
      'all_eval': all_eval.tolist(),
      'all_epoch': all_epoch.tolist()}

    with open(out_path + 'perform_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.txt', "w") as file:
        file.write(json.dumps(all_perform_dict))

    print("Save the performance in a TXT file.")
    print("----------")
    print()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel('Month')
    plt.ylabel('SSTA')
    plt.title('MSE: ' + str(round(test_mse, 4)), fontsize=12)
    patch_a = mpatches.Patch(color='C0', label='Predicted')
    patch_b = mpatches.Patch(color='C1', label='Observed')
    ax.legend(handles=[patch_a, patch_b])
    month = np.arange(0, len(ys), 1, dtype=int)
    ax.plot(month, np.array(preds), 'o', color='C0')
    ax.plot(month, np.array(ys), 'o', color='C1')
    plt.savefig(out_path + 'pred_a_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.xlabel('Observation')
    plt.ylabel('Prediction')
    plt.title('MSE: ' + str(round(test_mse, 4)), fontsize=12)
    ax.plot(np.array(ys), np.array(preds), 'o', color='C0')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.savefig(out_path + 'pred_b_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')
        
    print("Save the observed vs. predicted plots.")
    print("----------")
    print()
   
    plt.figure()
    plt.plot(all_epoch, all_loss)
    plt.plot(all_epoch, all_eval)
    blue_patch = mpatches.Patch(color='C0', label='Loss: ' + str(loss_function))
    orange_patch = mpatches.Patch(color='C1', label='Validation Metric: ' + 'MSE')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Performance')
    plt.savefig(out_path + 'perform_SSTAGraphDataset_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')

    print("Save the loss vs. evaluation metric plot.")
    print("--------------------")
    print()