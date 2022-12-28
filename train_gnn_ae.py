from utils.losses import *
from utils.graphs import *
from utils.gnns import *

import numpy as np
from numpy import asarray, save, load
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
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

from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# GNN-AE configurations

for lead_time in [1, 3, 6]:
#for lead_time in [1]:

    # Train 5 models for the same configuration and average the predictions, to minimize the effect of randomness.
    all_preds = []
    
    #for model_num in range(5):
    for model_num in range(1):
    
        net_class = "GCN" # 'GAT'
        num_layer = 3
        num_hid_feat = 200
        num_out_feat = 100
        #num_out_feat = 1
        window_size = 5
        train_split = 0.8
        lead_time = lead_time
        #loss_function = 'BMSE' # 'MSE', 'MAE', 'Huber', 'WMSE', 'WMAE', 'WHuber', 'WFMSE', 'WFMAE', 'BMSE
        weight = 3
        loss_function = "CmMAE" + str(weight)
        negative_slope = 0.1
        activation = "lrelu" + str(negative_slope) # "relu", "tanh", "sigm"
        alpha = 0.9
        optimizer = "RMSP" + str(alpha)
        learning_rate = 0.0001 # 0.05, 0.02, 0.01, 0.001
        momentum = 0.9
        weight_decay = 0.01
        normalization = "bn"
        reg_factor = 0.0001
        regularization = "L1" + str(reg_factor) + "_nd"
        batch_size = 64
        num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
        num_train_epoch = 200
        
        data_path = 'data/'
        models_path = 'out/'
        out_path = 'out/'
        
        # Create a DGL graph dataset.
        
        loc_name = "SODAMiniGraph"
        
        dataset = SSTAGraphDataset_NodeLabels()
        
        print("Create a DGL dataset: SSTAGraphDataset_NodeLabels_windowsize_" + str(window_size) + '_leadtime_' + str(lead_time) + '_trainsplit_' + str(train_split))
        print("The last graph and its label:")
        print(dataset[-1])
        print("Node features:", dataset[-1].ndata["feat"])
        print("Node labels:", dataset[-1].ndata["label"])
        print("Edge features:", dataset[-1].edata["w"])
        print("--------------------")
        print()
        
        # Create data loaders.
        
        num_examples = len(dataset)
        num_train = int(num_examples * train_split)
        
        """
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
        """
        
        # Train the GCN.
        
        #model = GCN(window_size, num_hid_feat, num_out_feat)
        #model = GCN2(window_size, 1, 1, F.relu, 0.5)
        #model = GCN3(window_size, num_hid_feat, num_out_feat)
        model = GCN4(window_size, num_hid_feat)
        #optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
        
        print("Start training.")
        print()
        print("----------")
        print()
        
        # Start time
        start = time.time()
        
        g = dataset[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        
        for epoch in range(num_train_epoch):

            model.train()
            preds = model(g, features)
            #preds = torch.tensor(preds, dtype=torch.long)
            loss = F.nll_loss(preds, labels.long())

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        # End time
        stop = time.time()
        
"""    
        all_loss = []
        all_eval = []
        
        for epoch in range(num_train_epoch):
            print("Epoch " + str(epoch+1))
            print()
        
            losses = []
            
            # The threshold for defining outliers using the 90th percentile
            y_train = y_all[:int(len(y_all)*0.8)]
            y_train_sorted = np.sort(y_train)
            threshold = y_train_sorted[int(len(y_train_sorted)*0.9):][0]
            
            for batched_graph, y in train_dataloader:
                pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w'])
                loss = cm_weighted_mae(pred, y, threshold=threshold, weight=weight)
                
                # L1 regularization
                l1_crit = nn.L1Loss(reduction="sum")
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))
                factor = 0.0001
                loss += reg_factor * reg_loss
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.cpu().detach().numpy())

            print('Training loss:', sum(losses) / len(losses))
            print()
            all_loss.append(sum(losses) / len(losses))
        
            preds = []
            ys = []
            model.eval() # Tell the model to evaluate it instead of training, to avoid the BatchNorm1d error.
            for batched_graph, y in test_dataloader:
                pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['w']) ###
                preds.append(pred.cpu().detach().numpy().squeeze(axis=0))
                ys.append(y.cpu().detach().numpy().squeeze(axis=0))
            val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
            print('Test MSE:', val_mse)
            print()
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
                    "epoch": num_train_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": loss
                    }, models_path + "checkpoint_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + "_" + str(model_num) + ".tar")
        
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
        
        # Add to the predictions of all models.
        all_preds.append(preds)
            
        print("----------")
        print()
        
        print('Final test MSE:', test_mse)
        print("----------")
        print() 
        
        # Show the results.
    
        all_loss = np.array(all_loss)
        all_eval = np.array(all_eval)
        all_epoch = np.array(list(range(1, num_train_epoch+1)))
        
        all_perform_dict = {
            "training_time": str(stop-start),
            "all_loss": all_loss.tolist(),
            "all_eval": all_eval.tolist(),
            "all_epoch": all_epoch.tolist()}
        
        with open(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) +  "_" + str(model_num) + ".txt", "w") as file:
            file.write(json.dumps(all_perform_dict))
        
        print("Save the performance in a TXT file.")
        print("----------")
        print()
    
    for i in range(len(preds)):
        preds[i] = np.squeeze(preds[i])
       
    #print(preds)
    #print(ys)

    # Average the predictions by all models.
    sum_preds = np.add.reduce(all_preds)
    avg_preds = sum_preds / (model_num + 1)
    preds = avg_preds.tolist()
    
    # Increase the fontsize.
    plt.rcParams.update({"font.size": 20})
    
    # Calculate the threshold for 90th percentile and mark the outliers.
    y_train = y_all[:int(len(y_all)*0.8)]
    y_train_sorted = np.sort(y_train)
    threshold = y_train_sorted[int(len(y_train_sorted)*0.9):][0]
    threshold_weak = y_train_sorted[int(len(y_train_sorted)*0.8):][0] # The weak threshold for 80th percentile
    y_outliers = []
    pred_outliers = []
    for i in range(len(ys)):
        if ys[i] >= threshold:
            y_outliers.append(ys[i])
            pred_outliers.append(preds[i])
        else:
            y_outliers.append(None)
            pred_outliers.append(None)
    
    # Calculate the outlier MSE; remove the NAs.
    temp_y_outliers = [i for i in y_outliers if i is not None]
    temp_pred_outliers = [i for i in pred_outliers if i is not None]
    ol_test_mse = mean_squared_error(np.array(temp_y_outliers), np.array(temp_pred_outliers), squared=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel("Month")
    plt.ylabel("SST Residual")
    plt.title("MSE: " + str(round(test_mse, 4)) + ", MSE above 90th: " + str(round(ol_test_mse, 4)))
    patch_a = mpatches.Patch(color="pink", label="Obs")
    patch_b = mpatches.Patch(color="red", label="Obs above 90th")
    patch_c = mpatches.Patch(color="skyblue", label="Pred")
    patch_d = mpatches.Patch(color="blue", label="Pred for Obs above 90th")
    ax.legend(handles=[patch_a, patch_b, patch_c, patch_d])
    month = np.arange(0, len(ys), 1, dtype=int)
    plt.plot(month, np.array(ys, dtype=object), linestyle="-", color="pink")
    ax.plot(month, np.array(ys, dtype=object), "o", color="pink")
    ax.plot(month, np.array(y_outliers, dtype=object), "o", color="red")
    plt.plot(month, np.array(preds, dtype=object), linestyle="-", color="skyblue")
    ax.plot(month, np.array(preds, dtype=object), "o", color="skyblue")
    ax.plot(month, np.array(pred_outliers, dtype=object), "o", color="blue")
    plt.savefig(out_path + "pred_a_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    lim = max(np.abs(np.array(preds)).max(), np.abs(np.array(ys)).max())
    ax.set_xlim([-lim-0.1, lim+0.1])
    ax.set_ylim([-lim-0.1, lim+0.1])
    plt.xlabel("Obs SST Residual")
    plt.ylabel("Pred SST Residual")
    plt.title("MSE: " + str(round(test_mse, 4)) + ", MSE above 90th: " + str(round(ol_test_mse, 4)))
    ax.plot(np.array(ys, dtype=object), np.array(preds, dtype=object), "o", color="black")
    transform = ax.transAxes
    line_a = mlines.Line2D([0, 1], [0, 1], color="red")
    line_a.set_transform(transform)
    ax.add_line(line_a)
    patch_a = mpatches.Patch(color="pink", label="Obs above 90th")
    ax.legend(handles=[patch_a])
    ax.axvspan(threshold, max(ys)+0.1, color="pink")
    plt.savefig(out_path + "pred_b_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
        
    print("Save the observed vs. predicted plots.")
    print("----------")
    print()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(all_epoch, all_loss)
    plt.plot(all_epoch, all_eval)
    blue_patch = mpatches.Patch(color="C0", label="Loss: " + str(loss_function))
    orange_patch = mpatches.Patch(color="C1", label="Test Metric: " + "MSE")
    ax.legend(handles=[blue_patch, orange_patch])
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Performance")
    plt.savefig(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
    print("Save the loss vs. evaluation metric plot.")
    print("--------------------")
    print()

    # Regression results
    
    # "Recall-R"
    recall_r = ol_test_mse
    # "Precision-R"
    y_outliers = []
    pred_outliers = []
    for i in range(len(preds)):
        if preds[i] >= threshold:
            y_outliers.append(ys[i])
            pred_outliers.append(preds[i])
        else:
            y_outliers.append(None)
            pred_outliers.append(None)
    temp_y_outliers = [i for i in y_outliers if i is not None]
    temp_pred_outliers = [i for i in pred_outliers if i is not None]
    # If the lists are empty, the precision-R and F1-R are NAs.
    if len(temp_pred_outliers) == 0:        
        regress_dict = {
          "MSE": str(round(test_mse, 4)),
          "MSE for Predicted Anomalies (Precision-R)": "NA",
          "MSE for Observed Anomalies (Recall-R)": str(round(recall_r, 4)),
          "F1-R": "NA"
          }
    else:
        precision_r = mean_squared_error(np.array(temp_y_outliers), np.array(temp_pred_outliers), squared=True)    
        # "F1-R"
        f1_r = 2 / (recall_r ** (-1) + precision_r ** (-1))
        regress_dict = {
          "MSE": str(round(test_mse, 4)),
          "MSE for Predicted Anomalies (Precision-R)": str(round(precision_r, 4)),
          "MSE for Observed Anomalies (Recall-R)": str(round(recall_r, 4)),
          "F1-R": str(round(f1_r, 4))
          }
    with open(out_path + "regression_SSTASODA"  + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as f:
        f.write(json.dumps(regress_dict))

    # Classification results
    
    ys_masked = ["MHW Weak Indicator (>80th)" if ys[i] >= threshold_weak else "None" for i in range(len(ys))]
    ys_masked = ["MHW Strong Indicator (>90th)" if ys[i] >= threshold else ys_masked[i] for i in range(len(ys_masked))]
    preds_masked = ["MHW Weak Indicator (>80th)" if preds[i] >= threshold_weak else "None" for i in range(len(preds))]
    preds_masked = ["MHW Strong Indicator (>90th)" if preds[i] >= threshold else preds_masked[i] for i in range(len(preds_masked))]
    
    classification_results = classification_report(ys_masked, preds_masked, digits=4)

    with open(out_path + "classification_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(normalization) + "_" + str(regularization) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as f:
        print(classification_results, file=f)
    
    print("Save the classification results in a TXT file.")
    print("----------")
    print()
"""