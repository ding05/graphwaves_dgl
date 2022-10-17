from utils.losses import *

import numpy as np
from numpy import asarray, save, load
import math

import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable

import time

from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# FCN configurations

for lead_time in [1, 2, 3, 6]:

    # Train 10 models for the same configuration and average the predictions, to minimize the effect of randomness.
    all_preds = []
    
    for model_num in range(10):
    #for model_num in range(1):
    
        net_class = "FCN" #
        num_layer = 2 #
        num_hid_feat = 50 # 500 for grids
        num_out_feat = 1 #
        window_size = 5
        train_split = 0.8
        lead_time = lead_time
        noise_var = 0.01
        #loss_function = "BMSE" + str(noise_var) # "MSE", "MAE", "Huber", "WMSE", "WMAE", "WHuber", "WFMSE", "WFMAE", "BMSE"
        weight = 50
        loss_function = "CmMAE" + str(weight)
        #loss_function = "MAE"
        negative_slope = 0.1
        activation = "lrelu" + str(negative_slope) # "relu", "tanh", "sigm"
        alpha = 0.9
        optimizer = "RMSP" + str(alpha) # SGD, Adam
        learning_rate = 0.002 # 0.000001 for grids
        #momentum = 0.9
        momentum = 0
        weight_decay = 0.01
        batch_norm = "bn"
        dropout = "nd"
        batch_size = 512 # >= 120 crashed for original size, >= 550 crashed for half size, >= 480 crashed for half size and two variables
        num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
        num_train_epoch = 200
        
        data_path = "data/"
        models_path = "out/"
        out_path = "out/"
        
        # Load the input.
        
        loc_name = "BoPwOceans"
        #loc_name = "BoPwSODAMini"
        
        x = load(data_path + "y.npy").squeeze(axis=1)
        x1 = load(data_path + "y_eastaus.npy").squeeze(axis=1)
        x2 = load(data_path + "y_equapacific.npy").squeeze(axis=1)
        x3 = load(data_path + "y_nepacific.npy").squeeze(axis=1)
        x4 = load(data_path + "y_nwpacific.npy").squeeze(axis=1)
        x5 = load(data_path + "y_southpacific.npy").squeeze(axis=1)
        x6 = load(data_path + "y_indian.npy").squeeze(axis=1)
        x7 = load(data_path + "y_indian.npy").squeeze(axis=1)
        x8 = load(data_path + "y_southatlantic.npy").squeeze(axis=1)
        
        grids = load(data_path + "grids_mini.npy")
        grids[np.isnan(grids)] = 0 # Convert NAs into zeroes.
        flattened_grids = grids.reshape(grids.shape[0], grids.shape[1] * grids.shape[2]) # Flatten the grids to fit the FCN.
        
        y = load(data_path + "y.npy").squeeze(axis=1)
        
        """
        x = x.squeeze(axis=1)
        x1 = x1.squeeze(axis=1)
        y = y.squeeze(axis=1)
        """
        y_all = y
        #num_var = 1
        #num_var = 2
        num_var = 9
        #num_var = grids.shape[1] * grids.shape[2]
        
        dataset = []
        
        for i in range(len(y)-window_size-lead_time):
            #dataset.append([torch.tensor(x[i:i+window_size]), torch.tensor(y[i+window_size+lead_time-1])])
            #dataset.append([torch.tensor(np.concatenate((x[i:i+window_size], x1[i:i+window_size]))), torch.tensor(y[i+window_size+lead_time-1])])
            dataset.append([torch.tensor(np.concatenate((x[i:i+window_size], x1[i:i+window_size], x2[i:i+window_size], x3[i:i+window_size], x4[i:i+window_size], x5[i:i+window_size], x6[i:i+window_size], x7[i:i+window_size], x8[i:i+window_size]))), torch.tensor(y[i+window_size+lead_time-1])])
            #dataset.append([torch.tensor(np.concatenate((flattened_grids[i:i+window_size]))), torch.tensor(y[i+window_size+lead_time-1])])
            
        print("--------------------")
        print()
        
        num_examples = len(dataset)
        num_train = int(num_examples * train_split)
        
        train_sampler = SequentialSampler(torch.arange(num_train))
        test_sampler = SequentialSampler(torch.arange(num_train, num_examples))
        
        train_dataloader = DataLoader(dataset, sampler=torch.arange(num_train), batch_size=batch_size, drop_last=False)
        test_dataloader = DataLoader(dataset, sampler=torch.arange(num_train, num_examples), batch_size=1, drop_last=False)
        
        # Set up a FCN.
        
        class FCN(nn.Module):
            def __init__(self):
                super(FCN, self).__init__()
                self.fc_bn = nn.BatchNorm1d(window_size * num_var)    
                self.fc1 = nn.Linear(window_size * num_var, num_hid_feat)
                self.fc2 = nn.Linear(num_hid_feat, 1)
                #self.dropout = nn.Dropout(0.25)
                self.double()
            
            def forward(self, x):
                #print(x)
                #print(x.shape)
                x = self.fc_bn(x)
                #print(x)
                #print(x.shape)
                x = self.fc1(x)       
                x = F.leaky_relu(x, negative_slope)
                #x = torch.sigmoid(x)
                #x = self.dropout(x)
                x = self.fc2(x)        
                x = F.leaky_relu(x, negative_slope)
                #x = torch.sigmoid(x)
                return x
        
        model = FCN()
        optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
        
        # Train the model.
        
        # Start time
        start = time.time()
        
        all_loss = []
        all_eval = []
        
        for epoch in range(num_train_epoch):
            print("Epoch " + str(epoch))
            print()
            
            losses = []
            
            # The threshold for defining outliers using the 90th percentile
            y_train = y_all[:int(len(y_all)*0.8)]
            y_train_sorted = np.sort(y_train)
            threshold = y_train_sorted[int(len(y_train_sorted)*0.9):][0]
            
            # Control the weight parameter in the customized MAE loss.
            #if epoch < num_train_epoch * 0.95:
            if epoch < num_train_epoch * 0.9:
                cur_weight = weight-((weight-3)/num_train_epoch)*epoch
                #cur_weight = 1
            else:
                cur_weight = 3
            print("cur_weight:", cur_weight)
            
            for x, y in train_dataloader:
                pred = torch.squeeze(model(x))
                #loss_func = nn.MSELoss()
                #loss_func = nn.L1Loss()
                #loss = loss_func(pred, y)
                loss = cm_weighted_mae(pred, y, threshold=threshold, weight=cur_weight)
                #loss = balanced_mse(pred, y, noise_var)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.cpu().detach().numpy())
            print()
            print("Training loss:", sum(losses) / len(losses))
            print()
            all_loss.append(sum(losses) / len(losses))
            
            preds = []
            ys = []
            model.eval() # Tell the model to evaluate it instead of training, to avoid the BatchNorm1d error.
            for x, y in test_dataloader:
                pred = torch.squeeze(model(x))
                preds.append(pred.cpu().detach().numpy())
                ys.append(y.cpu().detach().numpy())
            val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
            print("Test MSE:", val_mse)
            print()
            all_eval.append(val_mse)
        
            print("----------")
            print()
        
        # End time
        stop = time.time()
        
        print(f"Complete training. Time spent: {stop - start} seconds.")
        print("----------")
        print()
        
        torch.save({
                    "epoch": num_train_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": loss
                    }, models_path + "checkpoint_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + "_" + str(model_num) + ".tar")
        
        print("Save the checkpoint in a TAR file.")
        print("----------")
        print()
        
        # Test the model.
        
        preds = []
        ys = []
        for x, y in test_dataloader:
            pred = torch.squeeze(model(x))
            preds.append(pred.cpu().detach().numpy())
            ys.append(y.cpu().detach().numpy())
        test_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
        test_rmse = mean_squared_error(np.array(ys), np.array(preds), squared=False)
        
        # Add to the predictions of all models.
        all_preds.append(preds)
            
        print("----------")
        print()
        
        print("Final test MSE:", test_mse)
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
        
        with open(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) +  "_" + str(model_num) + ".txt", "w") as file:
            file.write(json.dumps(all_perform_dict))
        
        print("Save the performance in a TXT file.")
        print("----------")
        print()
    
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
    plt.savefig(out_path + "pred_a_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
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
    plt.savefig(out_path + "pred_b_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
        
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
    plt.savefig(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
    print("Save the loss vs. evaluation metric plot.")
    print("--------------------")
    print()

    # Confusion matrix
    
    """
    ys_masked = [1 if i >= threshold else 0 for i in ys]
    preds_masked = [1 if i >= threshold else 0 for i in preds]
    tn, fp, fn, tp = confusion_matrix(ys_masked, preds_masked).ravel()
    
    class_dict = {
      "Introduction": "Positive: data points >= the 90th percentile (anomalies, associated with marine heatwaves), Negative: others (norms)",
      "Number of data points": str(len(ys)),
      "True positive": str(tp),
      "True negative": str(tn),
      "True classifications": str(tp+tn),
      "False positive": str(fp),
      "False negative": str(fn),
      "False classifications": str(fp+fn),
      "True positive rate": str(round(tp/len(ys),4)),
      "True negative rate": str(round(tn/len(ys),4)),
      "False positive rate": str(round(fp/len(ys),4)),
      "False negative rate": str(round(fn/len(ys),4)),      
      "Accuracy": str(round((tp+tn)/len(ys),4)),
      "Precision": str(round(tp/(tp+fp),4)),
      "Recall": str(round(tp/(tp+fn),4))
      }
    with open(out_path + "classification_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as file:
        file.write(json.dumps(class_dict))
    """
    
    ys_masked = ["MHW Weak Indicator (>80th)" if ys[i] >= threshold_weak else "None" for i in range(len(ys))]
    ys_masked = ["MHW Strong Indicator (>90th)" if ys[i] >= threshold else ys_masked[i] for i in range(len(ys_masked))]
    preds_masked = ["MHW Weak Indicator (>80th)" if preds[i] >= threshold_weak else "None" for i in range(len(preds))]
    preds_masked = ["MHW Strong Indicator (>90th)" if preds[i] >= threshold else preds_masked[i] for i in range(len(preds_masked))]
    
    classification_results = classification_report(ys_masked, preds_masked, digits=4)

    with open(out_path + "classification_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_layer) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(batch_norm) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as f:
        print(classification_results, file=f)
    
    print("Save the classification results in a TXT file.")
    print("----------")
    print()