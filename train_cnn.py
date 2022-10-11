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

# CNN configurations

for lead_time in [1]:

    net_class = "CNN" #
    num_layer = 3 #
    num_hid_feat = 30 #
    num_out_feat = 500 #
    window_size = 5
    train_split = 0.8
    lead_time = lead_time
    noise_var = 0.01
    loss_function = "BMSE" + str(noise_var) # "MSE", "MAE", "Huber", "WMSE", "WMAE", "WHuber", "WFMSE", "WFMAE", "BMSE"
    weight = 75
    #loss_function = "CmMAE" + str(weight)
    loss_function = "MAE"
    negative_slope = 0.1
    #activation = "lrelu" + str(negative_slope) # "relu", "tanh", "sigm"
    activation = "tanh"
    alpha = 0.9
    optimizer = "RMSP" + str(alpha) # SGD, Adam
    learning_rate = 0.001 # 0.05, 0.02, 0.01
    momentum = 0.9
    weight_decay = 0.01
    dropout = "nd"
    batch_size = 512 # >= 120 crashed for original size, >= 550 crashed for half size, >= 480 crashed for half size and two variables
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
    num_train_epoch = 100
    
    data_path = "data/"
    models_path = "out/"
    out_path = "out/"
    
    """
    # If running this script for the first time, process the dataset.
    
    soda = xr.open_dataset("data/soda_224_pt_l5.nc", decode_times=False)
    
    soda_array = soda.to_array(dim="VARIABLE")
    soda_smaller = np.array(soda_array[:,:,:,:,:,:])
    soda_smaller = soda_smaller[2,:,0,:,::4,::4] # Drop the bnds dimension and the other two variables; take every 4th longitude and latitude.
    soda_smaller = np.squeeze(soda_smaller, axis=0)
    
    save(data_path + "grids_quarter.npy", soda_smaller)
    
    print("Save the grids in an NPY file")
    print("--------------------")
    print()
    """
    
    """
    # If running this script for the first time, process the dataset to get a smaller grid around Bay of Plenty.
    
    soda = xr.open_dataset("data/soda_224_pt_l5.nc", decode_times=False)
    
    soda["LONN359_360"] = soda.LONN359_360 + 180
    
    soda_bop = soda.where(soda.LAT < 0, drop=True)
    soda_bop = soda_bop.where(soda.LAT > -70, drop=True)
    soda_bop = soda_bop.where(soda.LONN359_360 > 107, drop=True)
    soda_bop = soda_bop.where(soda.LONN359_360 < 247, drop=True)
    
    soda_array_bop = soda_bop.to_array(dim="VARIABLE")
    soda_smaller_bop = np.array(soda_array_bop[:,:,:,:,:,:])
    soda_smaller_bop = soda_smaller_bop[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
    soda_smaller_bop = np.squeeze(soda_smaller_bop, axis=0)
    soda_smaller_bop = np.transpose(soda_smaller_bop, (2, 0, 1))
    
    save(data_path + "grids_bop.npy", soda_smaller_bop)
    
    print("Save the grids in an NPY file")
    print("--------------------")
    print()
    """
    
    # Load the grids.
    
    loc_name = "BoPQuarter"
    
    grids = load(data_path + "grids_quarter.npy")
    #grids_salt = load(data_path + "grids_salt_half.npy")
    y = load(data_path + "y.npy")
    
    y = y.squeeze(axis=1)
    y_all = y
    
    # Turn NAs into 0.
    grids[np.isnan(grids)] = 0
    #grids_salt[np.isnan(grids_salt)] = 0
    
    dataset = []
    
    # For one variable: SSTA
    
    for i in range(len(y)-window_size-lead_time):
      dataset.append([torch.tensor(grids[i:i+window_size]), torch.tensor(y[i+window_size+lead_time-1])])
    
    """
    # For two variables: SSTA and salinity
    
    for i in range(len(y)-window_size-lead_time):
      dataset.append([torch.tensor(np.concatenate((grids[i:i+window_size], grids_salt[i:i+window_size]))), torch.tensor(y[i+window_size+lead_time-1])])
    """
    
    #print("Dataset:", dataset[0][0].shape)
    
    print("--------------------")
    print()
    
    num_examples = len(dataset)
    num_train = int(num_examples * train_split)
    
    train_sampler = SequentialSampler(torch.arange(num_train))
    test_sampler = SequentialSampler(torch.arange(num_train, num_examples))
    
    train_dataloader = DataLoader(dataset, sampler=torch.arange(num_train), batch_size=batch_size, drop_last=False)
    test_dataloader = DataLoader(dataset, sampler=torch.arange(num_train, num_examples), batch_size=1, drop_last=False)
    
    #print(next(iter(train_dataloader)))
    #print(next(iter(test_dataloader)))
    
    # Set up a CNN.
    
    class CNN(nn.Module):
    
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(window_size, num_hid_feat, 8) # window_size: window size of three, two variables
            self.conv1_bn = nn.BatchNorm2d(num_hid_feat)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
            self.conv2_bn = nn.BatchNorm2d(num_hid_feat)
            self.pool2 = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
            self.conv3_bn = nn.BatchNorm2d(num_hid_feat)
            self.fc1 = nn.Linear(15960, num_out_feat) # 394440 for full, 87150 for half, 15960 for quarter
            #self.fc1_bn = nn.BatchNorm1d(num_out_feat) # # an error, not fixed
            self.fc2 = nn.Linear(num_out_feat, 1 )
            self.double()
    
        def forward(self, x):
            h = self.conv1(x)
            #h = F.leaky_relu(h, negative_slope)
            h = self.conv1_bn(h)
            h = torch.tanh(h)
            h = self.pool1(h)
            h = self.conv2(h)
            h = self.conv2_bn(h)
            #h = F.leaky_relu(h, negative_slope)
            h = torch.tanh(h)
            h = self.pool2(h)
            h = self.conv3(h)
            h = self.conv3_bn(h)
            h = torch.flatten(h, 1)
            h = self.fc1(h)
            #h = self.fc1_bn(h) # an error, not fi
            #h = F.leaky_relu(h, negative_slope)
            h = torch.tanh(h)
            output = self.fc2(h)
            return output
    
    model = CNN()
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
    
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
        """
        if epoch < num_train_epoch * 0.95:
            cur_weight = weight-((weight-5)/num_train_epoch)*epoch
            #cur_weight = 1
        else:
            cur_weight = 5
        print("cur_weight:", cur_weight)
        """
        
        for x, y in train_dataloader:
            pred = torch.squeeze(model(x))
            #loss_func = nn.MSELoss()
            loss_func = nn.L1Loss()
            loss = loss_func(pred, y)
            #loss = cm_weighted_mae(pred, y, threshold=threshold, weight=cur_weight)
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
                }, models_path + "checkpoint_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".tar")
    
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
    
    with open(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as file:
        file.write(json.dumps(all_perform_dict))
    
    print("Save the performance in a TXT file.")
    print("----------")
    print()
    
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
    plt.savefig(out_path + "pred_a_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
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
    plt.savefig(out_path + "pred_b_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
        
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
    plt.savefig(out_path + "perform_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
    print("Save the loss vs. evaluation metric plot.")
    print("--------------------")
    print()

    # Confusion matrix
    
    ys_masked = ["MHW Weak Indicator (>80th)" if ys[i] >= threshold_weak else "None" for i in range(len(ys))]
    ys_masked = ["MHW Strong Indicator (>90th)" if ys[i] >= threshold else ys_masked[i] for i in range(len(ys_masked))]
    preds_masked = ["MHW Weak Indicator (>80th)" if preds[i] >= threshold_weak else "None" for i in range(len(preds))]
    preds_masked = ["MHW Strong Indicator (>90th)" if preds[i] >= threshold else preds_masked[i] for i in range(len(preds_masked))]
    
    classification_results = classification_report(ys_masked, preds_masked, digits=4)

    with open(out_path + "classification_SSTASODA" + loc_name + "_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".txt", "w") as f:
        print(classification_results, file=f)
    
    print("Save the classification results in a TXT file.")
    print("----------")
    print()