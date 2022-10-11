from losses import *

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

from sklearn.metrics import mean_squared_error

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

checkpoint_path = "checkpoint_SSTASaltSODAHalf_CNN_30_500_3_1_1677_0.8_MSE_RMSP_tanh_0.001_0.9_0_540_400.tar"
performance_path = "perform_SSTASaltSODAHalf_CNN_30_500_3_1_1677_0.8_MSE_RMSP_tanh_0.001_0.9_0_540_400.txt"

# CNN configurations

net_class = "CNN" #
num_layer = 3 #
num_hid_feat = 30 #
num_out_feat = 500 #
window_size = 5
train_split = 0.8
lead_time = 1
noise_var = 0.01
loss_function = "BMSE" + str(noise_var) # "MSE", "MAE", "Huber", "WMSE", "WMAE", "WHuber", "WFMSE", "WFMAE", "BMSE
negative_slope = 0.1
activation = "lrelu" + str(negative_slope) # "relu", "tanh", "sigm"
alpha = 0.9
optimizer = "RMSP" + str(alpha) # SGD, Adam
learning_rate = 0.05 # 0.05, 0.02, 0.01
momentum = 0
weight_decay = 0
dropout = "nd"
batch_size = 512 # >= 120 crashed for original size, >= 550 crashed for half size, >= 480 crashed for half size and two variables
num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
num_train_epoch = 400

data_path = "data/"
models_path = "out/"
out_path = "out/"

# Load the grids.

grids = load(data_path + "grids_half.npy")
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

#print("Dataset:", dataset[0][0].shape)

num_examples = len(dataset)
num_train = int(num_examples * train_split)

train_sampler = SequentialSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_train, num_examples))

train_dataloader = DataLoader(dataset, sampler=torch.arange(num_train), batch_size=1, drop_last=False)
test_dataloader = DataLoader(dataset, sampler=torch.arange(num_train, num_examples), batch_size=1, drop_last=False)

#print(next(iter(train_dataloader)))
#print(next(iter(test_dataloader)))

# Set up a CNN.

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(window_size, num_hid_feat, 8) # window_size * 2: window size of three, two variables
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
        self.fc1 = nn.Linear(87150, num_out_feat)
        self.fc2 = nn.Linear(num_out_feat, 1 )
        self.double()

    def forward(self, x):
        h = self.conv1(x)
        act_f = nn.Tanh()
        #print("Conv 1 passed.")
        h = act_f(h)
        h = self.pool1(h)
        #print("Pool 1 passed.")
        h = self.conv2(h)
        h = act_f(h)
        #print("Conv 2 passed.")
        h = self.pool2(h)
        #print("Pool 2 passed.")
        h = self.conv3(h)
        #print("Conv 3 passed.")
        h = h.view(x.size(0), -1)
        h = self.fc1(h)
        #print("FCN 1 passed.")
        h = act_f(h)
        output = self.fc2(h)
        #print("FCN 2 passed.")
        #print("Output"s shape: ", output.shape)
        return output

model = CNN()
optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

checkpoint = torch.load(models_path + checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
optim.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

# Test the model.

preds = []
ys = []
for x, y in test_dataloader:
#for x, y in train_dataloader:
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

# Read the performance dictionary from the TXT file.

with open(models_path + performance_path) as f:
    data = f.read()
all_performance = json.loads(data)

all_loss = all_performance["all_loss"]
all_eval = all_performance["all_eval"]
all_epoch = all_performance["all_epoch"]

# Increase the fontsize.
plt.rcParams.update({"font.size": 20})

# Calculate the threshold for 90th percentile and mark the outliers.
y_train = y_all[:int(len(y_all)*0.8)]
y_train_sorted = np.sort(y_train)
threshold = y_train_sorted[int(len(y_train_sorted)*0.9):][0]
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
plt.savefig(out_path + "pred_a_SSTASODAHalfBoP_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")

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
plt.savefig(out_path + "pred_b_SSTASODAHalfBoP_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")
    
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
plt.savefig(out_path + "perform_SSTASODAHalfBoP_" + str(net_class) + "_" + str(num_hid_feat) + "_" + str(num_out_feat) + "_" + str(window_size) + "_" + str(lead_time) + "_" + str(num_sample) + "_" + str(train_split) + "_" + str(loss_function) + "_" + str(optimizer) + "_" + str(activation) + "_" + str(learning_rate) + "_" + str(momentum) + "_" + str(weight_decay) + "_" + str(dropout) + "_" + str(batch_size) + "_" + str(num_train_epoch) + ".png")

print("Save the loss vs. evaluation metric plot.")
print("--------------------")
print()