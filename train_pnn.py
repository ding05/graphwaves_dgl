from utils.losses import *

import numpy as np
from numpy import asarray, save, load
import math

import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

import json
import matplotlib.pyplot as plt

# PNN configurations

# Train multiple models for the same configuration and average the predictions, to minimize the effect of randomness.
all_preds = []

#for model_num in range(10):
for model_num in range(1):

    net_class = 'PNN'
    num_layer = 2
    num_hid_feat = 50
    num_out_feat = 1
    window_size = 6
    train_split = 0.8
    lead_time = 1
    weight = 3
    #loss_function = 'CmMAE' + str(weight)
    #loss_function = 'MSE'
    negative_slope = 0.1
    activation = 'sigm'
    alpha = 0.9
    optimizer = 'RMSP' + str(alpha)
    learning_rate = 0.002
    momentum = 0.9
    #momentum = 0
    weight_decay = 0.01
    normalization = 'L1_nbn'
    normalization = 'bn'
    reg_factor = 0.0001
    regularization = 'L1' + str(reg_factor) + '_nd'
    batch_size = 'full'
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
    num_train_epoch = 200
    
    data_path = 'data/'
    models_path = 'out/'
    out_path = 'out/'

    # Load the input.

    loc_name = 'BoPwNZ'
    
    x0 = load(data_path + 'y_bd.npy').squeeze(axis=1)
    x1 = load(data_path + 'y_bop.npy').squeeze(axis=1)
    x2 = load(data_path + 'y_ci.npy').squeeze(axis=1)
    x3 = load(data_path + 'y_cr.npy').squeeze(axis=1)
    x4 = load(data_path + 'y_cs.npy').squeeze(axis=1)
    x5 = load(data_path + 'y_f.npy').squeeze(axis=1)
    x6 = load(data_path + 'y_mg.npy').squeeze(axis=1)
    x7 = load(data_path + 'y_od.npy').squeeze(axis=1)
    x8 = load(data_path + 'y_r.npy').squeeze(axis=1)
    x9 = load(data_path + 'y_si.npy').squeeze(axis=1)
    x10 = load(data_path + 'y_t.npy').squeeze(axis=1)
    x11 = load(data_path + 'y_w.npy').squeeze(axis=1)
    
    y = load(data_path + 'y_bop.npy').squeeze(axis=1)
    
    x_all, y_all = [], []
        
    for i in range(len(y)-window_size-lead_time):
        x_all.append(np.concatenate((x0[i:i+window_size], x1[i:i+window_size], x2[i:i+window_size], x3[i:i+window_size], x4[i:i+window_size], x5[i:i+window_size], x6[i:i+window_size], x7[i:i+window_size], x8[i:i+window_size], x9[i:i+window_size], x10[i:i+window_size], x11[i:i+window_size])))
        y_all.append(y[i+window_size+lead_time-1])
    
    x_all, y_all = np.array(x_all), np.array(y_all)
    
    # Normalize the data to [-1, 1].
    data_all = np.concatenate((x_all, y_all.reshape(-1, 1)), axis=1)
    data_all_normalized = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) * 2 - 1
    print('Normalized feature and label grid:', data_all_normalized)
    print('Shape:', data_all_normalized.shape)
    print('----------')
    print()
    
    x_all, y_all = data_all_normalized[:,:x_all.shape[1]], np.squeeze(data_all_normalized[:,-1:])
    
    #print(x_all)
    print('Input data shape:', x_all.shape)
    #print(y_all)
    print('Output data shape:', y_all.shape)
    
    num_train = int(len(x_all) * train_split)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    
    print('Input training data shape:', x_train.shape)
    print('Output training data shape:', y_train.shape)
    print('Input test data shape:', x_test.shape)
    print('Output test data shape:', y_test.shape)
    
    print('--------------------')
    print()
    
    # Set up a PNN.
    
    class PNN(nn.Module):
        def __init__(self, num_input_feat, num_out_feat, num_hid_feat):
            super(PNN, self).__init__()
            self.hidden_layer = nn.Linear(num_input_feat, num_hid_feat)
            self.output_layer = nn.Linear(num_hid_feat, num_out_feat)
            self.rbf_layer = nn.Linear(num_out_feat, num_hid_feat, bias=False)
    
        def forward(self, input_seq):
            hidden_output = torch.sigmoid(self.hidden_layer(input_seq))
            output_mean = self.output_layer(hidden_output)
            output_activation = self.rbf_layer(output_mean)
            print(output_activation.shape)
            print(output_mean.shape)
            output_var = torch.sum((output_activation - output_mean)**2, dim=1, keepdim=True)
            output_seq = output_mean + output_var
            return output_seq
    
    model = PNN(num_input_feat=x_all.shape[1], num_out_feat=1, num_hid_feat=num_hid_feat)
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
    criterion = nn.MSELoss()
    
    # Train the model.
    
    # Start time
    start = time.time()
    
    for epoch in range(num_train_epoch):
        print('Epoch ' + str(epoch))
        print()
        
        running_loss = 0.0
        
        for i in range(len(x_train)):
            optim.zero_grad()
            input_seq = torch.from_numpy(x_train[i]).float()
            output_seq = torch.from_numpy(np.array([y_train[i]])).float()
            predicted_seq = model(input_seq)
            loss = criterion(predicted_seq, output_seq)
            loss.backward()
            optim.step()
    
            # accumulate loss
            running_loss += loss.item()
            
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / x_train.shape[0]))