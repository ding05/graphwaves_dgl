from loss import *

import numpy as np
from numpy import asarray, save, load

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

# for lead_time in [1]:

# CNN configurations

net_class = 'CNN' #
num_layer = 3 #
num_hid_feat = 30 #
num_out_feat = 50 #
window_size = 5
train_split = 0.8
lead_time = 1
loss_function = 'MSE' # 'MSE', 'MAE', 'Huber', 'WMSE', 'WMAE', 'WHuber', 'WFMSE', 'WFMAE', 'BMSE
activation = 'tanh' # 'relu', 'tanh' 
optimizer = 'RMSP' # SGD, Adam
learning_rate = 0.005 # 0.05, 0.02, 0.01
momentum = 0.9
weight_decay = 0.0001
batch_size = 540 # >= 120 crashed for original size, >= 550 crashed for half size
num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
num_train_epoch = 200

data_path = 'data/'
models_path = 'out/'
out_path = 'out/'

"""
# If running this script for the first time, process the dataset.

soda = xr.open_dataset('data/soda_224_pt_l5.nc', decode_times=False)

soda_array = soda.to_array(dim='VARIABLE')
soda_smaller = np.array(soda_array[:,:,:,:,:,:])
soda_smaller = soda_smaller[2,:,0,:,::4,::4] # Drop the bnds dimension and the other two variables; take every 4th longitude and latitude.
soda_smaller = np.squeeze(soda_smaller, axis=0)

save(data_path + 'grids_quarter.npy', soda_smaller)

print("Save the grids in an NPY file")
print("--------------------")
print()
"""

# Load the grids.

grids = load(data_path + 'grids_half.npy')
y = load(data_path + 'y.npy')

y = y.squeeze(axis=1)

# Turn NAs into 0.
grids[np.isnan(grids)] = 0

dataset = []
for i in range(len(y)-window_size-lead_time):
  dataset.append([torch.tensor(grids[i:i+window_size]), torch.tensor(y[i+window_size+lead_time-1])])

print("--------------------")
print()

#print('Dataset:', dataset[0][0].shape)

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
        self.conv1 = nn.Conv2d(window_size, num_hid_feat, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(num_hid_feat, num_hid_feat, 4)
        self.fc1 = nn.Linear(87150, num_out_feat) # 394440 for full, 15960 for quarter
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
        #h = h.view(h.size(0), -1)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        #print("FCN 1 passed.")
        output = self.fc2(h)
        #print("FCN 2 passed.")
        #print("Output's shape: ", output.shape)
        return output

model = CNN()
optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

# Start time
start = time.time()

all_loss = []
all_eval = []

for epoch in range(num_train_epoch):
    print("Epoch " + str(epoch))
    print()
    
    losses = []
    for x, y in train_dataloader:
        pred = torch.squeeze(model(x))
        
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
        
        print('pred: ', pred)
        print('y: ', y)
        print()
        loss_func = nn.MSELoss()
        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.cpu().detach().numpy())
    print("----------")
    print()
    print('Training loss:', sum(losses) / len(losses))
    print()
    all_loss.append(sum(losses) / len(losses))

    preds = []
    ys = []
    for x, y in test_dataloader:
        pred = torch.squeeze(model(x))
        preds.append(pred.cpu().detach().numpy())
        ys.append(y.cpu().detach().numpy())
    val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
    print('Validation MSE:', val_mse)
    print()
    all_eval.append(val_mse)

    print("----------")
    print()

# End time
stop = time.time()

print(f'Complete training. Time spent: {stop - start} seconds.')
print("----------")
print()

torch.save({
            'epoch': num_train_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss
            }, models_path + 'checkpoint_SSTASODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.tar')

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

with open(out_path + 'perform_SSTASODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.txt', "w") as file:
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
plt.savefig(out_path + 'pred_a_SSTASODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')

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
plt.savefig(out_path + 'pred_b_SSTASODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')
    
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
plt.savefig(out_path + 'perform_SSTASODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')

print("Save the loss vs. evaluation metric plot.")
print("--------------------")
print()