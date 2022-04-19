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

# CNN configurations

CNN_structure = ['', '']
window_size = 3
train_split = 0.8
lead_time = 1
loss_function = 'MSE'
optimizer = 'SGD' # Adam
learning_rate = 0.1 #0.05
momentum = 0.9
weight_decay = 0.0001
batch_size = 64
num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
num_train_epoch = 10

data_path = 'data/'
models_path = 'out/'
out_path = 'out/'

"""
# Process the dataset.

soda = xr.open_dataset('data/soda_224_pt_l5.nc', decode_times=False)

soda_array = soda.to_array(dim='VARIABLE')
soda_smaller = np.array(soda_array[:,:,:,:,:,:])
soda_smaller = soda_smaller[2,:,0,:,::20,::20] # Drop the bnds dimension and the other two variables; take every 20th longitude and latitude.
soda_smaller = np.squeeze(soda_smaller, axis=0)

save(data_path + 'grids.npy', soda_smaller)

print("Save the grids in an NPY file")
print("--------------------")
print()
"""

# Load the grids.

grids = load(data_path + 'grids.npy')
y = load(data_path + 'y.npy')

y = y.squeeze(axis=1)

# Turn NAs into 0.
grids[np.isnan(grids)] = 0

#grids_window_size = []
#y_window_size = []
dataset = []
for i in range(len(y)-window_size-lead_time):
  #grids_window_size.append(grids[i:i+window_size])
  #y_window_size.append(y[i+window_size])
  dataset.append([torch.tensor(grids[i:i+window_size]), torch.tensor(y[i+window_size+lead_time-1])])
#grids_window_size = np.array(grids_window_size)
#y_window_size = np.array(y_window_size)
#dataset = np.array(dataset)

#print("Shapes of the grids and the outputs:")
#print(grids_window_size.shape)
#print(y_window_size.shape)
#print(dataset.shape)
print("--------------------")
print()

#print('Dataset:', dataset[0][0].shape)

num_examples = len(dataset)
num_train = int(num_examples * train_split)

train_sampler = SequentialSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_train, num_examples))

train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1, drop_last=False)

# Set up a CNN.

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.ReLU(),
        )
        self.out = nn.Linear(17*36, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=2)
        output = self.out(x)
        return output

model = CNN().double()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
loss_f = nn.MSELoss()

# Start time
start = time.time()

for epoch in range(num_train_epoch):
    print("Epoch " + str(epoch))
    print()
    
    losses = []
    for x, y in train_dataloader:
        train_x = Variable(x)
        train_y = Variable(y)
        output = model(train_x)
        loss = loss_f(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    print('Training loss:', sum(losses) / len(losses))

    losses = []
    for x, y in test_dataloader:
        test_x = Variable(x)
        test_y = Variable(y)
        pred = model(test_x)
        loss = loss_f(pred, test_y)
        losses.append(loss.cpu().detach().numpy())
    print('Validation MSE:', sum(losses) / len(losses))

    print("----------")
    print()

# End time
stop = time.time()

print(f'Complete training. Time spent: {stop - start} seconds.')
print("----------")
print()

# Test the model.

losses = []
for x, y in test_dataloader:
    test_x = Variable(x)
    test_y = Variable(y)
    pred = model(test_x)
    loss = loss_f(pred, test_y)
    losses.append(loss.cpu().detach().numpy())
print('Test MSE:', sum(losses) / len(losses))

print("----------")
print()