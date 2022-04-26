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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

for lead_time in [1, 2, 3, 6, 12, 23]:

    # CNN configurations
    
    CNN_structure = ['', '']
    window_size = 3
    train_split = 0.8
    #lead_time = 1
    loss_function = 'MSE'
    optimizer = 'Adam' # SGD
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 16
    num_sample = 1680-window_size-lead_time+1 # max: node_features.shape[1]-window_size-lead_time+1
    num_train_epoch = 10
    
    data_path = 'data/'
    models_path = 'out/'
    out_path = 'out/'
    
    """
    # If running this script for the first time, process the dataset.
    
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
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Tanh(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.Tanh(),
            )
            self.out = nn.Linear(19584, 1)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            #x = torch.flatten(x, start_dim=2)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output
    
    model = CNN().double()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            output = model(train_x)[0]
            output = output.reshape(-1) 
            #output = model(train_x)
            #output = torch.squeeze(output)
            loss = loss_f(output, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print('Training loss:', sum(losses) / len(losses))
    
        preds = []
        ys = []
        for x, y in test_dataloader:
            test_x = Variable(x)
            test_y = Variable(y)
            test_y = torch.squeeze(test_y)
            pred = model(test_x)
            pred = torch.squeeze(pred)
            preds.append(pred.cpu().detach().numpy())
            ys.append(test_y.cpu().detach().numpy())
        val_mse = mean_squared_error(np.array(ys), np.array(preds), squared=True)
        print('Validation MSE:', val_mse)
    
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
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, models_path + 'checkpoint_CNN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.tar')
    
    print("Save the checkpoint in a TAR file.")
    print("----------")
    print()
    
    # Test the model.
    
    preds = []
    ys = []
    for x, y in test_dataloader:
        test_x = Variable(x)
        test_y = Variable(y)
        test_y = torch.squeeze(test_y)
        pred = model(test_x)
        pred = torch.squeeze(pred)
        preds.append(pred.cpu().detach().numpy())
        ys.append(test_y.cpu().detach().numpy())
    
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
    plt.title('CCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '_MSE_' + str(round(test_mse, 4)), fontsize=12)
    blue_patch = mpatches.Patch(color='blue', label='Predicted')
    red_patch = mpatches.Patch(color='red', label='Observed')
    ax.legend(handles=[blue_patch, red_patch])
    month = np.arange(0, len(ys), 1, dtype=int)
    ax.plot(month, np.array(preds), 'o', color='blue')
    ax.plot(month, np.array(ys), 'o', color='red')
    plt.savefig(out_path + 'plot_CCN_SSTAGraphDataset_windowsize_' + str(window_size) + '_leadtime_' + str(lead_time) + '_numsample_' + str(num_sample) + '_trainsplit_' + str(train_split) + '_numepoch_' + str(num_train_epoch) + '.png')
    
    print("Save the observed vs. predicted plot.")
    print("--------------------")
    print()