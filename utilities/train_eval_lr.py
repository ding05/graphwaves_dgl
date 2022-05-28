import numpy as np
from numpy import load

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

train_split = 0.8
lead_time = 1

data_path = 'data/'
out_path = 'out/'

for lead_time in [1, 2, 3, 6, 12, 23]:

    # Load the node feature matrix and process it fit for linear regression models.
    
    node_features = load(data_path + 'node_features.npy')
    y = load(data_path + 'y.npy')
    
    x = np.transpose(node_features)
    
    y = y[lead_time:]
    x = x[:len(x)-lead_time]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_split, shuffle=False)
    
    # Train a multiple linear regression model.
    
    reg = LinearRegression().fit(x_train, y_train)
    
    print("Complete training.")
    print("----------")
    print()
    
    # Test the model.
    
    pred = reg.predict(x_test)
    
    for i in range(len(pred)):
      print('Observed:', y_test[i], '; predicted:', pred[i])
    
    print("----------")
    print()
    
    test_mse = mean_squared_error(y_test, pred, squared=True)
    test_rmse = mean_squared_error(y_test, pred, squared=False)
    
    print('Test MSE:', test_mse)
    print("----------")
    print()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel('Month')
    plt.ylabel('SSTA')
    plt.title('LR_SSTAGraphDataset_leadtime_' + str(lead_time) + '_numsample_1679_trainsplit_0.8_MSE_' + str(round(test_mse, 4)), fontsize=12)
    blue_patch = mpatches.Patch(color='blue', label='Predicted')
    red_patch = mpatches.Patch(color='red', label='Observed')
    ax.legend(handles=[blue_patch, red_patch])
    month = np.arange(0, len(y_test), 1, dtype=int)
    ax.plot(month, pred, 'o', color='blue')
    ax.plot(month, y_test, 'o', color='red')
    plt.savefig(out_path + 'plot_LR_SSTAGraphDataset_leadtime_' + str(lead_time) + '_numsample_1679_trainsplit_0.8.png')
    
    print("Save the observed vs. predicted plot.")
    print("--------------------")
    print()