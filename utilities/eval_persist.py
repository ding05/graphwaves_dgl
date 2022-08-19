import numpy as np
from numpy import load

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

train_split = 0.8
lead_time = 1

data_path = 'data/'
out_path = 'out/'

y = load(data_path + 'y.npy')

for lead_time in [1]:

    # Create a persistence model.
    
    num_examples = len(y)
    num_train = int(num_examples * train_split)
    
    ys = y[num_train+lead_time:]
    preds = y[num_train:-lead_time]
    
    # Test the model.
    
    test_mse = mean_squared_error(ys, preds, squared=True)
    test_rmse = mean_squared_error(ys, preds, squared=False)
    
    print('Test MSE:', test_mse)
    print("----------")
    print()
    
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
    plt.savefig(out_path + 'plot_persist_SSTA_leadtime_' + str(lead_time) + '_numsample_1679_trainsplit_0.8.png')
    
    print("Save the observed vs. predicted plot.")
    print("--------------------")
    print()