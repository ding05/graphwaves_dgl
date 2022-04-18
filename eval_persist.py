import numpy as np
from numpy import load

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

train_split = 0.8
lead_time = 1

data_path = 'data/'
out_path = 'out/'

y = load(data_path + 'y.npy')

# Create a persistence model.

num_examples = len(y)
num_train = int(num_examples * train_split)

y_test = y[num_train+lead_time:]
pred = y[num_train:-lead_time]

# Test the model.

test_mse = mean_squared_error(pred, y_test, squared=False)
test_rmse = mean_squared_error(pred, y_test, squared=True)

print('Test RMSE:', test_rmse)
print("----------")
print()

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel('Month')
plt.ylabel('SSTA')
plt.title('Persist_SSTAGraphDataset_leadtime_1_numsample_1679_trainsplit_0.8_RMSE_' + str(round(test_rmse, 4)), fontsize=12)
blue_patch = mpatches.Patch(color='blue', label='Predicted')
red_patch = mpatches.Patch(color='red', label='Observed')
ax.legend(handles=[blue_patch, red_patch])
month = np.arange(0, len(y_test), 1, dtype=int)
ax.plot(month, pred, 'o', color='blue')
ax.plot(month, y_test, 'o', color='red')
plt.savefig(out_path + 'plot_persist_SSTAGraphDataset_leadtime_1_numsample_1679_trainsplit_0.8.png')

print("Save the observed vs. predicted plot.")
print("--------------------")
print()