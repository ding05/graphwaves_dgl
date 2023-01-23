import numpy as np
from numpy import asarray, save, load
import math

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

data_path = 'data/'
out_path = 'out/'

def plot_pred_obs(pred, y):
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
    plt.savefig(out_path + 'pred_a_SSTASaltSODAHalf_' + str(net_class) + '_' + str(num_hid_feat) + '_' + str(num_out_feat) + '_' + str(window_size) + '_' + str(lead_time) + '_' + str(num_sample) + '_' + str(train_split) + '_' + str(loss_function) + '_' + str(optimizer) + '_' + str(activation) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(weight_decay) + '_' + str(batch_size) + '_' + str(num_train_epoch) + '.png')