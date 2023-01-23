import numpy as np
from numpy import load

from sklearn.metrics import mean_squared_error, classification_report

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

train_split = 0.8

data_path = 'data/'
out_path = 'out/'

loc_name = 'Chatham'

y = load(data_path + 'y_chatham.npy')

for lead_time in [1, 2, 3, 6]:

    # Create a persistence model.
    
    num_examples = len(y)
    num_train = int(num_examples * train_split)
    
    ys = y[num_train+lead_time:]
    preds = y[num_train:-lead_time]
    
    # Test the model.
    
    test_mse = mean_squared_error(ys, preds, squared=True)
    test_rmse = mean_squared_error(ys, preds, squared=False)
    
    print('Test MSE:', test_mse)
    print('----------')
    print()

    # Increase the fontsize.
    plt.rcParams.update({'font.size': 20})
    
    # Calculate the threshold for 90th percentile and mark the outliers.
    y_temp = y.squeeze(axis=1)
    y_train = y_temp[:int(len(y)*0.8)]
    y_train_sorted = np.sort(y_train)
    threshold = y_train_sorted[int(len(y_train_sorted)*0.9):][0]
    threshold_weak = y_train_sorted[int(len(y_train_sorted)*0.8):][0]
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
    plt.xlabel('Month')
    plt.ylabel('SST Residual')
    plt.title('MSE: ' + str(round(test_mse, 4)) + ', MSE above 90th: ' + str(round(ol_test_mse, 4)))
    patch_a = mpatches.Patch(color='pink', label='Obs')
    patch_b = mpatches.Patch(color='red', label='Obs above 90th')
    patch_c = mpatches.Patch(color='skyblue', label='Pred')
    patch_d = mpatches.Patch(color='blue', label='Pred for Obs above 90th')
    ax.legend(handles=[patch_a, patch_b, patch_c, patch_d])
    month = np.arange(0, len(ys), 1, dtype=int)
    plt.plot(month, np.array(ys, dtype=object), linestyle='-', color='pink')
    ax.plot(month, np.array(ys, dtype=object), 'o', color='pink')
    ax.plot(month, np.array(y_outliers, dtype=object), 'o', color='red')
    plt.plot(month, np.array(preds, dtype=object), linestyle='-', color='skyblue')
    ax.plot(month, np.array(preds, dtype=object), 'o', color='skyblue')
    ax.plot(month, np.array(pred_outliers, dtype=object), 'o', color='blue')
    plt.savefig(out_path + 'pred_a_SSTASODA' + loc_name + '_persist_' + str(lead_time) + '_1679_0.8.png')

    fig, ax = plt.subplots(figsize=(12, 8))
    lim = max(np.abs(np.array(preds)).max(), np.abs(np.array(ys)).max())
    ax.set_xlim([-lim-0.1, lim+0.1])
    ax.set_ylim([-lim-0.1, lim+0.1])
    plt.xlabel('Obs SST Residual')
    plt.ylabel('Pred SST Residual')
    plt.title('MSE: ' + str(round(test_mse, 4)) + ', MSE above 90th: ' + str(round(ol_test_mse, 4)))
    ax.plot(np.array(ys, dtype=object), np.array(preds, dtype=object), 'o', color='black')
    transform = ax.transAxes
    line_a = mlines.Line2D([0, 1], [0, 1], color='red')
    line_a.set_transform(transform)
    ax.add_line(line_a)
    patch_a = mpatches.Patch(color='pink', label='Obs above 90th')
    ax.legend(handles=[patch_a])
    ax.axvspan(threshold, max(ys)+0.1, color='pink')
    plt.savefig(out_path + 'pred_b_SSTASODA' + loc_name + '_persist_' + str(lead_time) + '_1679_0.8.png')
    
    print('Save the observed vs. predicted plot.')
    print('--------------------')
    print()

    # Confusion matrix
    
    ys_masked = ['MHW Weak Indicator (>80th)' if ys[i] >= threshold_weak else 'None' for i in range(len(ys))]
    ys_masked = ['MHW Strong Indicator (>90th)' if ys[i] >= threshold else ys_masked[i] for i in range(len(ys_masked))]
    preds_masked = ['MHW Weak Indicator (>80th)' if preds[i] >= threshold_weak else 'None' for i in range(len(preds))]
    preds_masked = ['MHW Strong Indicator (>90th)' if preds[i] >= threshold else preds_masked[i] for i in range(len(preds_masked))]
    
    classification_results = classification_report(ys_masked, preds_masked, digits=4)

    with open(out_path + 'classification_SSTASODA' + loc_name + '_persist_' + str(lead_time) + '_1679_0.8.txt', 'w') as f:
        print(classification_results, file=f)
    
    print('Save the classification results in a TXT file.')
    print('----------')
    print()    