import numpy as np
import pandas as pd

def drop_rows_w_nas(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped=pd.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim==1:
        dropped=dropped.flatten()
    return dropped

def get_ssta(time_series):
  monthly_avg = []
  for month in range(12):
    monthly_sst = time_series[month:train_num_year*12:12]
    monthly_avg.append(avg(monthly_sst))
    time_series[month::12] -= monthly_avg[month]
  return time_series