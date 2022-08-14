import numpy as np
from numpy import asarray, save
import math

import pandas as pd
import xarray as xr

# Read the dataset.

soda = xr.open_dataset('data/soda_224_pt_l5.nc', decode_times=False)
print("SODA v2.2.4:")
print(soda)
print("--------------------")
print()

# Turn it into a smaller size.

soda_array = soda.to_array(dim='VARIABLE')
soda_smaller = np.array(soda_array[:,:,:,:,:,:])
soda_smaller = soda_smaller[2,:,0,:,::20,::20] # Drop the bnds dimension and the other two variables; take every 20th longitude and latitude.
soda_smaller = np.squeeze(soda_smaller, axis=0)

print("Shape of resized SODA:")
print(soda_smaller.shape)
print("--------------------")
print()

# Create the node feature matrix.

soda_smaller_transposed = soda_smaller.transpose(1,2,0)
soda_smaller_flattened = soda_smaller_transposed.reshape(soda_smaller.shape[1] * soda_smaller.shape[2],len(soda_smaller))

print("Shape of node feature matrix:")
print(soda_smaller_flattened.shape)
print("----------")
print()

# Drop the land nodes (the rows in the node feature matrix with NAs).
def dropna(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped=pd.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim==1:
        dropped=dropped.flatten()
    return dropped

soda_smaller_ocean_flattened = dropna(soda_smaller_flattened)

print("Shape of node feature matrix after land nodes were removed:")
print(soda_smaller_ocean_flattened.shape)
print("--------------------")
print()

# Replace SSTs with SSTAs in the node feature matrix.

train_split = 0.8
num_year = soda_smaller_ocean_flattened.shape[1] / 12
train_num_year = math.ceil(num_year * train_split)
test_num_year = int(num_year - train_num_year)

print("The number of years for training:", train_num_year)
print("The number of years for testing:", test_num_year)
print("----------")
print()

def avg(list):
  return sum(list) / len(list)

# Get SSTAs from an SST vector.
def get_ssta(time_series):
  monthly_avg = []
  for month in range(12):
    monthly_sst = time_series[month:train_num_year*12:12]
    monthly_avg.append(avg(monthly_sst))
    time_series[month::12] -= monthly_avg[month]
  return time_series

# Create the other output (y) vectors.

soda_westaus = soda.loc[dict(LAT='-29.75', LONN359_360='112.75')]
soda_westaus_sst = np.zeros((len(soda.TIME), 1))
soda_westaus_sst[:,:] = soda_westaus.variables["TEMP"][:,:]
soda_westaus_ssta = get_ssta(soda_westaus_sst)

soda_labrador = soda.loc[dict(LAT='53.75', LONN359_360='-54.25')]
soda_labrador_sst = np.zeros((len(soda.TIME), 1))
soda_labrador_sst[:,:] = soda_labrador.variables["TEMP"][:,:]
soda_labrador_ssta = get_ssta(soda_labrador_sst)

soda_equapacific = soda.loc[dict(LAT='-0.25', LONN359_360='-120.75')]
soda_equapacific_sst = np.zeros((len(soda.TIME), 1))
soda_equapacific_sst[:,:] = soda_equapacific.variables["TEMP"][:,:]
soda_equapacific_ssta = get_ssta(soda_equapacific_sst)

print("Output vector:")
print(soda_westaus_ssta)
print("Shape of output vector:")
print(soda_westaus_ssta.shape)
print("----------")
print()

# Save the output vector.

data_path = 'data/'
save(data_path + 'y_westaus.npy', soda_westaus_ssta)
save(data_path + 'y_labrador.npy', soda_labrador_ssta)
save(data_path + 'y_equapacific.npy', soda_equapacific_ssta)

print("Save the output vectors in NPY files.")
print("--------------------")
print()