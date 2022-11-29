import numpy as np
from numpy import asarray, save
import math

import pandas as pd
import xarray as xr

# Pre-process y.

# Read the dataset.

soda = xr.open_dataset("data/soda_224_pt_l5.nc", decode_times=False)
print("SODA v2.2.4:")
print(soda)
print("--------------------")
print()

# Turn it into a smaller size.

soda_array = soda.to_array(dim="VARIABLE")
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

soda_westaus = soda.loc[dict(LAT="-29.75", LONN359_360="112.75")]
soda_westaus_sst = np.zeros((len(soda.TIME), 1))
soda_westaus_sst[:,:] = soda_westaus.variables["TEMP"][:,:]
soda_westaus_ssta = get_ssta(soda_westaus_sst)

soda_labrador = soda.loc[dict(LAT="53.75", LONN359_360="-54.25")]
soda_labrador_sst = np.zeros((len(soda.TIME), 1))
soda_labrador_sst[:,:] = soda_labrador.variables["TEMP"][:,:]
soda_labrador_ssta = get_ssta(soda_labrador_sst)

soda_equapacific = soda.loc[dict(LAT="-0.25", LONN359_360="-120.75")]
soda_equapacific_sst = np.zeros((len(soda.TIME), 1))
soda_equapacific_sst[:,:] = soda_equapacific.variables["TEMP"][:,:]
soda_equapacific_ssta = get_ssta(soda_equapacific_sst)

soda_eastaus = soda.loc[dict(LAT="-37.25", LONN359_360="151.25")]
soda_eastaus_sst = np.zeros((len(soda.TIME), 1))
soda_eastaus_sst[:,:] = soda_eastaus.variables["TEMP"][:,:]
soda_eastaus_ssta = get_ssta(soda_eastaus_sst)

soda_chatham = soda.loc[dict(LAT="-44.25", LONN359_360="-176.75")]
soda_chatham_sst = np.zeros((len(soda.TIME), 1))
soda_chatham_sst[:,:] = soda_chatham.variables["TEMP"][:,:]
soda_chatham_ssta = get_ssta(soda_chatham_sst)

soda_med = soda.loc[dict(LAT="42.25", LONN359_360="6.75")]
soda_med_sst = np.zeros((len(soda.TIME), 1))
soda_med_sst[:,:] = soda_med.variables["TEMP"][:,:]
soda_med_ssta = get_ssta(soda_med_sst)

print("Output vector:")
print(soda_westaus_ssta)
print("Shape of output vector:")
print(soda_westaus_ssta.shape)
print("----------")
print()

# Save the output vector.

data_path = "data/"
"""
save(data_path + "y_westaus.npy", soda_westaus_ssta)
save(data_path + "y_labrador.npy", soda_labrador_ssta)
save(data_path + "y_equapacific.npy", soda_equapacific_ssta)
save(data_path + "y_eastaus.npy", soda_eastaus_ssta)
save(data_path + "y_chatham.npy", soda_chatham_ssta)
save(data_path + "y_med.npy", soda_med_ssta)
"""

# More places
# For convenience, define a function.

def extract_y(lat, lon, filename, data_path=data_path):
    soda_temp = soda.loc[dict(LAT=str(lat), LONN359_360=str(lon))]
    soda_temp_sst = np.zeros((len(soda.TIME), 1))
    soda_temp_sst[:,:] = soda_temp.variables["TEMP"][:,:]
    soda_temp_ssta = get_ssta(soda_temp_sst)
    save(data_path + "y_" + filename + ".npy", soda_temp_ssta)

"""
extract_y(26.75, 157.75, "nepacific")
extract_y(40.75, -147.75, "nwpacific")
extract_y(-40.75, -123.25, "southpacific")
extract_y(-11.25, 77.25, "indian")
extract_y(36.25, -43.75, "northatlantic")
extract_y(-29.25, -16.25, "southatlantic")

print("Save the output vectors in NPY files.")
print("--------------------")
print()
"""

# Pre-process X.

# The Tasman Sea.
soda = xr.open_dataset("data/soda_224_pt_l5.nc", decode_times=False)

soda_tasman = soda.where(soda.LAT < -25, drop=True)
soda_tasman = soda_tasman.where(soda.LAT > -40, drop=True)
soda_tasman = soda_tasman.where(soda.LONN359_360 > 147, drop=True)
soda_tasman = soda_tasman.where(soda.LONN359_360 < 170, drop=True)

soda_array_tasman = soda_tasman.to_array(dim="VARIABLE")
soda_smaller_tasman = np.array(soda_array_tasman[:,:,:,:,:,:])
soda_smaller_tasman = soda_smaller_tasman[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller_tasman = np.squeeze(soda_smaller_tasman, axis=0)
soda_smaller_tasman = np.transpose(soda_smaller_tasman, (2, 0, 1))

soda_tasman_ssta = get_ssta(soda_smaller_tasman)

save(data_path + "grids_tasman.npy", soda_tasman_ssta)

print("Save the grids in an NPY file")
print("--------------------")
print()

# The ENSO area.
soda_enso_e = soda.where(soda.LAT < 5, drop=True)
soda_enso_e = soda_enso_e.where(soda.LAT > -5, drop=True)
soda_enso_e = soda_enso_e.where(soda.LONN359_360 > 160, drop=True)
soda_enso_e = soda_enso_e.where(soda.LONN359_360 < 180, drop=True)

soda_array_enso_e = soda_enso_e.to_array(dim="VARIABLE")
soda_smaller_enso_e = np.array(soda_array_enso_e[:,:,:,:,:,:])
soda_smaller_enso_e = soda_smaller_enso_e[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller_enso_e = np.squeeze(soda_smaller_enso_e, axis=0)
soda_smaller_enso_e = np.transpose(soda_smaller_enso_e, (2, 0, 1))

soda_enso_w = soda.where(soda.LAT < 5, drop=True)
soda_enso_w = soda_enso_w.where(soda.LAT > -5, drop=True)
soda_enso_w = soda_enso_w.where(soda.LONN359_360 > -180, drop=True)
soda_enso_w = soda_enso_w.where(soda.LONN359_360 < -90, drop=True)

soda_array_enso_w = soda_enso_w.to_array(dim="VARIABLE")
soda_smaller_enso_w = np.array(soda_array_enso_w[:,:,:,:,:,:])
soda_smaller_enso_w = soda_smaller_enso_w[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller_enso_w = np.squeeze(soda_smaller_enso_w, axis=0)
soda_smaller_enso_w = np.transpose(soda_smaller_enso_w, (2, 0, 1))

soda_smaller_enso = np.concatenate((soda_smaller_enso_e, soda_smaller_enso_w), axis=2)
#soda_smaller_enso = np.c_[soda_smaller_enso_e, soda_smaller_enso_w]
soda_enso_ssta = get_ssta(soda_smaller_enso)

save(data_path + "grids_enso.npy", soda_enso_ssta)

#print(soda_smaller_enso_e.shape)
#print(soda_smaller_enso_w.shape)
#print(soda_enso_ssta.shape)

print("Save the grids in an NPY file")
print("--------------------")
print()

# The south Pacific Ocean
soda_southpacific_e = soda.where(soda.LAT < 0, drop=True)
soda_southpacific_e = soda_southpacific_e.where(soda.LONN359_360 > 146, drop=True)
soda_southpacific_e = soda_southpacific_e.where(soda.LONN359_360 < 180, drop=True)

soda_array_southpacific_e = soda_southpacific_e.to_array(dim="VARIABLE")
soda_smaller_southpacific_e = np.array(soda_array_southpacific_e[:,:,:,:,:,:])
soda_smaller_southpacific_e = soda_smaller_southpacific_e[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller_southpacific_e = np.squeeze(soda_smaller_southpacific_e, axis=0)
soda_smaller_southpacific_e = np.transpose(soda_smaller_southpacific_e, (2, 0, 1))

soda_southpacific_w = soda.where(soda.LAT < 0, drop=True)
soda_southpacific_w = soda_southpacific_w.where(soda.LONN359_360 > -180, drop=True)
soda_southpacific_w = soda_southpacific_w.where(soda.LONN359_360 < -67, drop=True)

soda_array_southpacific_w = soda_southpacific_w.to_array(dim="VARIABLE")
soda_smaller_southpacific_w = np.array(soda_array_southpacific_w[:,:,:,:,:,:])
soda_smaller_southpacific_w = soda_smaller_southpacific_w[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller_southpacific_w = np.squeeze(soda_smaller_southpacific_w, axis=0)
soda_smaller_southpacific_w = np.transpose(soda_smaller_southpacific_w, (2, 0, 1))

soda_smaller_southpacific = np.concatenate((soda_smaller_southpacific_e, soda_smaller_southpacific_w), axis=2)
soda_southpacific_ssta = get_ssta(soda_smaller_southpacific)

save(data_path + "grids_southpacific.npy", soda_southpacific_ssta)
"""
print(soda_smaller_southpacific_e.shape)
print(soda_smaller_southpacific_w.shape)
print(soda_southpacific_ssta.shape)
"""

print("Save the grids in an NPY file")
print("--------------------")
print()