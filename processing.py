from utils.processing_utils import *

import numpy as np
from numpy import asarray, save
import math

import pandas as pd
import xarray as xr

from sklearn.metrics.pairwise import haversine_distances

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
soda_smaller_flattened = soda_smaller_transposed.reshape(soda_smaller.shape[1] * soda_smaller.shape[2], len(soda_smaller))

print("Shape of node feature matrix:")
print(soda_smaller_flattened.shape)
print("----------")
print()

# Drop the land nodes (the rows in the node feature matrix with NAs).
soda_smaller_ocean_flattened = drop_rows_w_nas(soda_smaller_flattened)

print("Shape of node feature matrix after land nodes were removed:")
print(soda_smaller_ocean_flattened.shape)
print("--------------------")
print()

# Create the edge feature matrix.

soda_array_smaller = soda_array[:,:,:,:,::20,::20]
soda_array_smaller = soda_array_smaller[2,:,0,:,:,:]
lons, lats = np.meshgrid(soda_array_smaller.LONN359_360.values, soda_array_smaller.LAT.values)

# Remove the land nodes.
soda_time_0 = soda_array_smaller.isel(LEV1_1=0, TIME=0)
soda_time_0_lons, soda_time_0_lats = np.meshgrid(soda_time_0.LONN359_360.values, soda_time_0.LAT.values)
soda_masked = soda_time_0.where(abs(soda_time_0_lons) + abs(soda_time_0_lats) > 0)

soda_masked.values.flatten()[soda_masked.notnull().values.flatten()]

lons_ocean = soda_time_0_lons.flatten()[soda_masked.notnull().values.flatten()]
lons_ocean = lons_ocean[::]
lats_ocean = soda_time_0_lats.flatten()[soda_masked.notnull().values.flatten()]
lats_ocean = lats_ocean[::]

lons_ocean *= np.pi/180
lats_ocean *= np.pi/180

points_ocean = np.concatenate([np.expand_dims(lats_ocean.flatten(),-1), np.expand_dims(lons_ocean.flatten(),-1)],-1)

distance_ocean = 6371*haversine_distances(points_ocean)

distance_ocean_diag = distance_ocean
distance_ocean_diag[distance_ocean_diag==0] = 1

distance_ocean_recip = np.reciprocal(distance_ocean_diag)

print("Edge feature matrix:")
print(distance_ocean_recip)
print("Shape of edge feature matrix:")
print(distance_ocean_recip.shape)
print("--------------------")
print()

# Save the two output matrices.

data_path = "data/"
save(data_path + "node_features.npy", soda_smaller_ocean_flattened)
save(data_path + "edge_features.npy", distance_ocean_recip)

print("Save the two matrices in NPY files.")
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

# Get SSTAs from an SST vector.
soda_ssta = []
for row in soda_smaller_ocean_flattened:
  soda_ssta.append(get_ssta(row, train_num_year))
soda_ssta = np.array(soda_ssta)

print("Node feature matrix after SSTs were replaced by SSTAs:")
print(soda_ssta)
print("----------")
print()

print("Shape of the updated node feature matrix:")
print(soda_ssta.shape)
print("----------")
print()

save(data_path + "node_features.npy", soda_ssta)

print("Update the node feature matrix and saving it in an NPY file.")
print("--------------------")
print()

"""
# Create the output (y) vector.

soda_bop = soda.loc[dict(LAT="-34.75", LONN359_360="177.75")]
soda_bop_sst = np.zeros((len(soda.TIME), 1))
soda_bop_sst[:,:] = soda_bop.variables["TEMP"][:,:]

soda_bop_ssta = get_ssta(soda_bop_sst, train_num_year)

print("Output vector:")
print(soda_bop_ssta)
print("Shape of output vector:")
print(soda_bop_ssta.shape)
print("----------")
print()

# Save the output vector.

save(data_path + "y.npy", soda_bop_ssta)

print("Save the output vector in an NPY file.")
print("--------------------")
print()
"""
