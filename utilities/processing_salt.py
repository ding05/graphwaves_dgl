import numpy as np
from numpy import asarray, save
import math

import pandas as pd
import xarray as xr

# Read the dataset.

soda = xr.open_dataset("data/soda_224_salt_l5.nc", decode_times=False)
print("SODA v2.2.4:")
print(soda)
print("--------------------")
print()

# Turn it into a smaller size.

soda_array = soda.to_array(dim="VARIABLE")
soda_smaller = np.array(soda_array[:,:,:,:,:,:])
soda_smaller = soda_smaller[2,:,0,:,::,::] # Drop the bnds dimension and the other two variables.
soda_smaller = np.squeeze(soda_smaller, axis=0)

print("Shape of resized SODA:")
print(soda_smaller.shape)
print("--------------------")
print()

data_path = "data/"
save(data_path + "grids_salt.npy", soda_smaller)

print("Save the grids in an NPY file")
print("--------------------")
print()

soda_smaller = np.array(soda_array[:,:,:,:,:,:])
soda_smaller = soda_smaller[2,:,0,:,::2,::2] # Drop the bnds dimension and the other two variables; take every 2nd longitude and latitude.
soda_smaller = np.squeeze(soda_smaller, axis=0)

print("Shape of resized SODA:")
print(soda_smaller.shape)
print("--------------------")
print()


save(data_path + "grids_salt_half.npy", soda_smaller)

print("Save the grids in an NPY file")
print("--------------------")
print()

print(soda_smaller)