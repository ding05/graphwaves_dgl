import numpy as np
from numpy import load

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import BoundaryNorm

data_path = 'data/'
out_path = 'out/'

node_features = load(data_path + 'node_features.npy')
y = load(data_path + 'y.npy')

# Get the spatial coordinates.

soda = xr.open_dataset('data/soda_224_pt_l5.nc', decode_times=False)
soda_array = soda.to_array(dim='VARIABLE')
soda_array_smaller = soda_array[:,:,:,:,::20,::20]
soda_array_smaller = soda_array_smaller[2,:,0,:,:,:]
lons, lats = np.meshgrid(soda_array_smaller.LONN359_360.values, soda_array_smaller.LAT.values)
soda_time_0 = soda_array_smaller.isel(LEV1_1=0, TIME=0)
soda_time_0_lons, soda_time_0_lats = np.meshgrid(soda_time_0.LONN359_360.values, soda_time_0.LAT.values)
soda_masked = soda_time_0.where(abs(soda_time_0_lons) + abs(soda_time_0_lats) > 0)
lons_smaller = soda_time_0_lons.flatten()[soda_masked.notnull().values.flatten()]
lats_smaller = soda_time_0_lats.flatten()[soda_masked.notnull().values.flatten()]

# Define the colormap's scale.

cmap = plt.get_cmap('seismic')
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.arange(-1, 1, 0.1)
index = np.searchsorted(bounds, 0)
bounds = np.insert(bounds, index, 0)
norm = BoundaryNorm(bounds, cmap.N)

for lag in [1, 2, 3, 6]:
#for lag in [1]:

  # Compute the correlation coefficients.

  corrcoefs = []

  for node in node_features:
    corrmat = np.corrcoef(node[:len(node)-lag], y[lag:,0])
    corrcoef = corrmat[0][1]
    corrcoefs.append(corrcoef)

  soda_coordinates = []
  for index in range(len(corrcoefs)):
    soda_coordinates.append([lons_smaller[index], lats_smaller[index], corrcoefs[index]])

  df_fine_soda = pd.DataFrame(soda_coordinates, columns=['Longitude', 'Latitude', 'Correlation Coefficient'])

  print('Correlation coefficients by coordinate:')
  print(df_fine_soda)
  print('----------')
  print()

  print(df_fine_soda)

  # Make the plot centering around the Pacific Ocean.
  df_fine_soda.iloc[:,0] = df_fine_soda.iloc[:,0] + np.where(df_fine_soda.iloc[:,0]<0, 360, 0)

  print(df_fine_soda)
  
  #plt.rcParams.update({'font.size': 22})
  
  figure(figsize=(12, 8))
  plt.scatter(x='Longitude', y='Latitude', c='Correlation Coefficient', s=80, data=df_fine_soda, norm=norm, cmap=cmap)
  plt.colorbar()
  plt.title('Linear Correlations Between SSTR Time Series and Time Series of Others Locations with a ' + str(lag) + '-Month Lag', fontsize=12)
  #plt.title('Correlations Between SSTA Time Series at One Point in the Pacific Ocean around the Equator and All Locations with ' + str(lag) + '-Month Lag', fontsize=12)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.savefig(out_path + 'plot_lag_' + str(lag) + '_correlation.png')

  print('Save the correlation plots.')
  print('--------------------')
  print()