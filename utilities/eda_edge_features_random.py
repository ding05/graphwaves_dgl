import numpy as np
from numpy import load, save

data_path = 'data/'

# Create an edge feature matrix with all elements being one.

node_features = load(data_path + 'node_features.npy')

num_edge = node_features.shape[0]

edge_features_random = np.random.randint(0, 999999999, size=(num_edge, num_edge))

save(data_path + 'edge_features_random.npy', edge_features_random)

print("Save the edge feature matrix in an NPY file.")
print("--------------------")
print()