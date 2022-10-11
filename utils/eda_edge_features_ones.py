import numpy as np
from numpy import load, save

data_path = "data/"

# Create an edge feature matrix with all elements being one.

node_features = load(data_path + "node_features.npy")

num_edge = node_features.shape[0]

edge_features_ones = np.ones((num_edge, num_edge))

save(data_path + "edge_features_ones.npy", edge_features_ones)

print("Save the edge feature matrix in an NPY file.")
print("--------------------")
print()