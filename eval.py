import torch
import torch.nn as nn

from dgl.nn import GraphConv

window_size = 3

data_path = 'data/'
models_path = 'models/'

# Load the model.

# Define the model before loading.
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)
        self.double()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

model = GCN(window_size, 16, 1)
model.load_state_dict(torch.load(models_path + 'model_SSTAGraphDataset_windowsize_3_leadtime_1_trainsplit_0.8.pt'))
model.eval()

print("Loaded model:")
print(model.eval())
print("--------------------")