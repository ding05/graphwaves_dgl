import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import GraphConv
from dgl import DGLGraph

class GCN(nn.Module):
    """
    GCN without applying edge features
    """
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, out_feats)
        self.out = nn.Linear(out_feats, 1)
        self.double()

    def forward(self, g, in_feat, edge_feat=None):
        h = self.conv1(g, in_feat)
        act_f = nn.LeakyReLU(0.1)
        h = act_f(h)
        h = self.conv2(g, h)
        h = act_f(h)
        h = self.conv2(g, h)
        h = act_f(h)
        h = self.conv3(g, h)
        h = self.out(h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

class GNNLayer(nn.Module):
    """
    GCN layers that uses edge features, ref: https://discuss.dgl.ai/t/using-edge-features-for-gcn-in-dgl/427/
    """
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GNNLayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
        self.double()

    def message_func(self, edges):
        return {"m": F.relu(self.W_msg(torch.cat([edges.src["h"], torch.unsqueeze(edges.data["h"], dim=1)], 1)))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats
            g.update_all(self.message_func, fn.sum("m", "h_neigh"))
            g.ndata["h"] = F.relu(self.W_apply(torch.cat([g.ndata["h"], g.ndata["h_neigh"]], 1)))
            return g.ndata["h"]

class GCN2(nn.Module):
    """
    GCN that uses GNNLayer()
    """
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(GCN2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(ndim_in, edim, 32, activation))
        self.layers.append(GNNLayer(32, edim, 32, activation))
        self.layers.append(GNNLayer(32, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class GCN3(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN3, self).__init__()
        self.layer1 = GCNLayer(in_feats, h_feats)
        self.layer2 = GCNLayer(h_feats, h_feats)
        self.layer3 = GCNLayer(h_feats, out_feats)
        self.double()
    
    def forward(self, g, in_feat, edge_feat=None):
        act_f = nn.LeakyReLU(0.1)
        x = act_f(self.layer1(g, in_feat))
        x = act_f(self.layer2(g, x))
        x = self.layer3(g, x)
        return x

class GAT(nn.Module):
    """
    GAT (not working yet)
    """
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer3 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
    
    def forward(self, h):
        h = self.layer1(h)
        act_f = nn.LeakyReLU(0.1)
        h = act_f(h)
        h = self.layer2(h)
        h = act_f(h)
        h = self.layer3(h)
        return h