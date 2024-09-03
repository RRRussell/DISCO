import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, zdim, gnn_type='gcn', readout='mean'):
        super().__init__()

        # Define GNN layer types
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2_mu = GCNConv(hidden_dim, zdim)
            self.conv2_sigma = GCNConv(hidden_dim, zdim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2_mu = GATConv(hidden_dim, zdim)
            self.conv2_sigma = GATConv(hidden_dim, zdim)
        elif gnn_type == 'graphsage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2_mu = SAGEConv(hidden_dim, zdim)
            self.conv2_sigma = SAGEConv(hidden_dim, zdim)
        elif gnn_type == 'graphtransformer':
            self.conv1 = TransformerConv(input_dim, hidden_dim, heads=1)
            self.conv2_mu = TransformerConv(hidden_dim, zdim, heads=1)
            self.conv2_sigma = TransformerConv(hidden_dim, zdim, heads=1)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Define readout function types
        if readout == 'mean':
            self.readout = global_mean_pool
        elif readout == 'max':
            self.readout = global_max_pool
        elif readout == 'sum':
            self.readout = global_add_pool
        else:
            raise ValueError(f"Unknown readout type: {readout}")

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))

        z_mu = self.readout(F.elu(self.conv2_mu(x, edge_index)), batch)
        z_sigma = self.readout(F.elu(self.conv2_sigma(x, edge_index)), batch)

        return z_mu, z_sigma
