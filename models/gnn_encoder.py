import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool  

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, zdim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2_mu = GCNConv(hidden_dim, zdim)  # 用于生成z_mu
        self.conv2_sigma = GCNConv(hidden_dim, zdim)  # 用于生成z_sigma

        self.readout = torch.mean  # 可以选择mean, max, sum等

    def forward(self, x, edge_index, batch):

        x = F.elu(self.conv1(x, edge_index))
        z_mu = global_mean_pool(F.elu(self.conv2_mu(x, edge_index)), batch)
        z_sigma = global_mean_pool(F.elu(self.conv2_sigma(x, edge_index)), batch)
        
        return z_mu, z_sigma