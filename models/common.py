import torch
import torch.nn as nn
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

class ConcatSquashGNN(nn.Module):
    def __init__(self, input_dim, output_dim, ctx_dim, gnn_type="gcn", hidden_dim=128, heads=1, dropout=0.0, negative_slope=0.2):
        super().__init__()
        # _hyper_gate and _hyper_bias produce gate and bias from ctx
        self._hyper_gate = nn.Linear(ctx_dim, output_dim)
        self._hyper_bias = nn.Linear(ctx_dim, output_dim, bias=False)

        # Build GNN layers depending on gnn_type
        self.gnn_type = gnn_type.lower().strip()
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = F.leaky_relu

        # First conv
        self.conv1 = self._build_conv(self.gnn_type, input_dim, hidden_dim, heads)
        # Second conv
        self.conv2 = self._build_conv(self.gnn_type, hidden_dim * (heads if self.gnn_type=="gat" or self.gnn_type=="transformer" else 1), output_dim, heads)

    def _build_conv(self, gnn_type, in_dim, out_dim, heads):
        """
        Build a graph convolution layer based on gnn_type.
        Adjust parameters as needed for each GNN variant.
        """
        if gnn_type == "gcn":
            # GCNConv(in_channels, out_channels, ...)
            return GCNConv(in_dim, out_dim)
        elif gnn_type == "sage":
            # SAGEConv(in_channels, out_channels, ...)
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == "gat":
            # GATConv(in_channels, out_channels, heads=heads, concat=True by default)
            # out_dim is the per-head dimension, total output dim = out_dim * heads
            return GATConv(in_dim, out_dim, heads=heads, dropout=self.dropout)
        elif gnn_type == "transformer":
            # TransformerConv(in_channels, out_channels, heads=heads, ...)
            # Similar to GAT, output is out_dim * heads if concat=True.
            return TransformerConv(in_dim, out_dim, heads=heads, dropout=self.dropout, beta=False)
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")
        
    def forward(self, x, edge_index, ctx, batch=None):
        # x: (N, input_dim)
        # ctx: (N, ctx_dim)
        # edge_index: (2, E)
        # batch: (N,) optional, if needed for pooling (not mandatory for GCNConv itself)

        # Compute gate and bias
        gate = torch.sigmoid(self._hyper_gate(ctx))  # (N, output_dim)
        bias = self._hyper_bias(ctx)                 # (N, output_dim)

        x_out = self.act(self.conv1(x, edge_index), negative_slope=self.negative_slope)
        if self.dropout > 0.0:
            x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.conv2(x_out, edge_index)  # (N, output_dim)

        ret = x_out * gate + bias
        return ret

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

