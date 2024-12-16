import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph

from models.common import *
from utils import *

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class GNNDenoiseNet(Module):
    def __init__(self, input_dim, latent_dim, context_dim, residual=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        # We no longer handle batch_context here. Assume ctx is always node-level.
        # input_dim, latent_dim, context_dim+3 for time_emb dimension
        self.layers = ModuleList([
            ConcatSquashGNN(input_dim, latent_dim, context_dim+3),
            ConcatSquashGNN(latent_dim, input_dim, context_dim+3)
        ])

    def forward(self, x, edge_index, ctx, batch=None):
        # x: (N, d)
        # ctx: (N, F+3), already includes time_emb and context merged outside
        # edge_index: (2, E)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(x=out, edge_index=edge_index, ctx=ctx, batch=batch)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out
        
class DiffusionModel(Module):
    def __init__(self, net, var_sched):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None, mode="position", position=None, k=5, batch_context=True):
        B, N, d = x_0.size()
        device = x_0.device
        if t is None:
            t = self.var_sched.uniform_sample_t(B)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        coord_for_graph = position if (position is not None and mode == "expression") else x_0
        BN = B * N
        x_flat = coord_for_graph.view(BN, 2)
        batch = torch.repeat_interleave(torch.arange(B, device=device), N)
        edge_index = knn_graph(x_flat, k=k, batch=batch)

        x_input = (c0 * x_0 + c1 * e_rand).view(BN, d)

        if batch_context and context.dim() == 2:
            context = context.unsqueeze(1).expand(B, N, -1)
        context = context.reshape(BN, -1)

        beta_node = beta.repeat_interleave(N)
        time_emb = torch.cat([
            beta_node.unsqueeze(-1),
            torch.sin(beta_node).unsqueeze(-1),
            torch.cos(beta_node).unsqueeze(-1)
        ], dim=-1)  # (B*N, 3)

        ctx = torch.cat([context, time_emb], dim=-1)  # (B*N, F+3)

        e_theta = self.net(x=x_input, edge_index=edge_index, ctx=ctx, batch=batch)
        loss = F.mse_loss(e_theta, e_rand.view(BN, d), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=2, flexibility=0.0, ret_traj=False, expansion_factor=1,
               test_item_list=None, mode="position", position=None, k=5, batch_context=True):
        device = context.device
        B = context.size(0)
        x_T = torch.randn(B, num_points, point_dim, device=device)
        traj = {self.var_sched.num_steps: x_T}

        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*B]  # (B,)
            B, N, d = x_t.size()

            coord_for_graph = position if (position is not None and mode == "expression") else x_t
            BN = B * N
            x_flat = coord_for_graph.view(BN, 2)
            batch = torch.repeat_interleave(torch.arange(B, device=device), N)
            edge_index = knn_graph(x_flat, k=k, batch=batch)

            # Flatten x_t for net
            x_input = x_t.view(BN, d)

            if batch_context and context.dim() == 2:
                context = context.unsqueeze(1).expand(B, N, -1)
            ctx_flat = context.reshape(BN, -1)

            beta_node = beta.repeat_interleave(N)
            time_emb = torch.cat([
                beta_node.unsqueeze(-1),
                torch.sin(beta_node).unsqueeze(-1),
                torch.cos(beta_node).unsqueeze(-1)
            ], dim=-1)  # (B*N,3)

            ctx = torch.cat([ctx_flat, time_emb], dim=-1)

            e_theta = self.net(x=x_input, edge_index=edge_index, ctx=ctx, batch=batch)
            x_next = c0 * (x_t.view(BN,d) - c1 * e_theta) + sigma * z.view(BN,d)
            x_next = x_next.view(B, N, d)

            if test_item_list is not None:
                x_next = self.integrate_known_data(x_next, test_item_list, t, expansion_factor, mode, position=position)

            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        return traj if ret_traj else traj[0]

    def integrate_known_data(self, x_next, test_item_list, t, expansion_factor, mode="position", position=None):
        """
        Integrate the known positions or gene expressions from the test area into the sampled positions.

        Args:
            x_next: (B, expanded_num_points, dim) Sampled positions or expressions at time t.
            test_item_list: List of test items with ground truth data.
            t: Current time step in the diffusion process.
            mode: Integration mode - either "position" or "expression".
            position: (B, expanded_num_points, 2) Optional positions for finding surrounding cells.
        """
        # Use `position` if provided; otherwise, use `x_next` directly
        coord = position if position is not None else x_next

        # central_region_min = -1 / (2 * expansion_factor + 1)
        # central_region_max = 1 / (2 * expansion_factor + 1)
        c0 = 1.0 / torch.sqrt(self.var_sched.alphas[t])
        c1 = (1 - self.var_sched.alphas[t]) / torch.sqrt(1 - self.var_sched.alpha_bars[t])

        for i, test_item in enumerate(test_item_list):
            # Extract the surrounding (real) cells from the expanded region
            surrounding_real_positions, surrounding_real_gene_expressions = extract_cells_from_expanded_region(
                test_item, expansion_factor=expansion_factor
            )
            surrounding_real_gene_expressions = surrounding_real_gene_expressions.to(coord.device)
            num_surrounding_real_cells = surrounding_real_positions.shape[0]
            normalized_surrounding_real_positions = normalize_positions_within_test_area(
                surrounding_real_positions, test_item.test_area
            ).to(coord.device)
            normalized_surrounding_real_positions = normalize_positions(normalized_surrounding_real_positions)

            # Use the provided `coord` for central mask computation
            coord_normalized = normalize_positions(coord[i])
            # central_mask = (coord_normalized[:, 0] > central_region_min) & (coord_normalized[:, 0] < central_region_max) & \
            #             (coord_normalized[:, 1] > central_region_min) & (coord_normalized[:, 1] < central_region_max)
            
            central_positions, central_mask = select_central_cells(coord_normalized, expansion_factor=expansion_factor)
            predicted_outer_indices = (~central_mask).nonzero(as_tuple=True)[0]  # Indices of outer cells
            num_predicted_outer_cells = predicted_outer_indices.size(0)

            if mode == "position":
                # Handle position integration
                if num_surrounding_real_cells > num_predicted_outer_cells:
                    selected_surrounding_indices = torch.randperm(num_surrounding_real_cells)[:num_predicted_outer_cells]
                    noise = torch.randn_like(normalized_surrounding_real_positions[selected_surrounding_indices]).to(coord.device)
                    noisy_surrounding_real_positions = c0 * normalized_surrounding_real_positions[selected_surrounding_indices] + c1 * noise
                    x_next[i, predicted_outer_indices] = noisy_surrounding_real_positions
                else:
                    selected_predicted_indices = torch.randperm(num_predicted_outer_cells)[:num_surrounding_real_cells]
                    noise = torch.randn_like(normalized_surrounding_real_positions).to(coord.device)
                    noisy_surrounding_real_positions = c0 * normalized_surrounding_real_positions + c1 * noise
                    x_next[i, predicted_outer_indices[selected_predicted_indices]] = noisy_surrounding_real_positions

            elif mode == "expression":
                # Handle gene expression integration
                if num_surrounding_real_cells > num_predicted_outer_cells:
                    selected_surrounding_indices = torch.randperm(num_surrounding_real_cells)[:num_predicted_outer_cells]
                    noise = torch.randn_like(surrounding_real_gene_expressions[selected_surrounding_indices]).to(coord.device)
                    noisy_surrounding_real_expressions = c0 * surrounding_real_gene_expressions[selected_surrounding_indices] + c1 * noise
                    x_next[i, predicted_outer_indices] = noisy_surrounding_real_expressions
                else:
                    selected_predicted_indices = torch.randperm(num_predicted_outer_cells)[:num_surrounding_real_cells]
                    noise = torch.randn_like(surrounding_real_gene_expressions).to(coord.device)
                    noisy_surrounding_real_expressions = c0 * surrounding_real_gene_expressions + c1 * noise
                    x_next[i, predicted_outer_indices[selected_predicted_indices]] = noisy_surrounding_real_expressions

        return x_next
