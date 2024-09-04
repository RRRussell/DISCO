import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

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

class PositionDenoiseNet(Module):

    def __init__(self, position_dim, latent_dim, context_dim, residual, batch_context=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.batch_context = batch_context
        self.layers = ModuleList([
            ConcatSquashLinear(position_dim, latent_dim, context_dim+3),
            ConcatSquashLinear(latent_dim, position_dim, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Cell positions at some timestep t, (B, N, d).
            beta:  Time step information, (B, ).
            context:  Latents used as context, (B, F) or (B, N, F) depending on `batch_context`.
        """
        batch_size = x.size(0)
        num_points = x.size(1)

        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

        if self.batch_context:
            # Context is shared across the batch
            context = context.view(batch_size, 1, -1)   # (B, 1, F)
            ctx_emb = torch.cat([context, time_emb], dim=-1)    # (B, 1, F+3)
            ctx_emb = ctx_emb.expand(-1, num_points, -1)  # (B, N, F+3)
        else:
            # Context is independent for each point
            context = context.view(batch_size, num_points, -1)  # (B, N, F)
            ctx_emb = torch.cat([context, time_emb.expand(-1, num_points, -1)], dim=-1)  # (B, N, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out

class ExpressionDenoiseNet(Module):

    def __init__(self, expression_dim, latent_dim, context_dim, residual, batch_context=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.batch_context = batch_context
        self.layers = ModuleList([
            ConcatSquashLinear(expression_dim, latent_dim, context_dim+3),
            ConcatSquashLinear(latent_dim, expression_dim, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Cell positions at some timestep t, (B, N, d).
            beta:  Time step information, (B, ).
            context:  Latents used as context, (B, F) or (B, N, F) depending on `batch_context`.
        """
        batch_size = x.size(0)
        num_points = x.size(1)

        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

        if self.batch_context:
            # Context is shared across the batch
            context = context.view(batch_size, 1, -1)   # (B, 1, F)
            ctx_emb = torch.cat([context, time_emb], dim=-1)    # (B, 1, F+3)
            ctx_emb = ctx_emb.expand(-1, num_points, -1)  # (B, N, F+3)
        else:
            # Context is independent for each point
            context = context.view(batch_size, num_points, -1)  # (B, N, F)
            ctx_emb = torch.cat([context, time_emb.expand(-1, num_points, -1)], dim=-1)  # (B, N, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out
        
class DiffusionModel(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0: Input point, (B, N, d).
            context: Latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=2, flexibility=0.0, ret_traj=False, expansion_factor=1, test_item_list=None, mode="position"):
        """
        Sample from the diffusion model. Supports integrating known positions or gene expressions from test items.

        Args:
            num_points: Number of points to sample.
            context: Latent context tensor.
            point_dim: Dimensionality of the points.
            flexibility: Flexibility in the sampling process.
            ret_traj: If True, return the whole trajectory.
            test_item_list: List of test items with ground truth data.
            mode: Sampling mode - either "position" or "expression".
        """
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            
            # Integrate known positions or gene expressions
            if test_item_list is not None:
                x_next = self.integrate_known_data(x_next, test_item_list, t, expansion_factor, mode)
            
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

    def integrate_known_data(self, x_next, test_item_list, t, expansion_factor, mode="position"):
        """
        Integrate the known positions or gene expressions from the test area into the sampled positions.

        Args:
            x_next: (B, expanded_num_points, pos_dim) Sampled positions at time t.
            test_item_list: List of test items with ground truth data.
            t: Current time step in the diffusion process.
            mode: Integration mode - either "position" or "expression".
        """
        central_region_min = -1 / (2 * expansion_factor + 1)
        central_region_max = 1 / (2 * expansion_factor + 1)
        c0 = 1.0 / torch.sqrt(self.var_sched.alphas[t])
        c1 = (1 - self.var_sched.alphas[t]) / torch.sqrt(1 - self.var_sched.alpha_bars[t])

        for i, test_item in enumerate(test_item_list):
            # Extract the surrounding (real) cells from the expanded region
            surrounding_real_positions, surrounding_real_gene_expressions = extract_cells_from_expanded_region(test_item, expansion_factor=expansion_factor)
            num_surrounding_real_cells = surrounding_real_positions.shape[0]
            normalized_surrounding_real_positions = normalize_positions_within_test_area(surrounding_real_positions, test_item.test_area).to(x_next.device)
            x_next_normalized = normalize_positions(x_next[i])
            central_mask = (x_next_normalized[:, 0] > central_region_min) & (x_next_normalized[:, 0] < central_region_max) & \
                            (x_next_normalized[:, 1] > central_region_min) & (x_next_normalized[:, 1] < central_region_max)
            predicted_outer_indices = (~central_mask).nonzero(as_tuple=True)[0]  # Indices of outer cells
            num_predicted_outer_cells = predicted_outer_indices.size(0)
        
            if mode == "position":
                # Compare the number of surrounding (real) cells and predicted outer cells
                if num_surrounding_real_cells > num_predicted_outer_cells:
                    # If more real surrounding cells, randomly select the same number of real cells
                    selected_surrounding_indices = torch.randperm(num_surrounding_real_cells)[:num_predicted_outer_cells]
                    noise = torch.randn_like(normalized_surrounding_real_positions[selected_surrounding_indices]).to(x_next.device)
                    noisy_surrounding_real_positions = c0 * normalized_surrounding_real_positions[selected_surrounding_indices] + c1 * noise
                    x_next[i, predicted_outer_indices] = noisy_surrounding_real_positions
                else:
                    # If more predicted outer cells, randomly select the same number of predicted outer cells
                    selected_predicted_indices = torch.randperm(num_predicted_outer_cells)[:num_surrounding_real_cells]
                    noise = torch.randn_like(normalized_surrounding_real_positions).to(x_next.device)
                    noisy_surrounding_real_positions = c0 * normalized_surrounding_real_positions + c1 * noise
                    x_next[i, predicted_outer_indices[selected_predicted_indices]] = noisy_surrounding_real_positions

            elif mode == "expression":
                # Similar handling for gene expressions if needed
                # Normalize and integrate gene expression data
                pass

        return x_next
