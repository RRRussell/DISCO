import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *

class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=args.position_dim)
        self.tissue_embedding = torch.nn.Embedding(args.num_tissues, args.tissue_dim)
        
        # Diffusion model for position
        self.position_diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=args.position_dim, context_dim=args.latent_dim+args.tissue_dim, residual=args.residual, batch_context=True),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

        # Diffusion model for gene expression
        self.expression_diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=args.expression_dim, context_dim=args.position_dim+args.tissue_dim, residual=args.residual, batch_context=False),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        
    def get_position_loss(self, x, tissue_labels, kl_weight=1.0):
        """
        Compute loss for position prediction.
        Args:
            x:  Input point clouds, (B, N, d).
            tissue_labels: Tensor of tissue labels, (B,)
        """
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        loss_prior = torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mu ** 2 - z_sigma.exp(), dim=1), dim=0)
        
        # Get tissue embeddings and concatenate with z
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)

        loss_recons = self.position_diffusion.get_loss(x, z_with_tissue)

        loss = kl_weight * loss_prior + loss_recons
        return loss

    def get_expression_loss(self, predicted_positions, real_positions, real_expressions, tissue_labels):
        """
        Compute loss for gene expression prediction using the nearest real positions.
        Args:
            predicted_positions: Predicted point clouds, (B, N, d).
            real_positions: Real point clouds, (B, N, d).
            real_expressions: Real gene expressions, (B, N, G).
            tissue_labels: Tensor of tissue labels, (B,)
        """
        # Find nearest real positions for each predicted position
        nearest_expressions = self.get_nearest_expressions(predicted_positions, real_positions, real_expressions)
        # Get tissue embeddings and concatenate with predicted positions
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        # Expand tissue_embed to match the shape of predicted_positions
        tissue_embed = tissue_embed.unsqueeze(1).expand(-1, predicted_positions.size(1), -1)  # (B, N, tissue_dim)
        # Concatenate predicted positions with tissue embeddings to create context
        context = torch.cat([predicted_positions, tissue_embed], dim=-1)  # (B, N, d + tissue_dim)

        loss_recons = self.expression_diffusion.get_loss(nearest_expressions, context)
        return loss_recons

    def get_nearest_expressions(self, predicted_positions, real_positions, real_expressions):
        """
        Find the nearest real positions and return their gene expressions.
        Args:
            predicted_positions: Predicted point clouds, (B, N, d).
            real_positions: Real point clouds, (B, N, d).
            real_expressions: Real gene expressions, (B, N, G).
        Returns:
            nearest_expressions: Nearest gene expressions for predicted positions, (B, N, G).
        """
        B, N, d = predicted_positions.shape
        G = real_expressions.shape[2]  # Gene expression dimension

        # Compute pairwise distances between predicted and real positions
        distances = torch.cdist(predicted_positions, real_positions)  # (B, N, N)
        
        # Find the nearest indices in the real positions
        nearest_indices = distances.argmin(dim=-1)  # (B, N)

        # Gather the nearest gene expressions
        # We need to gather expressions along the batch dimension as well
        nearest_expressions = torch.gather(
            real_expressions,  # (B, N, G)
            dim=1,
            index=nearest_indices.unsqueeze(-1).expand(-1, -1, G)  # (B, N, G)
        )  # The result will be of shape (B, N, G)

        return nearest_expressions
    
    def sample_positions(self, z, tissue_labels, num_points, flexibility=0.0, truncate_std=None):
        """
        Sample positions using the position diffusion model.
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
            tissue_labels: Tensor of tissue labels, (B,)
            num_points: Number of points to sample per cloud.
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
            
        # Get tissue embeddings and concatenate with z
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)
        
        samples = self.position_diffusion.sample(num_points, context=z_with_tissue, point_dim=self.args.position_dim, flexibility=flexibility)
        
        # Apply min-max normalization to the x and y dimensions separately
        x_min = samples[:, :, 0].min()
        x_max = samples[:, :, 0].max()
        y_min = samples[:, :, 1].min()
        y_max = samples[:, :, 1].max()
        # Normalize the x dimension
        samples[:, :, 0] = (samples[:, :, 0] - x_min) / (x_max - x_min + 1e-8)
        # Normalize the y dimension
        samples[:, :, 1] = (samples[:, :, 1] - y_min) / (y_max - y_min + 1e-8)
        
        return samples
    
    def sample_expressions(self, predicted_positions, tissue_labels, flexibility=0.0):
        """
        Sample gene expressions using the expression diffusion model.
        Args:
            predicted_positions: Predicted point clouds, (B, N, d).
            tissue_labels: Tensor of tissue labels, (B,)
        """
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        tissue_embed = tissue_embed.unsqueeze(1).expand(-1, predicted_positions.size(1), -1)  # (B, N, tissue_dim)
        context = torch.cat([predicted_positions, tissue_embed], dim=-1)  # (B, N, d + tissue_dim)
        
        gene_expressions = self.expression_diffusion.sample(predicted_positions.shape[1], context=context, point_dim=self.args.expression_dim, flexibility=flexibility)
        return gene_expressions

    def sample(self, tissue_labels, num_points, flexibility=0.0):
        """
        High-level sample function to generate both positions and gene expressions.
        Args:
            tissue_labels: Tensor of tissue labels, (B,)
            num_points: Number of points to sample per cloud.
            num_genes: Number of genes to sample per position.
        Returns:
            predicted_positions: Sampled positions, (B, N, d).
            predicted_expressions: Sampled gene expressions, (B, N, G).
        """
        # Step 1: Generate random latent variables z
        z = torch.randn(tissue_labels.size(0), self.args.latent_dim).to(tissue_labels.device)
        
        # Step 2: Sample positions
        predicted_positions = self.sample_positions(z, tissue_labels, num_points=num_points, flexibility=flexibility)
        
        # Step 3: Sample gene expressions based on predicted positions
        predicted_expressions = self.sample_expressions(predicted_positions, tissue_labels, flexibility=flexibility)
        
        return predicted_positions, predicted_expressions
    
    def map_position_back(self, predicted_positions, test_area):
        predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
        predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y
        return predicted_positions


# class GaussianVAE(Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=args.input_dim)
#         self.tissue_embedding = torch.nn.Embedding(args.num_tissues, args.tissue_dim)
#         self.diffusion = DiffusionPoint(
#             net = PointwiseNet(point_dim=args.input_dim, context_dim=args.latent_dim+args.tissue_dim, residual=args.residual),
#             var_sched = VarianceSchedule(
#                 num_steps=args.num_steps,
#                 beta_1=args.beta_1,
#                 beta_T=args.beta_T,
#                 mode=args.sched_mode
#             )
#         )
        
#     def get_loss(self, x, tissue_labels, kl_weight=1.0):
#         """
#         Args:
#             x:  Input point clouds, (B, N, d).
#         """
#         z_mu, z_sigma = self.encoder(x)
#         z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
#         loss_prior = torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mu ** 2 - z_sigma.exp(), dim = 1), dim = 0)
        
#         # Get tissue embeddings and concatenate with z
#         tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
#         z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)

#         # loss_recons = self.diffusion.get_loss(x, z)
#         loss_recons = self.diffusion.get_loss(x, z_with_tissue)

#         loss = kl_weight * loss_prior + loss_recons
        
#         print("loss_prior:", loss_prior.item(), "loss_recons:", loss_recons.item())

#         return loss

#     def sample(self, z, tissue_labels, num_points, flexibility=0.0, truncate_std=None):
#         """
#         Args:
#             z:  Input latent, normal random samples with mean=0 std=1, (B, F)
#         """
#         if truncate_std is not None:
#             z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
            
#         # Get tissue embeddings and concatenate with z
#         tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
#         z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)
        
#         # samples = self.diffusion.sample(num_points, context=z, point_dim=self.args.input_dim, flexibility=flexibility)
#         samples = self.diffusion.sample(num_points, context=z_with_tissue, point_dim=self.args.input_dim, flexibility=flexibility)
#         # samples_traj = self.diffusion.sample(num_points, context=z_with_tissue, point_dim=self.args.input_dim, flexibility=flexibility, ret_traj=True)
#         # samples = samples_traj[0]
        
#         # print("in sample x min:", samples[:,0])
        
#         # Apply min-max normalization to the x and y dimensions separately
#         x_min = samples[:, :, 0].min()
#         x_max = samples[:, :, 0].max()
#         y_min = samples[:, :, 1].min()
#         y_max = samples[:, :, 1].max()
#         # Normalize the x dimension
#         samples[:, :, 0] = (samples[:, :, 0] - x_min) / (x_max - x_min + 1e-8)
#         # Normalize the y dimension
#         samples[:, :, 1] = (samples[:, :, 1] - y_min) / (y_max - y_min + 1e-8)
        
#         return samples
#         # return samples, samples_traj
    
#     def map_position_back(self, predicted_positions, test_area):
#         predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
#         predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y
#         return predicted_positions
