import torch
from torch.nn import Module

from .common import *
from .diffusion import *
from .encoder import *

class GaussianVAE(Module):
    """
    A Variational Autoencoder (VAE) with Gaussian latent variables, 
    tailored for spatial transcriptomics data. It includes diffusion models 
    for both position and gene expression.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Encoder for integrating positional and expression data using a GCN.
        self.region_encoder = GNNEncoder(
            input_dim=args.expression_dim,  # Input dimension = expr_dim
            hidden_dim=args.latent_dim,  # Hidden layer dimension
            zdim=args.latent_dim  # Latent dimension
        )
        
        # Encoding positions before passing them into the diffusion model.
        self.position_encoder = SineCosinePositionalEncoding(
            zdim=args.encoded_position_dim  # Set the dimension of the encoding
        )
        
        # Embedding for tissue types
        self.tissue_embedding = torch.nn.Embedding(args.num_tissues, args.tissue_dim)  # (B, tissue_dim)

        # Diffusion model for generating or reconstructing positions
        self.position_diffusion = DiffusionModel(
            net=GNNDenoiseNet(
                input_dim=args.position_dim, 
                latent_dim=args.latent_dim,
                context_dim=args.latent_dim + args.tissue_dim  # Context dimension = latent_dim + tissue_dim
            ),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

        # Diffusion model for generating or reconstructing gene expressions
        self.expression_diffusion = DiffusionModel(
            net=GNNDenoiseNet(
                input_dim=args.expression_dim,
                latent_dim=args.latent_dim,
                context_dim=args.latent_dim + args.tissue_dim + args.encoded_position_dim  # Context dimension = latent_dim + tissue_dim + encoded_pos_dim
            ),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
    
    def encode_all(self, expressions, edge_index):
        """
        Encode both positions (edges) and expressions (node features) using a GNN.
        Args:
            positions: (B, N, pos_dim)
            expressions: (B, N, expr_dim)
            edge_index: (B, 2, E) - E is the number of edges
        Returns:
            z_mu: Mean of the latent variables (B, latent_dim)
            z_sigma: Log variance of the latent variables (B, latent_dim)
        """
        B, N, _ = expressions.size()
        # Flatten batch dimension for processing with GCN.
        expressions_batch = torch.cat([expressions[i] for i in range(B)], dim=0)  # (B * N, expr_dim)
        # Adjust edge_index for each graph in the batch.
        edge_index_batch = torch.cat([edge_index[i].to(torch.long) + i * N for i in range(B)], dim=1)  # (2, E * B)
        # Create batch vector for pooling.
        batch = torch.tensor([i for i in range(B) for _ in range(N)]).to(expressions.device)  # (B * N,)
        z_mu, z_sigma = self.region_encoder(expressions_batch, edge_index_batch, batch)
        return z_mu, z_sigma
    
    def get_position_loss(self, positions, expressions, edge_index, tissue_labels, kl_weight=1.0):
        """
        Compute the loss for position prediction using the diffusion model.
        Args:
            positions: (B, N, pos_dim)
            expressions: (B, N, expr_dim)
            edge_index: (B, 2, E)
            tissue_labels: (B,)
            kl_weight: Weight for KL divergence term
        Returns:
            position_loss: The computed loss value
        """
        z_mu, z_sigma = self.encode_all(expressions, edge_index)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, latent_dim)
        loss_prior = torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mu ** 2 - z_sigma.exp(), dim=1), dim=0)
        # Get tissue embeddings and concatenate with latent variables.
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, latent_dim + tissue_dim)
        # Compute reconstruction loss using the diffusion model.
        loss_recons = self.position_diffusion.get_loss(positions, z_with_tissue)
        position_loss = kl_weight * loss_prior + loss_recons
        return position_loss

    # def get_expression_loss(self, predicted_positions, real_positions, real_expressions, edge_index, tissue_labels):
    def get_expression_loss(self, real_positions, real_expressions, edge_index, tissue_labels):
        """
        Compute loss for gene expression prediction using the nearest real positions.
        Args:
            predicted_positions: (B, N, pos_dim)
            real_positions: (B, N, pos_dim)
            real_expressions: (B, N, expr_dim)
            edge_index: (B, 2, E)
            tissue_labels: (B,)
        Returns:
            loss_recons: The computed reconstruction loss value
        """
        z_mu, z_sigma = self.encode_all(real_expressions, edge_index)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, latent_dim)
        z_expand = z.unsqueeze(1).expand(-1, real_positions.size(1), -1)  # (B, N, latent_dim)
        # Get tissue embeddings and concatenate with predicted positions.
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        tissue_embed_expand = tissue_embed.unsqueeze(1).expand(-1, real_positions.size(1), -1)  # (B, N, tissue_dim)
        encoded_positions = self.position_encoder(real_positions)  # (B, N, encoded_pos_dim)
        context = torch.cat([z_expand, tissue_embed_expand, encoded_positions], dim=-1)  # (B, N, latent_dim + tissue_dim + encoded_pos_dim)
        # Find nearest real positions for each predicted position.
        
        # print("train gene: predicted_positions:", predicted_positions.min(), predicted_positions.max(), predicted_positions.shape,
        #                     "real_positions:", real_positions.min(), real_positions.max(), real_positions.shape)
        
        # nearest_expressions = self.get_nearest_expressions(predicted_positions, real_positions, real_expressions)
        nearest_expressions = real_expressions
        # loss_recons = self.expression_diffusion.get_loss(nearest_expressions, context, mode="expression", position=predicted_positions, k=5, batch_context=False)
        loss_recons = self.expression_diffusion.get_loss(nearest_expressions, context, mode="expression", position=real_positions, k=5, batch_context=False)
        
        return loss_recons

    def get_nearest_expressions(self, predicted_positions, real_positions, real_expressions):
        """
        Find the nearest real positions and return their gene expressions.
        Args:
            predicted_positions: (B, N, pos_dim)
            real_positions: (B, N, pos_dim)
            real_expressions: (B, N, expr_dim)
        Returns:
            nearest_expressions: (B, N, expr_dim) Nearest gene expressions for predicted positions.
        """
        G = real_expressions.shape[2]  # Gene expression dimension
        # Compute pairwise distances between predicted and real positions.
        distances = torch.cdist(predicted_positions, real_positions)  # (B, N, N)
        # Find the nearest indices in the real positions.
        nearest_indices = distances.argmin(dim=-1)  # (B, N)
        # Gather the nearest gene expressions.
        nearest_expressions = torch.gather(
            real_expressions,  # (B, N, G)
            dim=1,
            index=nearest_indices.unsqueeze(-1).expand(-1, -1, G)  # (B, N, G)
        )
        return nearest_expressions
    
    def sample_positions(self, z, tissue_labels, num_points, flexibility=0.0, truncate_std=None, expansion_factor=1, test_item_list=None, mode="position"):
        """
        Sample positions using the position diffusion model.
        Args:
            z: (B, latent_dim) Input latent variables, normal random samples with mean=0 std=1
            tissue_labels: (B,) Tensor of tissue labels
            num_points: Number of points to sample per cloud
        Returns:
            samples: (B, N, pos_dim) Sampled positions
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)

        # Get tissue embeddings and concatenate with latent variables.
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        context = torch.cat([z, tissue_embed], dim=-1)  # (B, latent_dim + tissue_dim)

        # Sample positions using diffusion model
        samples = self.position_diffusion.sample(num_points, context=context, 
                                                point_dim=self.args.position_dim, 
                                                flexibility=flexibility, 
                                                expansion_factor=expansion_factor, 
                                                test_item_list=test_item_list, 
                                                mode=mode)

        # If we are using the neighborhood information (test_item_list is not None)
        if test_item_list is not None:
            # central_region_min = -1 / (2 * expansion_factor + 1)
            # central_region_max = 1 / (2 * expansion_factor + 1)
            selected_positions = []

            for i, test_item in enumerate(test_item_list):
                # Normalize the predicted positions
                samples_normalized = normalize_positions(samples[i])
                
                central_positions, _ = select_central_cells(samples_normalized, expansion_factor=expansion_factor)

                # Identify the cells in the central region
                # central_mask = (samples_normalized[:, 0] > central_region_min) & (samples_normalized[:, 0] < central_region_max) & \
                #             (samples_normalized[:, 1] > central_region_min) & (samples_normalized[:, 1] < central_region_max)

                # central_positions = samples[i][central_mask]  # Get positions inside the central region

                # # If more than 50 positions are in the central region, randomly sample 50 positions
                # if central_positions.shape[0] > 50:
                #     selected_indices = torch.randperm(central_positions.shape[0])[:50]  # Randomly select 50 indices
                #     central_positions = central_positions[selected_indices]
                # # If fewer than 50 positions are in the central region, repeat selection until 50 positions
                # elif central_positions.shape[0] < 50:
                #     num_to_add = 50 - central_positions.shape[0]  # Number of additional positions needed
                #     if central_positions.shape[0] > 0:
                #         additional_indices = torch.randint(0, central_positions.shape[0], (num_to_add,))
                #         additional_positions = central_positions[additional_indices]
                #         central_positions = torch.cat([central_positions, additional_positions], dim=0)
                #     else:
                #         # Handle edge case where no positions exist in the central region
                #         raise ValueError("No positions found in the central region for test_item {}".format(i))

                selected_positions.append(normalize_positions(central_positions))

            # Convert list of selected positions back to a tensor
            selected_positions = torch.stack(selected_positions, dim=0)

            return selected_positions

        # If not using neighborhood information, normalize the output as usual
        else:
            # Min-max normalization
            min_val = samples.min(dim=1, keepdim=True)[0]
            max_val = samples.max(dim=1, keepdim=True)[0]
            samples = (samples - min_val) / (max_val - min_val + 1e-7)  # Adding epsilon to avoid division by zero
            # Scaling to range [-1, 1]
            samples = samples * 2 - 1
            return samples
    
    def sample_expressions(self, z, tissue_labels, predicted_positions, flexibility=0.0, truncate_std=None, expansion_factor=1, test_item_list=None, mode="expression"):
        """
        Sample gene expressions using the expression diffusion model.
        Args:
            z: (B, latent_dim) Input latent variables, normal random samples with mean=0 std=1
            predicted_positions: (B, N, pos_dim) Predicted point clouds
            tissue_labels: (B,) Tensor of tissue labels
            Returns:
            gene_expressions: (B, N, expr_dim) Sampled gene expressions
        """
        num_points = predicted_positions.shape[1]

        # Truncate latent variables if specified
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)

        # Expand latent and tissue embeddings
        z_expand = z.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, latent_dim)
        tissue_embed = self.tissue_embedding(tissue_labels)  # (B, tissue_dim)
        tissue_embed_expand = tissue_embed.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, tissue_dim)
        encoded_positions = self.position_encoder(predicted_positions)  # (B, N, encoded_pos_dim)

        # Concatenate latent, tissue, and position embeddings to form the context
        context = torch.cat([z_expand, tissue_embed_expand, encoded_positions], dim=-1)  # (B, N, latent_dim + tissue_dim + encoded_pos_dim)

        # Sample expressions using the diffusion model
        gene_expressions = self.expression_diffusion.sample(
            num_points=num_points,
            context=context,
            point_dim=self.args.expression_dim,
            flexibility=flexibility,
            expansion_factor=expansion_factor,
            test_item_list=test_item_list,
            mode=mode,
            position=predicted_positions  # Pass positions explicitly
        )
        
        # print("s:", gene_expressions.shape)

        # # If in test mode and neighborhood information is provided
        # if test_item_list is not None:
        #     # central_region_min = -1 / (2 * expansion_factor + 1)
        #     # central_region_max = 1 / (2 * expansion_factor + 1)
        #     selected_expressions = []

        #     for i, test_item in enumerate(test_item_list):
        #         # Normalize the predicted positions
        #         positions_normalized = normalize_positions(predicted_positions[i])
                
        #         _, central_mask = self.select_central_cells(positions_normalized, expansion_factor=expansion_factor)

        #         # # Identify the cells in the central region
        #         # central_mask = (positions_normalized[:, 0] > central_region_min) & (positions_normalized[:, 0] < central_region_max) & \
        #         #             (positions_normalized[:, 1] > central_region_min) & (positions_normalized[:, 1] < central_region_max)

        #         central_expressions = gene_expressions[i][central_mask]  # Get expressions of cells in the central region

        #         # # If more than 50 cells in the central region, randomly select 50
        #         # if central_expressions.shape[0] > 50:
        #         #     selected_indices = torch.randperm(central_expressions.shape[0])[:50]
        #         #     central_expressions = central_expressions[selected_indices]
        #         # # If fewer than 50 cells in the central region, repeat selection until 50
        #         # elif central_expressions.shape[0] < 50:
        #         #     num_to_add = 50 - central_expressions.shape[0]
        #         #     if central_expressions.shape[0] > 0:
        #         #         additional_indices = torch.randint(0, central_expressions.shape[0], (num_to_add,))
        #         #         additional_expressions = central_expressions[additional_indices]
        #         #         central_expressions = torch.cat([central_expressions, additional_expressions], dim=0)
        #         #     else:
        #         #         # Handle edge case where no cells exist in the central region
        #         #         raise ValueError("No cells found in the central region for test_item {}".format(i))

        #         selected_expressions.append(central_expressions)

        #     # Convert list of selected expressions back to a tensor
        #     selected_expressions = torch.stack(selected_expressions, dim=0)

        #     return selected_expressions

        # # If in train mode, return the full generated expressions
        # else:
        return gene_expressions

    def sample(self, tissue_labels, num_points, flexibility=0.0):
        """
        High-level sample function to generate both positions and gene expressions.
        Args:
            tissue_labels: (B,) Tensor of tissue labels
            num_points: Number of points to sample per cloud
        Returns:
            predicted_positions: (B, N, pos_dim) Sampled positions
            predicted_expressions: (B, N, expr_dim) Sampled gene expressions
        """
        # Step 1: Generate random latent variables z
        batch_size = tissue_labels.size(0)
        device = tissue_labels.device
        z = torch.randn(batch_size, self.args.latent_dim).to(device)
        # Step 2: Sample positions
        predicted_positions = self.sample_positions(z, tissue_labels, num_points=num_points, flexibility=flexibility)
        # Step 3: Sample gene expressions based on predicted positions
        predicted_expressions = self.sample_expressions(z, tissue_labels, predicted_positions, flexibility=flexibility)
        return predicted_positions, predicted_expressions
    
