import os
import re
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import truncnorm
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F

def seed_everything(seed=2024):
    random.seed(seed)    # Python random module
    np.random.seed(seed) # Numpy module
    os.environ['PYTHONHASHSEED'] = str(seed) # Env variable
    torch.manual_seed(seed)  # Torch
    torch.cuda.manual_seed(seed)  # CUDA
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Seeding all randomness with seed={seed}')

def generate_edges(positions, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    edge_index = []

    for i in range(indices.shape[0]):
        for j in range(1, k+1):
            edge_index.append((i, indices[i, j]))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def select_central_cells(predicted_positions, expansion_factor=1):
    """
    Select a certain fraction of cells around the center of predicted_positions based on distances.
    Also return a central_mask to indicate which cells are selected.

    Args:
        predicted_positions: (N, d) predicted coordinates for a single batch.
        expansion_factor: int, expansion factor to define fraction of cells to pick.
    
    Returns:
        central_positions: (M, d) subset of positions considered 'central'.
        central_mask: (N,) boolean mask indicating which cells are selected as central.
    """
    N = predicted_positions.size(0)
    # Compute the center (mean) of predicted positions
    center = predicted_positions.mean(dim=0, keepdim=True)  # (1, d)
    
    # Compute distances of each cell to the center
    distances = torch.norm(predicted_positions - center, dim=1)  # (N,)
    
    # fraction to select: 1 / ((2*expansion_factor + 1)^2)
    fraction = 1.0 / ((2 * expansion_factor + 1) ** 2)
    num_to_select = max(int(fraction * N), 1)  # at least select 1 cell
    
    # Sort by distance and select top fraction
    sorted_indices = torch.argsort(distances)  # ascending order
    central_indices = sorted_indices[:num_to_select]

    central_positions = predicted_positions[central_indices]

    central_mask = torch.zeros(N, dtype=torch.bool, device=predicted_positions.device)
    central_mask[central_indices] = True

    return central_positions, central_mask
    
def extract_cells_from_expanded_region(test_item, expansion_factor=1):
    """
    Extract positions and gene expressions from the expanded region in test_item.adata.
    
    Args:
        test_item: The test item containing adata and test_area information.
        expansion_factor: Factor by which to expand the region for extraction.
        
    Returns:
        expanded_positions: Tensor of positions in the expanded region (N, 2).
        expanded_gene_expressions: Tensor of gene expressions in the expanded region (N, G).
    """
    # Calculate expanded region bounds
    min_x = test_item.test_area.hole_min_x - (test_item.test_area.hole_max_x - test_item.test_area.hole_min_x) * expansion_factor
    max_x = test_item.test_area.hole_max_x + (test_item.test_area.hole_max_x - test_item.test_area.hole_min_x) * expansion_factor
    min_y = test_item.test_area.hole_min_y - (test_item.test_area.hole_max_y - test_item.test_area.hole_min_y) * expansion_factor
    max_y = test_item.test_area.hole_max_y + (test_item.test_area.hole_max_y - test_item.test_area.hole_min_y) * expansion_factor

    # Extract cells within this expanded region from test_item.adata
    adata = test_item.adata
    cells_in_expanded_region = adata[(adata.obs['center_x'] >= min_x) & (adata.obs['center_x'] <= max_x) &
                                     (adata.obs['center_y'] >= min_y) & (adata.obs['center_y'] <= max_y)]
    
    # Extract positions and gene expressions
    expanded_positions = cells_in_expanded_region.obs[['center_x', 'center_y']].values
    expanded_gene_expressions = cells_in_expanded_region.X  # Assuming the gene expressions are stored in adata.X
    
    # Convert to tensors
    expanded_positions = torch.tensor(expanded_positions, dtype=torch.float32)
    expanded_gene_expressions = torch.tensor(expanded_gene_expressions.toarray(), dtype=torch.float32)  # Convert sparse matrix to dense if necessary
    
    return expanded_positions, expanded_gene_expressions

def normalize_positions(positions):
    """
    Normalize the positions along x and y axis separately to the range [-1, 1].
    Args:
        positions: (N, 2) The positions to normalize.
    Returns:
        normalized_positions: (N, 2) The normalized positions.
    """
    normalized_positions = positions.clone()
    normalized_positions[:, 0] = 2 * (positions[:, 0] - positions[:, 0].min()) / (positions[:, 0].max() - positions[:, 0].min()) - 1
    normalized_positions[:, 1] = 2 * (positions[:, 1] - positions[:, 1].min()) / (positions[:, 1].max() - positions[:, 1].min()) - 1
    return normalized_positions
    
def normalize_positions_within_test_area(positions, test_area):
    """
    Normalize the positions to the range [-1, 1].
    Args:
        positions: (N, pos_dim) The positions to normalize.
        test_area: An object containing the original coordinate ranges.
    Returns:
        normalized_positions: (N, pos_dim) The normalized positions.
    """
    normalized_positions = positions.clone()
    normalized_positions[:, 0] = 2 * (positions[:, 0] - test_area.hole_min_x) / (test_area.hole_max_x - test_area.hole_min_x) - 1
    normalized_positions[:, 1] = 2 * (positions[:, 1] - test_area.hole_min_y) / (test_area.hole_max_y - test_area.hole_min_y) - 1
    return normalized_positions

def normalize_positions_within_expanded_test_area(positions, test_area, expansion_factor=1):
    """
    Normalize the positions to the range [-1, 1].
    Args:
        positions: (N, pos_dim) The positions to normalize.
        test_area: An object containing the original coordinate ranges.
    Returns:
        normalized_positions: (N, pos_dim) The normalized positions.
    """
    size_x = test_area.hole_max_x - test_area.hole_min_x
    size_y = test_area.hole_max_y - test_area.hole_min_y
    min_x = test_area.hole_min_x - expansion_factor * size_x
    max_x = test_area.hole_max_x + expansion_factor * size_x
    min_y = test_area.hole_min_y - expansion_factor * size_y
    max_y = test_area.hole_max_y + expansion_factor * size_y
    normalized_positions = positions.clone()
    normalized_positions[:, 0] = 2 * (positions[:, 0] - min_x) / (max_x - min_x) - 1
    normalized_positions[:, 1] = 2 * (positions[:, 1] - min_y) / (max_y - min_y) - 1
    return normalized_positions

def map_position_back(predicted_positions, test_area, expansion_factor=None):
    """
    Map predicted positions back to the original coordinate range.
    Args:
        predicted_positions: (B, N, pos_dim) Predicted positions in normalized space
        test_area: An object containing the original coordinate ranges
        expansion_factor: Scaling factor that affects the normalized position range
    Returns:
        predicted_positions: (B, N, pos_dim) Predicted positions in original space
    """
    if expansion_factor is not None:
        # Calculate the normalization range
        norm_min = -1 / (2 * expansion_factor + 1)
        norm_max = 1 / (2 * expansion_factor + 1)

        # Rescale predicted_positions from [norm_min, norm_max] to [0, 1]
        predicted_positions = (predicted_positions - norm_min) / (norm_max - norm_min)

    else:
        # Default normalization from [-1, 1] to [0, 1]
        predicted_positions = (predicted_positions + 1) / 2.0

    # Map to the original test_area coordinate range
    predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
    predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y

    return predicted_positions
    
# Define Chamfer Loss for unordered data
def chamfer_loss(targets, predictions):
    pred_to_target_dist = torch.cdist(predictions, targets, p=2)
    target_to_pred_dist = torch.cdist(targets, predictions, p=2)
    loss = torch.mean(pred_to_target_dist.min(dim=1)[0]) + torch.mean(target_to_pred_dist.min(dim=1)[0])
    return loss

def compute_unordered_gene_expression_loss(true_positions, true_expressions, pred_positions, pred_expressions):
    """
    Compute the gene expression loss using nearest neighbor alignment and MSE.

    Args:
    - true_positions (torch.Tensor): Ground truth cell coordinates, shape [batch_size, num_cells, 2].
    - true_expressions (torch.Tensor): Ground truth gene expression values, shape [batch_size, num_cells, num_genes].
    - pred_positions (torch.Tensor): Predicted cell coordinates, shape [batch_size, num_cells, 2].
    - pred_expressions (torch.Tensor): Predicted gene expression values, shape [batch_size, num_cells, num_genes].

    Returns:
    - expression_loss (torch.Tensor): Mean squared error (MSE) loss for gene expression alignment.
    """
    # Compute pairwise Euclidean distance between predicted and true positions
    dist_matrix = torch.cdist(pred_positions, true_positions, p=2)
    
    # Find the indices of the closest true positions for each predicted position
    closest_indices = dist_matrix.argmin(dim=2)
    
    # Gather the gene expression values of the closest true neighbors
    closest_true_expressions = true_expressions.gather(
        1, closest_indices.unsqueeze(-1).expand(-1, -1, true_expressions.size(2))
    )
    
    # Compute MSE between predicted and closest true gene expressions
    expression_loss = F.mse_loss(pred_expressions, closest_true_expressions, reduction='mean')
    
    return expression_loss


def visualize_test_region(slice_obs_df, test_area, title='', new_coords=None):
    slice_obs_df['fov'] = slice_obs_df['fov'].astype(int)
    for field in ['min_x', 'max_x', 'min_y', 'max_y', 'center_x', 'center_y']:
        slice_obs_df[field] = slice_obs_df[field].astype(float)

    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=slice_obs_df['fov'].min(), vmax=slice_obs_df['fov'].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    ax1 = axs[0]
    fov_boundaries = slice_obs_df.groupby('fov').agg(
        min_x=('min_x', 'min'),
        max_x=('max_x', 'max'),
        min_y=('min_y', 'min'),
        max_y=('max_y', 'max')
    ).reset_index()
    for _, bounds in fov_boundaries.iterrows():
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        color = sm.to_rgba(bounds['fov'])
        rect = patches.Rectangle(
            (bounds['min_x'], bounds['min_y']),
            width, height,
            linewidth=1, edgecolor='r', facecolor=color, alpha=0.7
        )
        ax1.add_patch(rect)
    ax1.set_title('FOV Boundaries')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    plt.colorbar(sm, ax=ax1, label='FOV')

    ax2 = axs[1]
    tissue_colors = {
        'corpus callosum': 'red',
        'pia mater': 'green',
        'striatum': 'blue',
        'olfactory region': 'cyan',
        'brain ventricle': 'magenta',
        'cortical layer V': 'yellow',
        'cortical layer VI': 'orange',
        'cortical layer II/III': 'purple'
    }
    for tissue, group in slice_obs_df.groupby('tissue'):
        ax2.scatter(group['center_x'], group['center_y'], color=tissue_colors.get(tissue, 'gray'), label=tissue, alpha=0.7, edgecolors='none', s=10)
    ax2.set_title('Spot Visualization by Tissue')
    ax2.set_xlabel('Center X')
    ax2.set_ylabel('Center Y')
    ax2.legend()

    rect = patches.Rectangle(
        (test_area.hole_min_x, test_area.hole_min_y),
        test_area.hole_max_x - test_area.hole_min_x,
        test_area.hole_max_y - test_area.hole_min_y,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax2.add_patch(rect)

    if new_coords is not None:
        ax2.scatter(new_coords[:, 0], new_coords[:, 1], color='k', label='Generated Coords', alpha=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def filter_training_sample_by_tissue(training_samples, tissue_name):
    filtered_samples = [sample for sample in training_samples if sample['metadata']['dominant_tissue'] == tissue_name]
    return filtered_samples

def count_dominant_tissue(data):
    tissue_count = {}
    
    for sample in data:
        tissue = sample['metadata']['dominant_tissue']
        if tissue in tissue_count:
            tissue_count[tissue] += 1
        else:
            tissue_count[tissue] = 1
            
    return tissue_count

def plot_tissue_distribution(tissue_count):
    tissues = list(tissue_count.keys())
    counts = list(tissue_count.values())
    
    plt.figure(figsize=(5, 3))
    plt.bar(tissues, counts, color='skyblue')
    plt.xlabel('Tissue Type')
    plt.ylabel('Count')
    plt.title('Distribution of Dominant Tissue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_coords(true_coords, pred_coords, title="Coordinate Distribution Comparison"):
    """
    Visualizes the distribution of true and generated coordinates.
    
    :param true_coords: numpy array of true coordinates, shape (n_samples, 2)
    :param pred_coords: numpy array of generated coordinates, shape (n_samples, 2)
    :param title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot true coordinates
    plt.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', marker='o', label='True Coords', alpha=0.5)
    
    # Plot VAE-generated coordinates
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', marker='x', label='Generated Coords', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to sanitize directory and file names
def sanitize_name(name):
    # Replace spaces and special characters like slashes with underscores
    return re.sub(r'[^\w\-\.]', '_', name)

def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Generate truncated normal distribution values.
    Args:
        tensor: Input tensor to be modified in-place.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        trunc_std: The number of standard deviations to truncate at.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def truncated_normal_2d(size, mean=0, std=1, trunc_std=2):
    """Generate 2D truncated normal distribution data within [-trunc_std*std + mean, trunc_std*std + mean]."""
    x = torch.empty(size)
    y = torch.empty(size)
    truncated_normal_(x, mean=mean, std=std, trunc_std=trunc_std)
    truncated_normal_(y, mean=mean, std=std, trunc_std=trunc_std)
    return np.column_stack((x.numpy(), y.numpy()))

def visualize_noising_process_animation(model, initial_positions):
    """
    Create an animation visualizing the noising process by showing positions at different steps.
    Args:
        model: The GaussianVAE model containing the necessary parameters.
        initial_positions: Initial point cloud positions (N, 2).
    """
    num_points, point_dim = initial_positions.shape
    positions_list = [initial_positions]

    # Retrieve parameters from the model
    num_steps = model.args.num_steps
    beta_1 = model.args.beta_1
    beta_T = model.args.beta_T

    # Create a linear schedule of beta values
    betas = np.linspace(beta_1, beta_T, num_steps)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    # Go through the diffusion steps and add noise
    for t in range(1, num_steps + 1):
        alpha_bar = alpha_bars[t-1]
        noise = truncated_normal_2d(size=num_points, mean=0, std=1, trunc_std=2)
        c0 = np.sqrt(alpha_bar)
        c1 = np.sqrt(1 - alpha_bar)
        noised_positions = c0 * initial_positions + c1 * noise
        positions_list.append(noised_positions)

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    scat = ax.scatter([], [], c='red', s=5)

    def init():
        scat.set_offsets(np.zeros((num_points, 2)))
        return scat,

    def update(frame):
        positions = positions_list[frame]
        scat.set_offsets(positions)
        ax.set_title(f"Step {frame}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(positions_list),
                                  init_func=init, blit=True, repeat=False)
    
    return ani

# def visualize_sampling_process_animation(model, num_points=500, point_dim=2, flexibility=0.0):
#     """
#     Create an animation visualizing the sampling process by showing positions at different steps.
#     Args:
#         model: The diffusion model containing the necessary parameters.
#         num_points: Number of points to sample per cloud.
#         context: The context tensor used for sampling.
#         point_dim: Dimension of the points (default=2).
#         flexibility: Flexibility parameter for sigma.
#     """
#     # Get the trajectory of sampled positions
#     z = torch.randn(1, model.args.latent_dim).to(model.args.device)
#     tissue_embed = model.tissue_embedding(torch.tensor([0], device=model.args.device))  # (B, tissue_dim)
#     z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)
#     traj = model.position_diffusion.sample(num_points=500, context=z_with_tissue, point_dim=point_dim, flexibility=flexibility, ret_traj=True)
    
#     # Convert trajectory dictionary to a list of positions
#     positions_list = [traj[t].cpu().numpy().squeeze(0) for t in sorted(traj.keys())]
        
#     normalized_positions_list = positions_list

#     # Create animation
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     scat = ax.scatter([], [], c='blue', s=5)

#     def init():
#         scat.set_offsets(np.zeros((num_points, point_dim)))
#         return scat,

#     def update(frame):
#         positions = normalized_positions_list[frame]
#         scat.set_offsets(positions)
#         ax.set_title(f"Step {frame + 1}")
#         return scat,

#     ani = animation.FuncAnimation(fig, update, frames=len(normalized_positions_list),
#                                   init_func=init, blit=True, repeat=False)
    
#     return ani

def visualize_sampling_process_animation(model, num_points=500, point_dim=2, flexibility=0.0, expansion_factor=1, test_item_list=None, mode="position"):
    """
    Create an animation visualizing the sampling process by showing positions at different steps.
    Args:
        model: The diffusion model containing the necessary parameters.
        num_points: Number of points to sample per cloud.
        context: The context tensor used for sampling.
        point_dim: Dimension of the points (default=2).
        flexibility: Flexibility parameter for sigma.
    """
    # Get the trajectory of sampled positions
    z = torch.randn(1, model.args.latent_dim).to(model.args.device)
    tissue_embed = model.tissue_embedding(torch.tensor([0], device=model.args.device))  # (B, tissue_dim)
    z_with_tissue = torch.cat([z, tissue_embed], dim=-1)  # (B, F + tissue_dim)
    traj = model.position_diffusion.sample(num_points=num_points, 
                                           context=z_with_tissue, 
                                           point_dim=point_dim, 
                                           flexibility=flexibility, 
                                           ret_traj=True, 
                                           expansion_factor=expansion_factor, 
                                           test_item_list=test_item_list, 
                                           mode=mode)
    
    # Convert trajectory dictionary to a list of positions
    positions_list = [traj[t].cpu().numpy().squeeze(0) for t in sorted(traj.keys())]
        
    normalized_positions_list = positions_list

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    scat = ax.scatter([], [], c='blue', s=5)

    def init():
        scat.set_offsets(np.zeros((num_points, point_dim)))
        return scat,

    def update(frame):
        positions = normalized_positions_list[frame]
        scat.set_offsets(positions)
        ax.set_title(f"Step {frame + 1}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(normalized_positions_list),
                                  init_func=init, blit=True, repeat=False)
    
    return ani
