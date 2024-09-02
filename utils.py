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

# Define Chamfer Loss for unordered data
def chamfer_loss(predictions, targets):
    pred_to_target_dist = torch.cdist(predictions, targets, p=2)
    target_to_pred_dist = torch.cdist(targets, predictions, p=2)
    loss = torch.mean(pred_to_target_dist.min(dim=1)[0]) + torch.mean(target_to_pred_dist.min(dim=1)[0])
    return loss

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

def visualize_sampling_process_animation(model, num_points=500, point_dim=2, flexibility=0.0):
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
    traj = model.position_diffusion.sample(num_points=500, context=z_with_tissue, point_dim=point_dim, flexibility=flexibility, ret_traj=True)
    
    # Convert trajectory dictionary to a list of positions
    positions_list = [traj[t].cpu().numpy().squeeze(0) for t in sorted(traj.keys())]
    
    # # 对 positions_list 中的每个 positions 进行归一化处理
    # normalized_positions_list = []
    # for positions in positions_list:
    #     positions_tensor = torch.tensor(positions)
    #     # Min-max normalization
    #     min_val = positions_tensor.min(dim=0, keepdim=True)[0]
    #     max_val = positions_tensor.max(dim=0, keepdim=True)[0]
    #     positions_normalized = (positions_tensor - min_val) / (max_val - min_val + 1e-7)  # Adding epsilon to avoid division by zero
    #     # Scaling to range [-1, 1]
    #     positions_normalized = 2*positions_normalized-1
    #     normalized_positions_list.append(positions_normalized.numpy())
        
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