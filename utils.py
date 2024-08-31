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

<<<<<<< Updated upstream
# Define Chamfer Loss for unordered data
def chamfer_loss(predictions, targets):
    pred_to_target_dist = torch.cdist(predictions, targets, p=2)
    target_to_pred_dist = torch.cdist(targets, predictions, p=2)
    loss = torch.mean(pred_to_target_dist.min(dim=1)[0]) + torch.mean(target_to_pred_dist.min(dim=1)[0])
    return loss
=======
class GroundTruth:
    def __init__(self, hole_cells, gene_expression, tissue_percentages):
        self.hole_cells = hole_cells
        self.gene_expression = gene_expression
        self.tissue_percentages = tissue_percentages

class TestArea:
    def __init__(self, hole_min_x, hole_max_x, hole_min_y, hole_max_y, dominant_tissue):
        self.hole_min_x = hole_min_x
        self.hole_max_x = hole_max_x
        self.hole_min_y = hole_min_y
        self.hole_max_y = hole_max_y
        self.dominant_tissue = dominant_tissue

class TestItem:
    def __init__(self, adata, ground_truth, test_area, meta_data):
        self.adata = adata
        self.ground_truth = ground_truth
        self.test_area = test_area
        self.meta_data = meta_data

def load_test_data(platform: str = "MERFISH", dataset: str = "MouseBrainAging", \
                hole_size: int = 250, num_holes: int = 2, dominance_threshold: float = 0.9, num_cells: int = 50):
    
    seed_everything()
    
    fold_dir = "C:\\Docs_and_Stuff\\Emory\\Lab\\DISCO\\DISCO"
    file_name = "9545288f-ad53-4fea-bc3d-a356cd2786a6.h5ad"
    
    if platform == "MERFISH":
        if dataset == "MouseBrainAging":
            adata = ad.read_h5ad(fold_dir + "\\" + platform + "\\" + file_name)
            obs = adata.obs
            for field in ['min_x', 'max_x', 'min_y', 'max_y', 'center_x', 'center_y']:
                obs[field] = obs[field].astype(float)
            donor_id_list = list(obs['donor_id'].unique())
            all_test_items = []

            for donor_id in donor_id_list:
                print(f'Donor_id: {donor_id}')
                donor_obs = obs[obs['donor_id'] == donor_id]
                donor_x = adata[obs['donor_id'] == donor_id]
                slice_list = list(donor_obs['slice'].unique())
                slice_list.sort()
                for slice_id in slice_list: 
                    print(f'Slice_id: {slice_id}')
                    slice_obs = donor_obs[donor_obs['slice'] == slice_id]
                    slice_x = donor_x[donor_obs['slice'] == slice_id]
                    
                    slice_obs_df = pd.DataFrame(slice_obs)
                    slice_obs_df['fov'] = slice_obs_df['fov'].cat.remove_unused_categories()
                    
                    fov_boundaries = slice_obs_df.groupby('fov').agg(
                        min_x=('min_x', 'min'),
                        max_x=('max_x', 'max'),
                        min_y=('min_y', 'min'),
                        max_y=('max_y', 'max')
                    ).reset_index()

                    fov_boundaries['center_x'] = (fov_boundaries['min_x'] + fov_boundaries['max_x']) / 2
                    fov_boundaries['center_y'] = (fov_boundaries['min_y'] + fov_boundaries['max_y']) / 2

                    holes_found = 0
                    while holes_found < num_holes:
                        fov = fov_boundaries.sample(1).iloc[0]
                        if fov['center_x'] < 0:
                            rand_center_x = fov['center_x'] + random.uniform(0, hole_size / 2)
                        else:
                            rand_center_x = fov['center_x'] - random.uniform(0, hole_size / 2)
                        if fov['center_y'] < 0:
                            rand_center_y = fov['center_y'] + random.uniform(0, hole_size / 2)
                        else:
                            rand_center_y = fov['center_y'] - random.uniform(0, hole_size / 2)

                        hole_min_x = max(rand_center_x - hole_size / 2, fov['min_x'])
                        hole_max_x = min(rand_center_x + hole_size / 2, fov['max_x'])
                        hole_min_y = max(rand_center_y - hole_size / 2, fov['min_y'])
                        hole_max_y = min(rand_center_y + hole_size / 2, fov['max_y'])

                        hole_cells = slice_obs_df[
                            (slice_obs_df['center_x'] >= hole_min_x) & 
                            (slice_obs_df['center_x'] <= hole_max_x) &
                            (slice_obs_df['center_y'] >= hole_min_y) & 
                            (slice_obs_df['center_y'] <= hole_max_y)
                        ]
                        
                        if len(hole_cells) >= num_cells:
                            sampled_cells = hole_cells.sample(n=num_cells, replace=False)
                            dominant_tissue = sampled_cells['tissue'].value_counts().idxmax()
                            if dominant_tissue in ['brain ventricle', 'olfactory region']:
                                continue
                            dominant_tissue_ratio = sampled_cells['tissue'].value_counts(normalize=True)[dominant_tissue]

                            if dominant_tissue_ratio > dominance_threshold:
                                gene_expression = slice_x[sampled_cells.index].X
                                ground_truth = GroundTruth(hole_cells=sampled_cells, gene_expression=gene_expression, tissue_percentages=sampled_cells['tissue'].value_counts(normalize=True).to_dict())

                                test_area = TestArea(
                                    hole_min_x=hole_min_x,
                                    hole_max_x=hole_max_x,
                                    hole_min_y=hole_min_y,
                                    hole_max_y=hole_max_y,
                                    dominant_tissue=dominant_tissue
                                )
                                
                                meta_data = {
                                    'platform': platform,
                                    'dataset': dataset,
                                    'donor_id': donor_id,
                                    'slice_id': slice_id,
                                    'test_region_id': holes_found
                                }
                                
                                test_item = TestItem(
                                    adata=slice_x[~slice_x.obs.index.isin(hole_cells.index)],
                                    ground_truth=ground_truth,
                                    test_area=test_area,
                                    meta_data=meta_data
                                )

                                all_test_items.append(test_item)
                                holes_found += 1
                #     break
                # break

            return all_test_items
        
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
    fold_dir = "C:\\Docs_and_Stuff\\Emory\\Lab\\DISCO\\DISCO"
    file_name = "9545288f-ad53-4fea-bc3d-a356cd2786a6.h5ad"
    
    if platform == "MERFISH" and dataset == "MouseBrainAging":
        adata = ad.read_h5ad(fold_dir + "\\" + platform + "\\" + file_name)
        obs = adata.obs
        for field in ['min_x', 'max_x', 'min_y', 'max_y', 'center_x', 'center_y']:
            obs[field] = obs[field].astype(float)
        donor_id_list = list(obs['donor_id'].unique())
        training_samples = []

        for donor_id in donor_id_list:
            donor_obs = obs[obs['donor_id'] == donor_id]
            slice_list = donor_obs['slice'].unique()

            for slice_id in slice_list:
                slice_obs = donor_obs[donor_obs['slice'] == slice_id]
                slice_x = adata[obs['slice'] == slice_id]

                num_samples_generated = 0
                while num_samples_generated < num_samples_per_slice:
                    try:
                        center_x = np.random.uniform(min(slice_obs['min_x']), max(slice_obs['max_x']))
                        center_y = np.random.uniform(min(slice_obs['min_y']), max(slice_obs['max_y']))

                        hole_min_x = max(min(slice_obs['min_x']), center_x - region_size / 2)
                        hole_max_x = min(max(slice_obs['max_x']), center_x + region_size / 2)
                        hole_min_y = max(min(slice_obs['min_y']), center_y - region_size / 2)
                        hole_max_y = min(max(slice_obs['max_y']), center_y + region_size / 2)
                        
                        if hole_max_x - hole_min_x != region_size or hole_max_y - hole_min_y != region_size:
                            continue

                        selected_index = slice_obs[
                            (slice_obs['center_x'] >= hole_min_x) &
                            (slice_obs['center_x'] <= hole_max_x) &
                            (slice_obs['center_y'] >= hole_min_y) &
                            (slice_obs['center_y'] <= hole_max_y)
                        ].index
                        
                        # Check if there are at least the minimum required number of cells
                        if len(selected_index) >= num_cells:
                            # Randomly select num_cells cells without replacement
                            selected_index = np.random.choice(selected_index, size=num_cells, replace=False)
                        else:
                            continue

                        slice_obs.loc[selected_index, 'normalized_x'] = (slice_obs.loc[selected_index, 'center_x'] - hole_min_x) / (hole_max_x - hole_min_x)
                        slice_obs.loc[selected_index, 'normalized_y'] = (slice_obs.loc[selected_index, 'center_y'] - hole_min_y) / (hole_max_y - hole_min_y)

                        dominant_tissue = slice_obs.loc[selected_index, 'tissue'].value_counts().idxmax()
                        if dominant_tissue in ['brain ventricle', 'olfactory region']:
                            continue
                        dominant_ratio = slice_obs.loc[selected_index, 'tissue'].value_counts(normalize=True)[dominant_tissue]

                        if dominant_ratio < dominance_threshold:
                            continue

                        training_samples.append({
                            'normalized_positions': slice_obs.loc[selected_index, ['normalized_x', 'normalized_y']].values,
                            'gene_expressions': slice_x[slice_obs.loc[selected_index].index, :].X,
                            'metadata': {
                                'platform': platform,
                                'dataset': dataset,
                                'donor_id': donor_id,
                                'slice_id': slice_id,
                                'dominant_tissue': dominant_tissue,
                                'tissue_percentage': dominant_ratio
                            }
                        })
                        
                        num_samples_generated += 1

                    except Exception as e:
                        print(f"Failed to process region due to error: {e}")
                        
            #     break
            # break

    return training_samples

if __name__ == "__main__":
>>>>>>> Stashed changes
    
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

def truncated_normal_2d(size, mean=0.5, std=0.15, lower=0.0, upper=1.0):
    """Generate 2D truncated normal distribution data within [lower, upper]."""
    a, b = (lower - mean) / std, (upper - mean) / std
    x = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    y = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    return np.column_stack((x, y))

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
        noise = truncated_normal_2d(size=num_points, mean=0.5, std=0.15, lower=0.0, upper=1.0)
        c0 = np.sqrt(alpha_bar)
        c1 = np.sqrt(1 - alpha_bar)
        noised_positions = c0 * initial_positions + c1 * noise
        positions_list.append(noised_positions)

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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
