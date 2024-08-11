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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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



