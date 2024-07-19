import os
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
                hole_size: int = 200, num_holes: int = 2, dominance_threshold: float = 0.9):
    
    seed_everything()
    
    fold_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/"
    
    if platform == "MERFISH":
        if dataset == "MouseBrainAging":
            adata = ad.read_h5ad(fold_dir + platform + "/" + dataset + "/2330673b-b5dc-4690-bbbe-8f409362df31.h5ad")
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
                        rand_center_x = fov['center_x'] + random.uniform(-hole_size / 4, hole_size / 4)
                        rand_center_y = fov['center_y'] + random.uniform(-hole_size / 4, hole_size / 4)

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

                        if not hole_cells.empty:
                            dominant_tissue = hole_cells['tissue'].value_counts().idxmax()
                            dominant_tissue_ratio = hole_cells['tissue'].value_counts(normalize=True).loc[dominant_tissue]
                            tissue_percentages = hole_cells['tissue'].value_counts(normalize=True).to_dict()

                            if dominant_tissue_ratio > dominance_threshold:
                                hole_cells_index = hole_cells.index
                                gene_expression = slice_x[slice_obs.index.isin(hole_cells_index)].X

                                adata_copy = slice_x[~slice_obs.index.isin(hole_cells_index)]

                                ground_truth = GroundTruth(hole_cells=hole_cells, gene_expression=gene_expression, tissue_percentages=tissue_percentages)

                                test_area = TestArea(
                                    hole_min_x=hole_min_x,
                                    hole_max_x=hole_max_x,
                                    hole_min_y=hole_min_y,
                                    hole_max_y=hole_max_y,
                                    dominant_tissue=dominant_tissue
                                )
                                
                                meta_data = {}
                                meta_data['platform'] = "MERFISH"
                                meta_data['dataset'] = "MouseBrainAging"
                                meta_data['donor_id'] = donor_id
                                meta_data['slice_id'] = slice_id
                                meta_data['test_region_id'] = holes_found
                                
                                test_item = TestItem(
                                    adata=adata_copy,
                                    ground_truth=ground_truth,
                                    test_area=test_area,
                                    meta_data=meta_data
                                )

                                all_test_items.append(test_item)
                                holes_found += 1
                                
                    break
                break

            return all_test_items
        
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

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

def generate_training_samples(platform: str = "MERFISH", dataset: str = "MouseBrainAging",
                              dominance_threshold: float = 0.9, region_size: float = 200,
                              num_samples_per_slice: int = 10, minimum_cell: int = 40):
    seed_everything()

    fold_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/"
    
    if platform == "MERFISH" and dataset == "MouseBrainAging":
        adata = ad.read_h5ad(fold_dir + platform + "/" + dataset + "/2330673b-b5dc-4690-bbbe-8f409362df31.h5ad")
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

                        selected_index = slice_obs[
                            (slice_obs['center_x'] >= hole_min_x) &
                            (slice_obs['center_x'] <= hole_max_x) &
                            (slice_obs['center_y'] >= hole_min_y) &
                            (slice_obs['center_y'] <= hole_max_y)
                        ].index
                        
                        if selected_index.shape[0] < minimum_cell:
                            continue

                        slice_obs.loc[selected_index, 'normalized_x'] = (slice_obs.loc[selected_index, 'center_x'] - hole_min_x) / (hole_max_x - hole_min_x)
                        slice_obs.loc[selected_index, 'normalized_y'] = (slice_obs.loc[selected_index, 'center_y'] - hole_min_y) / (hole_max_y - hole_min_y)

                        dominant_tissue = slice_obs.loc[selected_index, 'tissue'].value_counts().idxmax()
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
                        
                break
            break

    return training_samples

if __name__ == "__main__":
    
    all_test_items = load_test_data()

    for i, test_item in enumerate(all_test_items):
        print(f"Test Area {i+1}:")
        print(f"  Min X: {test_item.test_area.hole_min_x}")
        print(f"  Max X: {test_item.test_area.hole_max_x}")
        print(f"  Min Y: {test_item.test_area.hole_min_y}")
        print(f"  Max Y: {test_item.test_area.hole_max_y}")
        print(f"  Dominant Tissue: {test_item.test_area.dominant_tissue}")
        print(f"  Number of cells in ground truth: {len(test_item.ground_truth.hole_cells)}")
        print(f"  Number of cells in adata after masking: {test_item.adata.shape[0]}")
        print(f"  Gene expression shape: {test_item.ground_truth.gene_expression.shape}")
        
        print("  Tissue Percentages in Ground Truth:")
        for tissue, percentage in test_item.ground_truth.tissue_percentages.items():
            print(f"    {tissue}: {percentage:.2%}")
        
        visualize_test_region(pd.DataFrame(test_item.adata.obs), test_item.test_area, title=f'Test Region {i+1}')