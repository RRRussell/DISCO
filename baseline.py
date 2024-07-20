import numpy as np
import pandas as pd

class Baseline:
    def __init__(self, adata, test_area, num_cells=50):
        self.adata = adata
        self.test_area = test_area
        self.num_cells = num_cells
    
    def fill_region(self):
        raise NotImplementedError("Subclasses should implement this method")

class RandomRegionBaseline(Baseline):
    def fill_region(self):
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)
        
        valid_cells = False
        attempts = 0
        
        while not valid_cells and attempts < 1000:
            rand_center_x = np.random.uniform(slice_obs_df['center_x'].min(), slice_obs_df['center_x'].max())
            rand_center_y = np.random.uniform(slice_obs_df['center_y'].min(), slice_obs_df['center_y'].max()) 
            
            hole_min_x = rand_center_x - (self.test_area.hole_max_x - self.test_area.hole_min_x) / 2
            hole_max_x = rand_center_x + (self.test_area.hole_max_x - self.test_area.hole_min_x) / 2
            hole_min_y = rand_center_y - (self.test_area.hole_max_y - self.test_area.hole_min_y) / 2
            hole_max_y = rand_center_y + (self.test_area.hole_max_y - self.test_area.hole_min_y) / 2

            filled_cells = slice_obs_df[
                (slice_obs_df['center_x'] >= hole_min_x) & 
                (slice_obs_df['center_x'] <= hole_max_x) &
                (slice_obs_df['center_y'] >= hole_min_y) & 
                (slice_obs_df['center_y'] <= hole_max_y)
            ]

            if len(filled_cells) >= self.num_cells:
                filled_cells = filled_cells.sample(n=self.num_cells, replace=False)
                valid_cells = True

            attempts += 1

        if not valid_cells:
            raise ValueError("Failed to find a valid region with at least 50 cells after 100 attempts.")

        filled_gene_expressions = self.adata[filled_cells.index].X

        new_coords = filled_cells[['center_x', 'center_y']].copy()
        new_coords['center_x'] = new_coords['center_x'] - new_coords['center_x'].mean() + (self.test_area.hole_min_x + self.test_area.hole_max_x) / 2
        new_coords['center_y'] = new_coords['center_y'] - new_coords['center_y'].mean() + (self.test_area.hole_min_y + self.test_area.hole_max_y) / 2
        
        return new_coords.values, filled_gene_expressions

class TissueSpecificRandomRegionBaseline(Baseline):
    def fill_region(self):
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)

        tissue_cells = slice_obs_df[slice_obs_df['tissue'] == self.test_area.dominant_tissue]

        if len(tissue_cells) < self.num_cells:
            raise ValueError("Not enough cells of the dominant tissue to form a valid region.")

        valid_cells = False
        attempts = 0
        while not valid_cells and attempts < 1000:
            random_cell = tissue_cells.sample(1)
            center_x = random_cell['center_x'].values[0]
            center_y = random_cell['center_y'].values[0]

            # Define a potential region around the randomly selected cell
            hole_min_x = center_x - (self.test_area.hole_max_x - self.test_area.hole_min_x) / 2
            hole_max_x = center_x + (self.test_area.hole_max_x - self.test_area.hole_min_x) / 2
            hole_min_y = center_y - (self.test_area.hole_max_y - self.test_area.hole_min_y) / 2
            hole_max_y = center_y + (self.test_area.hole_max_y - self.test_area.hole_min_y) / 2

            # Check if there are sufficient cells in this region
            possible_cells = tissue_cells[
                (tissue_cells['center_x'] >= hole_min_x) & 
                (tissue_cells['center_x'] <= hole_max_x) &
                (tissue_cells['center_y'] >= hole_min_y) & 
                (tissue_cells['center_y'] <= hole_max_y)
            ]

            if len(possible_cells) >= 50:
                sampled_cells = possible_cells.sample(n=50, replace=False)
                valid_cells = True
            attempts += 1

        if not valid_cells:
            raise ValueError("Failed to find a valid region with at least 50 dominant tissue cells after 100 attempts.")

        filled_gene_expressions = self.adata[sampled_cells.index].X

        new_coords = sampled_cells[['center_x', 'center_y']].copy()
        new_coords['center_x'] = new_coords['center_x'] - new_coords['center_x'].mean() + (self.test_area.hole_min_x + self.test_area.hole_max_x) / 2
        new_coords['center_y'] = new_coords['center_y'] - new_coords['center_y'].mean() + (self.test_area.hole_min_y + self.test_area.hole_max_y) / 2
        
        return new_coords.values, filled_gene_expressions
