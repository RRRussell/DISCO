import numpy as np
import pandas as pd

class Baseline:
    def __init__(self, adata, test_area):
        self.adata = adata
        self.test_area = test_area
    
    def fill_region(self):
        raise NotImplementedError("Subclasses should implement this method")

class RandomRegionBaseline(Baseline):
    def fill_region(self):
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)
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

        filled_gene_expressions = self.adata[filled_cells.index].X

        new_coords = filled_cells[['center_x', 'center_y']].copy()
        new_coords['center_x'] = new_coords['center_x'] - new_coords['center_x'].mean() + (self.test_area.hole_min_x + self.test_area.hole_max_x) / 2
        new_coords['center_y'] = new_coords['center_y'] - new_coords['center_y'].mean() + (self.test_area.hole_min_y + self.test_area.hole_max_y) / 2
        
        return new_coords.values, filled_gene_expressions
