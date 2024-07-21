import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
    

class KNNClusteringBaseline(Baseline): 
    def fill_region(self):
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)
        
        # get dimensions of the hole
        min_x = self.test_area.hole_min_x
        max_x = self.test_area.hole_max_x
        min_y = self.test_area.hole_min_y
        max_y = self.test_area.hole_max_y
        x_len = max_x - min_x
        y_len = max_y - min_y
        
        # find the 8 neighboring patches
        neighbors = []
        # left
        neighbors.append({
            'min_x': min_x - x_len,
            'max_x': min_x,
            'min_y': min_y,
            'max_y': max_y
        })
        # right
        neighbors.append({
            'min_x': max_x,
            'max_x': max_x + x_len,
            'min_y': min_y,
            'max_y': max_y
        })
        # bottom
        neighbors.append({
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y - y_len,
            'max_y': min_y
        })
        # top
        neighbors.append({
            'min_x': min_x,
            'max_x': max_x,
            'min_y': max_y,
            'max_y': max_y + y_len
        })
        # bottom-left
        neighbors.append({
            'min_x': min_x - x_len,
            'max_x': min_x,
            'min_y': min_y - y_len,
            'max_y': min_y
        })
        # bottom-right
        neighbors.append({
            'min_x': max_x,
            'max_x': max_x + x_len,
            'min_y': min_y - y_len,
            'max_y': min_y
        })
        # top-left
        neighbors.append({
            'min_x': min_x - x_len,
            'max_x': min_x,
            'min_y': max_y,
            'max_y': max_y + y_len
        })
        # top-right
        neighbors.append({
            'min_x': max_x,
            'max_x': max_x + x_len,
            'min_y': max_y,
            'max_y': max_y + y_len
        })

        # find cells in the neighboring regions
        cells_obs = pd.DataFrame()
        cells_x = None
        for region in neighbors: 
            mask = (
                (slice_obs_df['center_x'] >= region['min_x']) & 
                (slice_obs_df['center_x'] < region['max_x']) & 
                (slice_obs_df['center_y'] >= region['min_y']) & 
                (slice_obs_df['center_y'] < region['max_y'])
            )
            current_obs = slice_obs_df[mask].copy()
            current_obs['relative_x'] = current_obs['center_x'] - region['min_x']
            current_obs['relative_y'] = current_obs['center_y'] - region['min_y']
            cells_obs = pd.concat([cells_obs, current_obs])
            if cells_x is None: 
                cells_x = self.adata.X[mask]
            else: 
                cells_x = np.vstack((cells_x, self.adata.X[mask]))
            
        if cells_obs.shape[0] < self.num_cells: 
            raise ValueError("Not enough cells of the neighboring area to perform KNN")
        
        # use K Means clustering to complete the missing region
        cells_obs = cells_obs.reset_index()
        kmeans = KMeans(n_clusters=self.num_cells, random_state=2024)
        cells_obs['cluster'] = kmeans.fit_predict(cells_obs[['relative_x', 'relative_y']])
        # find the mean coordinates for each predicted cell
        mean_coordinates = cells_obs.groupby('cluster')[['relative_x', 'relative_y']].mean().reset_index()
        # find the gene expressions for each predicted cell
        cells_x_df = pd.DataFrame(cells_x)
        cells_x_df['cluster'] = cells_obs['cluster']
        mean_expression = cells_x_df.groupby('cluster').mean().reset_index()
        
        mean_coordinates = mean_coordinates.drop(columns=['cluster'])
        mean_coordinates['relative_x'] = mean_coordinates['relative_x'] + min_x
        mean_coordinates['relative_y'] = mean_coordinates['relative_y'] + min_y
        mean_coordinates = mean_coordinates.to_numpy()
        mean_expression = mean_expression.drop(columns=['cluster']).to_numpy()

        return mean_coordinates, mean_expression
