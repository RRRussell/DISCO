import torch
from torch.utils.data import Dataset

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
        
class STDataset(Dataset):
    def __init__(self, training_samples):
        """
        Initialize the dataset with the training samples.
        :param training_samples: A list of dictionaries containing sample data.
        """
        self.samples = training_samples

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset.
        :param idx: The index of the item.
        :return: A dictionary containing tensors for positions, gene expressions, and metadata.
        """
        sample = self.samples[idx]
        positions = torch.tensor(sample['normalized_positions'], dtype=torch.float32)
        expressions = torch.tensor(sample['gene_expressions'], dtype=torch.float32)
        
        # Metadata could be returned as is, or parts of it could be converted to tensors if needed
        metadata = sample['metadata']

        return {'positions': positions, 'expressions': expressions, 'metadata': metadata}
    
def filter_by_tissue(dataset, tissue_name):
    """
    Filter the dataset to only include samples with the specified tissue type.
    :param dataset: The original dataset.
    :param tissue_name: The name of the tissue type to filter by.
    :return: A filtered dataset containing only the specified tissue type.
    """
    filtered_samples = [sample for sample in dataset if sample['metadata']['dominant_tissue'] == tissue_name]
    return STDataset(filtered_samples)








