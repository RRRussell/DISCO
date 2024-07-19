import torch
from torch.utils.data import Dataset, DataLoader

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
    
    








