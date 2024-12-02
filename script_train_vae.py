from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import generate_training_samples
from dataset import STDataset
from baseline import VAE
from utils import sanitize_name

def train_leave_one_out_vaes(dataloader, base_save_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    base_save_path = Path(base_save_path)
    
    # Extract unique donor IDs from the dataset
    donor_ids = set([sample['metadata']['donor_id'] for sample in dataloader.dataset.samples])
    
    for leave_out_donor in donor_ids:
        print(f"Training models with leave-one-out: excluding donor {leave_out_donor}")
        
        # Filter out the current donor's samples
        train_samples = [sample for sample in dataloader.dataset.samples if sample['metadata']['donor_id'] != leave_out_donor]
        
        # Create a dataloader for the remaining donors
        train_dataset = STDataset(train_samples)
        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
        
        # Train General VAE using data from all other donors
        vae_general = VAE(gene_expression_dim=374, position_dim=2, latent_dim=10, device=device)
        vae_general.train_model(train_dataloader, epochs=10)
        
        # Save the general VAE model
        general_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / "general"
        general_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(vae_general.model.state_dict(), general_save_path / f"vae_general.pth")
        print(f"General model saved excluding donor {leave_out_donor} at {general_save_path}")
        
        # Get unique tissue types from the remaining samples
        tissue_types = set([sample['metadata']['dominant_tissue'] for sample in train_samples])
        
        for tissue_type in tissue_types:
            print(f"Training tissue-specific model excluding donor {leave_out_donor} for tissue {tissue_type}")
            
            # Filter out the specific tissue type samples from remaining donors
            tissue_samples = [sample for sample in train_samples if sample['metadata']['dominant_tissue'] == tissue_type]
            
            # Create a dataloader for the tissue-specific samples
            tissue_dataset = STDataset(tissue_samples)
            tissue_dataloader = DataLoader(tissue_dataset, batch_size=10, shuffle=True, num_workers=4)
            
            # Train Tissue-Specific VAE
            vae_tissue = VAE(gene_expression_dim=374, position_dim=2, latent_dim=10, device=device)
            vae_tissue.train_model(tissue_dataloader, epochs=10)
            
            # Save the tissue-specific VAE model
            tissue_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / f"tissue_{sanitize_name(tissue_type)}"
            tissue_save_path.mkdir(parents=True, exist_ok=True)
            torch.save(vae_tissue.model.state_dict(), tissue_save_path / f"vae_tissue_{sanitize_name(tissue_type)}.pth")
            print(f"Tissue-specific model saved excluding donor {leave_out_donor} for tissue {tissue_type} at {tissue_save_path}")

def load_vae_model(model_path, device, gene_expression_dim=374, position_dim=2, latent_dim=10):
    """
    Loads a VAE model from the specified path.
    """
    vae = VAE(gene_expression_dim=gene_expression_dim, position_dim=position_dim, latent_dim=latent_dim, device=device)
    vae.model.load_state_dict(torch.load(model_path, map_location=device))
    vae.model.to(device)
    return vae

if __name__ == "__main__":

    training_samples = generate_training_samples(num_samples_per_slice=500)
    dataset = STDataset(training_samples)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    base_save_path = '/home/zihend1/Diffusion/DISCO/DISCO/saved_models'
    train_leave_one_out_vaes(dataloader, base_save_path)



