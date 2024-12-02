from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import generate_training_samples
from dataset import STDataset
from baseline import LatentSpaceGAN
from utils import sanitize_name

def train_leave_one_out_lgan(dataloader, base_save_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        
        # Train General lGAN using data from all other donors
        lgan_general = LatentSpaceGAN(gene_expression_dim=374, position_dim=2, device=device)
        lgan_general.train_model(train_dataloader, epochs=10)
        
        # Save the general VAE model
        general_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / "general"
        general_save_path.mkdir(parents=True, exist_ok=True)
        save_lgan_model(lgan_general, general_save_path)
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
            
            # Train Tissue-Specific lGAN
            lgan_tissue = LatentSpaceGAN(gene_expression_dim=374, position_dim=2, device=device)
            lgan_tissue.train_model(tissue_dataloader, epochs=10)
            
            # Save the tissue-specific VAE model
            tissue_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / f"tissue_{sanitize_name(tissue_type)}"
            tissue_save_path.mkdir(parents=True, exist_ok=True)
            save_lgan_model(lgan_tissue, tissue_save_path)
            print(f"Tissue-specific model saved excluding donor {leave_out_donor} for tissue {tissue_type} at {tissue_save_path}")

def save_lgan_model(lgan_model, save_path):
    """
    Saves the entire LatentSpaceGAN model's state_dicts.

    Args:
    - lgan_model (LatentSpaceGAN): The LatentSpaceGAN model instance.
    - save_path (str or Path): Path to save the model weights.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a combined state_dict for all components
    state_dict = {
        "vae": lgan_model.vae.model.state_dict(),
        "generator": lgan_model.generator.state_dict(),
        "discriminator": lgan_model.discriminator.state_dict(),
    }

    # Save the combined state_dict
    torch.save(state_dict, save_path / "lgan.pth")
    print(f"LatentSpaceGAN model saved to {save_path / 'lgan.pth'}")

def load_lgan_model(model_path, device, gene_expression_dim=374, position_dim=2, latent_dim=10, noise_dim=10, learning_rate=1e-3):
    """
    Loads the LatentSpaceGAN model's state_dicts.

    Args:
    - model_path (str or Path): Path to load the model weights from.
    - device (torch.device): The device to map the model to.
    - gene_expression_dim (int): The gene expression dimension (default: 374).
    - position_dim (int): The position dimension (default: 2).
    - latent_dim (int): The latent dimension (default: 10).
    - noise_dim (int): The noise dimension for the GAN generator (default: 10).
    - learning_rate (float): The learning rate for training (default: 1e-3).

    Returns:
    - lgan_model (LatentSpaceGAN): The LatentSpaceGAN model instance with loaded weights.
    """
    model_path = Path(model_path)

    # Initialize the LatentSpaceGAN model
    lgan_model = LatentSpaceGAN(
        gene_expression_dim=gene_expression_dim,
        position_dim=position_dim,
        latent_dim=latent_dim,
        noise_dim=noise_dim,
        learning_rate=learning_rate,
        device=device
    )

    # Load the combined state_dict
    state_dict = torch.load(model_path / "lgan.pth", map_location=device)

    # Load the weights into the corresponding components
    lgan_model.vae.model.load_state_dict(state_dict["vae"])
    lgan_model.generator.load_state_dict(state_dict["generator"])
    lgan_model.discriminator.load_state_dict(state_dict["discriminator"])

    # Move models to the specified device
    lgan_model.vae.model.to(device)
    lgan_model.generator.to(device)
    lgan_model.discriminator.to(device)

    print(f"LatentSpaceGAN model loaded from {model_path / 'lgan.pth'}")
    return lgan_model

training_samples = generate_training_samples(num_samples_per_slice=1)
dataset = STDataset(training_samples)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

base_save_path = '/home/zihend1/Diffusion/DISCO/DISCO/saved_models'

if __name__ == "__main__":

    training_samples = generate_training_samples(num_samples_per_slice=500)
    dataset = STDataset(training_samples)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    base_save_path = '/home/zihend1/Diffusion/DISCO/DISCO/saved_models'
    train_leave_one_out_lgan(dataloader, base_save_path)














