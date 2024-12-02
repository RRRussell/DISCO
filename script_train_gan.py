from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import generate_training_samples
from dataset import STDataset
from baseline import GANBaseline
from utils import sanitize_name

def train_leave_one_out_gan(dataloader, base_save_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        
        # Train General GAN using data from all other donors
        gan_general = GANBaseline(gene_expression_dim=374, position_dim=2, device=device)
        gan_general.train_model(train_dataloader, epochs=10)
        
        # Save the general VAE model
        general_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / "general"
        general_save_path.mkdir(parents=True, exist_ok=True)
        save_gan_model(gan_general, general_save_path)
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
            
            # Train Tissue-Specific GAN
            gan_tissue = GANBaseline(gene_expression_dim=374, position_dim=2, device=device)
            gan_tissue.train_model(tissue_dataloader, epochs=10)
            
            # Save the tissue-specific VAE model
            tissue_save_path = base_save_path / f"leave_out_donor_{sanitize_name(leave_out_donor)}" / f"tissue_{sanitize_name(tissue_type)}"
            tissue_save_path.mkdir(parents=True, exist_ok=True)
            save_gan_model(gan_tissue, tissue_save_path)
            print(f"Tissue-specific model saved excluding donor {leave_out_donor} for tissue {tissue_type} at {tissue_save_path}")

def save_gan_model(gan_model, save_path):
    """
    Saves the entire GAN model's state_dict into a single file.

    Args:
    - gan_model (GANBaseline): The GANBaseline model instance.
    - save_path (str or Path): Path to save the model weights.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a combined state_dict for both generator and discriminator
    state_dict = {
        "generator": gan_model.generator.state_dict(),
        "discriminator": gan_model.discriminator.state_dict(),
    }

    # Save the combined state_dict
    torch.save(state_dict, save_path / "gan.pth")
    print(f"GAN model saved to {save_path / 'gan.pth'}")


def load_gan_model(model_path, device):
    """
    Loads the GAN model's state_dict from a single file.

    Args:
    - model_path (str or Path): Path to load the model weights from.
    - device (torch.device): The device to map the model to.

    Returns:
    - gan_model (GANBaseline): The GANBaseline model instance with loaded weights.
    """
    model_path = Path(model_path)

    # Initialize the GAN model with the same parameters used during training
    gan_model = GANBaseline(device=device)  # Adjust the arguments as needed

    # Load the combined state_dict
    state_dict = torch.load(model_path / "gan.pth", map_location=device)
    gan_model.generator.load_state_dict(state_dict["generator"])
    gan_model.discriminator.load_state_dict(state_dict["discriminator"])

    # Move the models to the specified device
    gan_model.generator.to(device)
    gan_model.discriminator.to(device)

    print(f"GAN model loaded from {model_path / 'gan.pth'}")
    return gan_model

if __name__ == "__main__":

    training_samples = generate_training_samples(num_samples_per_slice=500)
    dataset = STDataset(training_samples)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    base_save_path = '/home/zihend1/Diffusion/DISCO/DISCO/saved_models'
    train_leave_one_out_gan(dataloader, base_save_path)



