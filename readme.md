# DISCO: Diffusion Model for Spatial Transcriptomics Data Completion

**DISCO** (**DI**ffusion model for **S**patial transcriptomics data **CO**mpletion) is a framework for reconstructing large missing regions in spatial transcriptomics datasets. By integrating graph neural networks (GNNs) and denoising diffusion models, DISCO enables biologically coherent and spatially consistent region completion.

---

## Model Architecture

- **GNN-based Region Encoder** (`models/encoder.py`)  
  Encodes fine-grained spatial and gene expression features from observed regions.  

- **Position Diffusion Module (PDM)** (`models/diffusion.py`)  
  Predicts spatial coordinates for cells in missing regions via a denoising diffusion process.

- **Gene Expression Diffusion Module (GEDM)** (`models/diffusion.py`)  
  Predicts gene expression profiles conditioned on the generated spatial coordinates.

- **Neighborhood Integration**  
  Incorporates surrounding observed regions during inference to ensure smooth transitions.

---

## Repository Structure

```plaintext
DISCO/
│
├── models/                     # Core model components
│   ├── autoencoder.py          # Autoencoder implementation (if applicable)
│   ├── common.py               # Shared components
│   ├── diffusion.py            # Position and gene diffusion modules
│   ├── encoder.py              # GNN-based Region Encoder
│   └── vae_gaussian.py         # Gaussian VAE module
│
├── GAN.py                      # GAN baseline implementation
├── baseline.py                 # Other baseline models
├── data.py                     # Data loading and preprocessing
├── dataset.py                  # Dataset construction
├── evaluate.py                 # Evaluation metrics and scripts
├── parameter.py                # Model parameter configuration
├── utils.py                    # Utility functions
