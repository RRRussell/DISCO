from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
    
    def forward(self, t):
        frequency = torch.exp(-torch.arange(0, self.embedding_size, 2, device=self.device) * (torch.log(torch.tensor(10000.0)) / self.embedding_size))
        arguments = t.unsqueeze(1) * frequency.unsqueeze(0)
        embeddings = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=-1)
        return embeddings

class STModel(nn.Module):
    def __init__(self, n_positions, n_features, n_tissues, device, hidden_dim=32):
        super(STModel, self).__init__()
        self.n_positions = n_positions
        self.n_features = n_features
        self.n_tissues = n_tissues
        self.device = device
        self.hidden_dim = hidden_dim

        # Convolution layers for positions and expressions
        self.position_conv = nn.Sequential(
            nn.Conv1d(self.n_positions, self.hidden_dim, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )
        
        self.expression_conv = nn.Sequential(
            nn.Conv1d(self.n_features, self.hidden_dim, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )

        # Embeddings for tissue type and temporal information
        self.context_embedding = nn.Embedding(n_tissues, self.hidden_dim * 2)
        self.time_embedding = SinusoidalPositionalEmbedding(self.hidden_dim * 2, self.device)

        # Final convolution to map to desired output size
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(self.hidden_dim, self.n_positions+self.n_features, kernel_size=1)
        )

    def forward(self, x_t, tissue_type, t, context_mask):
        positions = x_t[:, :, :self.n_positions]
        expressions = x_t[:, :, self.n_positions:]

        # Ensure correct input orientation for convolutions
        positions = positions.permute(0, 2, 1)
        expressions = expressions.permute(0, 2, 1)
        
        # Process through respective convolutions
        pos_encoded = self.position_conv(positions)
        exp_encoded = self.expression_conv(expressions)

        # Embedding and time modulation
        tissue_embed = self.context_embedding(tissue_type)
        time_embed = self.time_embedding(t.unsqueeze(-1)).squeeze(-2)
        
        # Expand embeddings to match feature dimensions
        tissue_embed = tissue_embed.unsqueeze(2).expand(-1, -1, pos_encoded.size(2))
        time_embed = time_embed.unsqueeze(2).expand_as(tissue_embed)
        
        # Combine features and embeddings
        combined_features = (tissue_embed * torch.cat([pos_encoded, exp_encoded], dim=1)) + time_embed
        output = self.final_conv(combined_features)
        
        # Ensure output has the same sample dimension order as input
        output = output.permute(0, 2, 1)
        
        # # Apply sigmoid to position predictions to constrain them between 0 and 1
        # output[:, :, :self.n_positions] = torch.sigmoid(output[:, :, :self.n_positions])
        
        # print("Output shape after permute and sigmoid:", output.shape)
        return output

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # Precompute and buffer schedule parameters
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # Sampling timestep randomly
        noise = torch.randn_like(x)  # Gaussian noise for both positions and expressions

        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise  # Perturbed input

        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)  # Dropout simulation
        predicted_noise = self.nn_model(x_t, c, _ts / self.n_T, context_mask)  # Model prediction

        # Position loss: MSE loss for positions
        pos_loss = self.loss_mse(predicted_noise[:, :, :2], noise[:, :, :2])

        # Expression loss: Simple MSE loss
        exp_loss = self.loss_mse(predicted_noise[:, :, 2:], noise[:, :, 2:])

        print("position loss:", pos_loss.item(), "expression loss:", exp_loss.item())
        
        total_loss = pos_loss + exp_loss  # Combine losses
        
        return total_loss
    
    def sample_single(self, tissue_index):
        # Initialize a single sample with random noise (initial condition at the last timestep)
        x_i = torch.randn(1, 50, 376).to(self.device)  # 50 samples, 376 features (2 positions + 374 expressions)
        
        # Get the tissue index as context
        c_i = torch.tensor([tissue_index]).to(self.device)
        
        # No dropout during testing
        context_mask = torch.zeros_like(c_i).to(self.device)

        # Reverse the noise schedule to generate from noise to data
        for i in range(self.n_T, 0, -1):
            t_i = torch.tensor([i / self.n_T], dtype=torch.float32).to(self.device)
            
            # Forward through the model to predict the reverse step
            predicted_noise = self.nn_model(x_i, c_i, t_i, context_mask)
            
            # Reverse the diffusion process
            x_i = (
                self.oneover_sqrta[i] * (x_i - predicted_noise * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * torch.randn_like(x_i)
            )

            # # Ensure positions stay within [0, 1]
            # x_i[:, :, :2] = torch.sigmoid(x_i[:, :, :2])

            # print(f"Step {self.n_T - i + 1}/{self.n_T}, Sample Shape: {x_i.shape}")

        return x_i.squeeze(0)  # Remove the batch dimension for output

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    