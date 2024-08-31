import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import chamfer_loss
from GAN import Generator, Discriminator

class Baseline:
    def __init__(self, gene_expression_dim=374, position_dim=2):
        self.gene_expression_dim = gene_expression_dim
        self.position_dim = position_dim
    
    def fill_region(self):
        raise NotImplementedError("Subclasses should implement this method")

class RandomRegion(Baseline):
    def __init__(self, gene_expression_dim=374, position_dim=2, max_attempts=1000):
        self.gene_expression_dim = gene_expression_dim
        self.position_dim = position_dim
        self.max_attempts = max_attempts
        
    def fill_region(self, adata, test_area, num_cells=50):
        slice_obs_df = pd.DataFrame(adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)
        
        valid_cells = False
        attempts = 0
        
        while not valid_cells and attempts < self.max_attempts:
            rand_center_x = np.random.uniform(slice_obs_df['center_x'].min(), slice_obs_df['center_x'].max())
            rand_center_y = np.random.uniform(slice_obs_df['center_y'].min(), slice_obs_df['center_y'].max()) 
            
            hole_min_x = rand_center_x - (test_area.hole_max_x - test_area.hole_min_x) / 2
            hole_max_x = rand_center_x + (test_area.hole_max_x - test_area.hole_min_x) / 2
            hole_min_y = rand_center_y - (test_area.hole_max_y - test_area.hole_min_y) / 2
            hole_max_y = rand_center_y + (test_area.hole_max_y - test_area.hole_min_y) / 2

            filled_cells = slice_obs_df[
                (slice_obs_df['center_x'] >= hole_min_x) & 
                (slice_obs_df['center_x'] <= hole_max_x) &
                (slice_obs_df['center_y'] >= hole_min_y) & 
                (slice_obs_df['center_y'] <= hole_max_y)
            ]

            if len(filled_cells) >= num_cells:
                filled_cells = filled_cells.sample(n=num_cells, replace=False)
                valid_cells = True

            attempts += 1

        if not valid_cells:
            raise ValueError("Failed to find a valid region with at least 50 cells after 100 attempts.")

        filled_gene_expressions = adata[filled_cells.index].X

        new_coords = filled_cells[['center_x', 'center_y']].copy()
        new_coords['center_x'] = new_coords['center_x'] - new_coords['center_x'].mean() + (test_area.hole_min_x + test_area.hole_max_x) / 2
        new_coords['center_y'] = new_coords['center_y'] - new_coords['center_y'].mean() + (test_area.hole_min_y + test_area.hole_max_y) / 2
        
        return new_coords.values, filled_gene_expressions

class TissueSpecificRandomRegion(Baseline):
    def __init__(self, gene_expression_dim=374, position_dim=2, max_attempts=1000):
        self.gene_expression_dim = gene_expression_dim
        self.position_dim = position_dim
        self.max_attempts = max_attempts
        
    def fill_region(self, adata, test_area, num_cells=50):
        slice_obs_df = pd.DataFrame(adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)

        tissue_cells = slice_obs_df[slice_obs_df['tissue'] == test_area.dominant_tissue]

        if len(tissue_cells) < num_cells:
            raise ValueError("Not enough cells of the dominant tissue to form a valid region.")

        valid_cells = False
        attempts = 0
        while not valid_cells and attempts < self.max_attempts:
            random_cell = tissue_cells.sample(1)
            center_x = random_cell['center_x'].values[0]
            center_y = random_cell['center_y'].values[0]

            # Define a potential region around the randomly selected cell
            hole_min_x = center_x - (test_area.hole_max_x - test_area.hole_min_x) / 2
            hole_max_x = center_x + (test_area.hole_max_x - test_area.hole_min_x) / 2
            hole_min_y = center_y - (test_area.hole_max_y - test_area.hole_min_y) / 2
            hole_max_y = center_y + (test_area.hole_max_y - test_area.hole_min_y) / 2

            # Check if there are sufficient cells in this region
            possible_cells = tissue_cells[
                (tissue_cells['center_x'] >= hole_min_x) & 
                (tissue_cells['center_x'] <= hole_max_x) &
                (tissue_cells['center_y'] >= hole_min_y) & 
                (tissue_cells['center_y'] <= hole_max_y)
            ]

            if len(possible_cells) >= num_cells:
                sampled_cells = possible_cells.sample(n=num_cells, replace=False)
                valid_cells = True
            attempts += 1

        if not valid_cells:
            raise ValueError("Failed to find a valid region with at least 50 dominant tissue cells after 100 attempts.")

        filled_gene_expressions = adata[sampled_cells.index].X

        new_coords = sampled_cells[['center_x', 'center_y']].copy()
        new_coords['center_x'] = new_coords['center_x'] - new_coords['center_x'].mean() + (test_area.hole_min_x + test_area.hole_max_x) / 2
        new_coords['center_y'] = new_coords['center_y'] - new_coords['center_y'].mean() + (test_area.hole_min_y + test_area.hole_max_y) / 2
        
        return new_coords.values, filled_gene_expressions
    
class KNNClustering(Baseline): 
    def fill_region(self, adata, test_area, num_cells=50):
        slice_obs_df = pd.DataFrame(adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)
        
        # get dimensions of the hole
        min_x = test_area.hole_min_x
        max_x = test_area.hole_max_x
        min_y = test_area.hole_min_y
        max_y = test_area.hole_max_y
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
                cells_x = adata.X[mask]
            else: 
                cells_x = np.vstack((cells_x, adata.X[mask]))
            
        if cells_obs.shape[0] < num_cells: 
            raise ValueError("Not enough cells of the neighboring area to perform KNN")
        
        # use K Means clustering to complete the missing region
        cells_obs = cells_obs.reset_index()
        kmeans = KMeans(n_clusters=num_cells, random_state=2024, n_init='auto')
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

<<<<<<< Updated upstream
# Define the Variational Autoencoder Model
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        # Encoder
        self.encoder_fc = nn.Linear(input_dim, 128)
        self.encoder_mu = nn.Linear(128, latent_dim)
        self.encoder_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = F.elu(self.encoder_fc(x))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        
        # Normalize the position outputs to be between 0 and 1
        decoded[:, :, :2] = (torch.tanh(decoded[:, :, :2]) + 1) / 2

        return decoded, mu, logvar

    def get_loss(self, reconstructed, original, mu, logvar):
        # Calculate Chamfer loss for positions
        position_loss = chamfer_loss(reconstructed[:, :, :2], original[:, :, :2])
        
        # Align expressions using nearest neighbor mapping
        pred_positions = reconstructed[:, :, :2]
        true_positions = original[:, :, :2]
        
        # Calculate distances
        dist_matrix = torch.cdist(pred_positions, true_positions, p=2)
        
        # Find the closest true position for each predicted position
        closest_indices = dist_matrix.argmin(dim=2)
        
        # Gather the corresponding expressions
        closest_true_expressions = original.gather(1, closest_indices.unsqueeze(-1).expand(-1, -1, original.size(2)))[:, :, 2:]
        pred_expressions = reconstructed[:, :, 2:]
        
        # Compute MSE for expressions
        expression_loss = F.mse_loss(pred_expressions, closest_true_expressions, reduction='mean')
        
        # Kullback-Leibler divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        total_loss = position_loss + expression_loss + kl_divergence
        return total_loss, position_loss, expression_loss, kl_divergence

# Define the VAE Baseline Class
class VAE(Baseline):
    def __init__(self, gene_expression_dim=374, position_dim=2, latent_dim=10, learning_rate=1e-3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(gene_expression_dim=gene_expression_dim, position_dim=position_dim)
        self.model = VariationalAutoencoder(input_dim=gene_expression_dim + position_dim, latent_dim=latent_dim)
        self.learning_rate = learning_rate
        self.device = device
        self.model.to(self.device)

    def train_model(self, dataloader, epochs=50, max_grad_norm=1.0):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                positions = batch['positions']
                expressions = batch['expressions']
                x = torch.cat((positions, expressions), dim=2).to(self.device)
                optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(x)
                loss, pos_loss, expr_loss, kl_div = self.model.get_loss(reconstructed, x, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                # print(f"Position Loss: {pos_loss.item()}, Expression Loss: {expr_loss.item()}, KL Divergence: {kl_div.item()}")
            print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

    def fill_region(self, adata, test_area, num_cells=50):
        with torch.no_grad():
            self.model.eval()
            sampled_latents = torch.randn(num_cells, self.model.encoder_mu.out_features).to(self.device)
            
            # Use the decoder to generate predicted positions and expressions from the latent space
            decoded = self.model.decoder(sampled_latents)

            # Normalize the decoded position outputs to be between 0 and 1
            decoded[:, :2] = (torch.tanh(decoded[:, :2]) + 1) / 2

            predicted_positions = decoded[:, :2].cpu().numpy()
            predicted_expressions = decoded[:, 2:].cpu().numpy()

            # Scale predicted positions to fit within the test area
            predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
            predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y

            return predicted_positions, predicted_expressions

=======
class GANBaseline(Baseline): 
    def __init__(self, adata, test_area, training_dataloader, num_cells=50, num_epochs=100): 
        super().__init__(adata, test_area, num_cells)
        self.num_epochs = num_epochs
        self.training_dataloader = training_dataloader
        self.generator = None
        self.discriminator = None
        self.input_channel = None
        self.output_channel = None
    
    def fill_region(self): 
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)

        # get dimensions of the hole
        min_x = self.test_area.hole_min_x
        max_x = self.test_area.hole_max_x
        min_y = self.test_area.hole_min_y
        max_y = self.test_area.hole_max_y
        
        # Define input channel for GAN
        self.input_channel = 376
        self.output_channel = 376

        # Define the generator and discriminator and train
        self.generator = Generator(self.input_channel, self.output_channel)
        self.discriminator = Discriminator(self.output_channel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)
        self.discriminator.to(device)

        self.generator, self.discriminator, slice_shape = self.train_gan(num_epochs=self.num_epochs)

        # Generate the missing slice
        noise = torch.randn(1, slice_shape[0], slice_shape[1]).to(device)
        generated_data = self.generator(noise)
        # Apply sigmoid to coordinates
        # generated_data[:, :, :2] = torch.sigmoid(generated_data[:, :, :2])
        generated_data = generated_data.detach().cpu().squeeze(0).squeeze(0)

        coordinates = generated_data[:, :2]
        expressions = generated_data[:, 2:]

        coordinates[:, 0] *= (max_x - min_x)
        coordinates[:, 1] *= (max_y - min_y)

        coordinates[:, 0] += min_x
        coordinates[:, 1] += min_y

        return coordinates.numpy(), expressions.numpy()

    
    def train_gan(self, num_epochs):
        criterion = nn.BCELoss()
        position_criterion = nn.MSELoss()
        expression_criterion = nn.MSELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        slice_shape = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for epoch in range(num_epochs):
            for batch in self.training_dataloader:
                positions = batch['positions'].to(device)
                expressions = batch['expressions'].to(device)
                # metadata = batch['metadata']
                
                train_data = torch.cat((positions, expressions), dim=2).to(device)
                slice_shape = [train_data.shape[1], train_data.shape[2]]

                # Split data into position and expression parts
                real_position_data = train_data[:, :, :2]
                real_expression_data = train_data[:, :, 2:]

                # real_data = train_data.unsqueeze(1)  # Add channel dimension [batch, 1, height, width]
                real_data = train_data

                # real_data.to(device)

                # Train discriminator
                optimizer_d.zero_grad()
                real_labels = torch.ones(real_data.size(0), 1).to(device)
                fake_labels = torch.zeros(real_data.size(0), 1).to(device)

                outputs = self.discriminator(real_data)

                # Discriminator outputs (real and fake)
                real_outputs = self.discriminator(train_data)
                noise = torch.randn(train_data.size(0), train_data.size(1), train_data.size(2)).to(device)
                fake_data = self.generator(noise)
                # fake_data[:, :, :2] = torch.sigmoid(fake_data[:, :, :2])
                fake_outputs = self.discriminator(fake_data)

                d_loss_real = criterion(real_outputs, real_labels)
                d_loss_fake = criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                optimizer_d.step()

                # Train generator
                optimizer_g.zero_grad()
                # Regenerate fake data
                fake_data = self.generator(noise)
                fake_outputs = self.discriminator(fake_data)
                # Calculate generator loss
                g_loss_adv = criterion(fake_outputs, real_labels)  # Adversarial loss
                fake_position_data = fake_data[:, :, :2]
                fake_expression_data = fake_data[:, :, 2:]

                g_loss_position = position_criterion(fake_position_data, positions)
                g_loss_expression = expression_criterion(fake_expression_data, expressions)
                g_loss = g_loss_adv + g_loss_position + g_loss_expression

                g_loss.backward()
                optimizer_g.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        
        return self.generator, self.discriminator, slice_shape
>>>>>>> Stashed changes
