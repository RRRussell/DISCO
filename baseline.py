import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import chamfer_loss, compute_unordered_gene_expression_loss, normalize_positions
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
        
        best_candidate = None
        best_candidate_size = 0
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
            else:
                # Update the best candidate if this region has more cells
                if len(filled_cells) > best_candidate_size:
                    best_candidate = filled_cells
                    best_candidate_size = len(filled_cells)

            attempts += 1

        # If no valid region is found, use the best candidate
        if not valid_cells:
            if best_candidate is None:
                raise ValueError("Failed to find any region with cells after {} attempts.".format(self.max_attempts))
            filled_cells = best_candidate
            print(f"Warning: Using best candidate with {best_candidate_size} cells instead of {num_cells} cells.")

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

        best_candidate = None
        best_candidate_size = 0
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
            else:
                # Update the best candidate if this region has more cells
                if len(possible_cells) > best_candidate_size:
                    best_candidate = possible_cells
                    best_candidate_size = len(possible_cells)

            attempts += 1

        if not valid_cells:
            if best_candidate is None:
                raise ValueError("Failed to find any region with cells after {} attempts.".format(self.max_attempts))
            sampled_cells = best_candidate
            print(f"Warning: Using best candidate with {best_candidate_size} cells instead of {num_cells} cells.")

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
        
        # # Normalize the position outputs to be between 0 and 1
        # decoded[:, :, :2] = (torch.tanh(decoded[:, :, :2]) + 1) / 2
        # decoded[:, :, :2] = 2*torch.sigmoid(decoded[:, :, :2])-1

        return decoded, mu, logvar

    def get_loss(self, reconstructed, original, mu, logvar):
        true_positions = original[:, :, :2]
        true_expressions = original[:, :, 2:]
        pred_positions = reconstructed[:, :, :2]
        pred_expressions = reconstructed[:, :, 2:]
        
        position_loss = chamfer_loss(true_positions, pred_positions)
        expression_loss = compute_unordered_gene_expression_loss(true_positions, true_expressions, pred_positions, pred_expressions)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total_loss = position_loss + expression_loss + kl_divergence
        # total_loss = position_loss + kl_divergence
        return total_loss, position_loss, expression_loss, kl_divergence

# Define the VAE Baseline Class
class VAE(Baseline):
    def __init__(self, gene_expression_dim=374, position_dim=2, latent_dim=10, learning_rate=1e-3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(gene_expression_dim=gene_expression_dim, position_dim=position_dim)
        self.model = VariationalAutoencoder(input_dim=gene_expression_dim + position_dim, latent_dim=latent_dim)
        self.learning_rate = learning_rate
        self.device = device
        self.model.to(self.device)

    def train_model(self, dataloader, epochs=500, max_grad_norm=1.0):
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
                print(f"Position Loss: {pos_loss.item()}, Expression Loss: {expr_loss.item()}, KL Divergence: {kl_div.item()}")
            print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

    def fill_region(self, adata, test_area, num_cells=50):
        with torch.no_grad():
            self.model.eval()
            sampled_latents = torch.randn(num_cells, self.model.encoder_mu.out_features).to(self.device)
            
            # Use the decoder to generate predicted positions and expressions from the latent space
            decoded = self.model.decoder(sampled_latents)

            # # Normalize the decoded position outputs to be between 0 and 1
            # decoded[:, :2] = (torch.tanh(decoded[:, :2]) + 1) / 2
            # decoded[:, :2] = 2*torch.sigmoid(decoded[:, :2])-1
    
            decoded[:, :2] = normalize_positions(decoded[:, :2])

            predicted_positions = decoded[:, :2].cpu().numpy()
            predicted_expressions = decoded[:, 2:].cpu().numpy()
            
            predicted_positions[:, 0] = (predicted_positions[:, 0] + 1) / 2
            predicted_positions[:, 1] = (predicted_positions[:, 1] + 1) / 2

            # Scale predicted positions to fit within the test area
            predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
            predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y

            return predicted_positions, predicted_expressions

class GANBaseline(Baseline): 
    def __init__(self, num_cells=50, gene_expression_dim=374, position_dim=2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")): 
        super().__init__(gene_expression_dim=gene_expression_dim, position_dim=position_dim)
        self.num_cells = num_cells
        self.input_channel = gene_expression_dim+position_dim
        self.output_channel = gene_expression_dim+position_dim
        self.device = device
        
        # Define the generator and discriminator and train
        self.generator = Generator(self.input_channel, self.output_channel).to(self.device)
        self.discriminator = Discriminator(self.output_channel).to(self.device)
        
    def train_model(self, dataloader, epochs=50):
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        for epoch in range(epochs):
            for batch in dataloader:
                positions = batch['positions'].to(self.device)
                expressions = batch['expressions'].to(self.device)
                # metadata = batch['metadata']
                
                train_data = torch.cat((positions, expressions), dim=2).to(self.device)

                # real_data = train_data.unsqueeze(1)  # Add channel dimension [batch, 1, height, width]
                real_data = train_data
                
                # Train discriminator
                optimizer_d.zero_grad()
                real_labels = torch.ones(real_data.size(0), 1).to(self.device)
                fake_labels = torch.zeros(real_data.size(0), 1).to(self.device)

                # Discriminator outputs (real and fake)
                real_outputs = self.discriminator(train_data)
                noise = torch.randn(train_data.size(0), train_data.size(1), train_data.size(2)).to(self.device)
                fake_data = self.generator(noise)
                # fake_data[:, :, :2] = 2*torch.sigmoid(fake_data[:, :, :2])-1
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
                # fake_data[:, :, :2] = 2*torch.sigmoid(fake_data[:, :, :2])-1
                fake_outputs = self.discriminator(fake_data)
                # Calculate generator loss
                g_loss_adv = criterion(fake_outputs, real_labels)  # Adversarial loss
                fake_position_data = fake_data[:, :, :2]
                fake_expression_data = fake_data[:, :, 2:]

                g_loss_position = chamfer_loss(positions, fake_position_data)
                g_loss_expression = compute_unordered_gene_expression_loss(positions, expressions, fake_position_data, fake_expression_data)
                g_loss = g_loss_adv + g_loss_position + g_loss_expression

                g_loss.backward()
                optimizer_g.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    
    def fill_region(self, adata, test_area): 
        self.adata = adata
        self.test_area = test_area
        
        slice_obs_df = pd.DataFrame(self.adata.obs)
        slice_obs_df['center_x'] = slice_obs_df['center_x'].astype(float)
        slice_obs_df['center_y'] = slice_obs_df['center_y'].astype(float)

        # Generate the missing slice
        noise = torch.randn(self.num_cells, self.output_channel).to(self.device)
        generated_data = self.generator(noise)
        # Apply sigmoid to coordinates
        # generated_data[:, :2] = 2*torch.sigmoid(generated_data[:, :2])-1
        generated_data[:, :2] = normalize_positions(generated_data[:, :2])
        generated_data = generated_data.detach().cpu().squeeze(0).squeeze(0)

        coordinates = generated_data[:, :2]
        expressions = generated_data[:, 2:]
        
        coordinates[:, 0] = (coordinates[:, 0] + 1) / 2
        coordinates[:, 1] = (coordinates[:, 1] + 1) / 2
        
        coordinates[:, 0] = coordinates[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
        coordinates[:, 1] = coordinates[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y

        return coordinates.numpy(), expressions.numpy()

class LatentSpaceGAN(Baseline):
    def __init__(self, gene_expression_dim=374, position_dim=2, latent_dim=10, noise_dim=10, learning_rate=1e-3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(gene_expression_dim=gene_expression_dim, position_dim=position_dim)
        
        # Initialize the VAE
        self.vae = VAE(gene_expression_dim=gene_expression_dim, position_dim=position_dim, latent_dim=latent_dim, learning_rate=learning_rate, device=device)
        
        # Initialize the GAN components
        self.generator = Generator(noise_dim, latent_dim)
        self.discriminator = Discriminator(latent_dim)
        self.generator.to(device)
        self.discriminator.to(device)
        
        self.learning_rate = learning_rate
        self.device = device

    def train_model(self, dataloader, epochs=50, max_grad_norm=1.0):
        # Step 1: Train the VAE first
        print("Training VAE...")
        self.vae.train_model(dataloader, epochs=epochs, max_grad_norm=max_grad_norm)

        # Step 2: Train the GAN on the latent space of the VAE
        print("Training GAN...")
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)
        criterion = nn.BCELoss()

        self.vae.model.eval()  # Freeze the VAE during GAN training
        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            total_g_loss = 0
            total_d_loss = 0
            for batch in dataloader:
                positions = batch['positions'].to(self.device)
                expressions = batch['expressions'].to(self.device)
                x = torch.cat((positions, expressions), dim=2).to(self.device)
                
                # Encode input data to obtain real latent codes
                with torch.no_grad():
                    mu, logvar = self.vae.model.encode(x)
                    real_latent_code = self.vae.model.reparameterize(mu, logvar)
                
                batch_size = x.size(0)
                noise = torch.randn(batch_size, x.size(1), self.generator.model[0].in_features).to(self.device)

                # -------------------
                # Train Discriminator
                # -------------------
                optimizer_D.zero_grad()

                # Discriminator loss on real latent codes
                real_output = self.discriminator(real_latent_code)
                real_labels = torch.ones_like(real_output).to(self.device)
                real_loss = criterion(real_output, real_labels)

                # Discriminator loss on fake latent codes
                fake_latent_code = self.generator(noise)
                fake_output = self.discriminator(fake_latent_code)
                fake_labels = torch.zeros_like(fake_output).to(self.device)
                fake_loss = criterion(fake_output, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
                optimizer_D.step()
                scheduler_D.step()

                total_d_loss += d_loss.item()

                # -------------------
                # Train Generator
                # -------------------
                optimizer_G.zero_grad()

                # Generate latent code and get discriminator output
                fake_latent_code = self.generator(noise)
                fake_output = self.discriminator(fake_latent_code)
                g_loss_adv = criterion(fake_output, real_labels)  # Adversarial loss

                # Decode the generated latent codes
                decoded = self.vae.model.decoder(fake_latent_code)
                decoded_positions = decoded[:, :, :2]
                decoded_expressions = decoded[:, :, 2:]

                # Ensure decoded positions are in the correct range
                # decoded_positions = (torch.tanh(decoded_positions) + 1) / 2
                # decoded_positions = 2*torch.sigmoid(decoded_positions[:, :, :2])-1

                # Compute position loss
                g_loss_position = chamfer_loss(positions, decoded_positions)

                # Compute expression loss
                g_loss_expression = compute_unordered_gene_expression_loss(positions, expressions, decoded_positions, decoded_expressions)

                # Total generator loss
                g_loss = g_loss_adv + g_loss_position + g_loss_expression

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_grad_norm)
                optimizer_G.step()
                scheduler_G.step()

                total_g_loss += g_loss.item()

            print(f"Epoch {epoch + 1}, D Loss: {total_d_loss / len(dataloader):.4f}, G Loss: {total_g_loss / len(dataloader):.4f}")

    def fill_region(self, adata, test_area, num_cells=50):
        with torch.no_grad():
            self.generator.eval()
            sampled_latents = torch.randn(num_cells, self.generator.model[0].in_features).to(self.device)
            
            # Generate latent codes using the generator
            generated_latents = self.generator(sampled_latents)
            
            # Use the VAE decoder to generate predicted positions and expressions
            decoded = self.vae.model.decoder(generated_latents)
            decoded_positions = decoded[:, :2]
            decoded_expressions = decoded[:, 2:]

            # Ensure decoded positions are in the correct range
            # decoded_positions = (torch.tanh(decoded_positions) + 1) / 2
            # decoded_positions = 2*torch.sigmoid(decoded_positions[:, :2])-1
            decoded_positions = normalize_positions(decoded_positions)

            predicted_positions = decoded_positions.cpu().numpy()
            predicted_expressions = decoded_expressions.cpu().numpy()
            
            predicted_positions[:, 0] = (predicted_positions[:, 0] + 1) / 2
            predicted_positions[:, 1] = (predicted_positions[:, 1] + 1) / 2

            # Scale predicted positions to fit within the test area
            predicted_positions[:, 0] = predicted_positions[:, 0] * (test_area.hole_max_x - test_area.hole_min_x) + test_area.hole_min_x
            predicted_positions[:, 1] = predicted_positions[:, 1] * (test_area.hole_max_y - test_area.hole_min_y) + test_area.hole_min_y

            return predicted_positions, predicted_expressions
