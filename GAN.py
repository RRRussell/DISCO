import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    # def __init__(self, input_channels, output_channels):
    #     super().__init__()
    #     self.model = nn.Sequential(
    #         nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 64, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
    #         nn.Sigmoid()
    #     )

    # def forward(self, x):
    #     return self.model(x)
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    # def __init__(self, input_channels):
    #     super().__init__()
    #     self.model = nn.Sequential(
    #         nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 64, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 1, kernel_size=3, padding=1),
    #         nn.Sigmoid()
    #     )

    # def forward(self, x):
    #     return self.model(x).view(x.size(0), -1).mean(1, keepdim=True)
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=1)  # Average across the 50 samples dimension
        return x
    

# def train_gan(generator, discriminator, dataloader, num_epochs=100):
#     criterion = nn.BCELoss()
#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
    
#     for epoch in range(num_epochs):
#         for real_data in dataloader:
#             # Train discriminator
#             optimizer_d.zero_grad()
#             real_labels = torch.ones(real_data.size(0), 1)
#             fake_labels = torch.zeros(real_data.size(0), 1)

#             outputs = discriminator(real_data)
#             d_loss_real = criterion(outputs, real_labels)

#             noise = torch.randn(real_data.size(0), input_dim)
#             fake_data = generator(noise)
#             outputs = discriminator(fake_data.detach())
#             d_loss_fake = criterion(outputs, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             # Train generator
#             optimizer_g.zero_grad()
#             outputs = discriminator(fake_data)
#             g_loss = criterion(outputs, real_labels)
#             g_loss.backward()
#             optimizer_g.step()
        
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    
#     return generator, discriminator


# def generate_missing_values(generator, missing_data_shape):
#     noise = torch.randn(missing_data_shape[0], input_dim)
#     generated_data = generator(noise)
#     return generated_data.detach().numpy()


