import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
import random
#EnvironmentSetUp
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed()

#Dataset Preparation
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(28),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])  # Normalize to range [-1, 1]
                   ])),
    batch_size=64, shuffle=True)

#Genratrot Network
class Generator(nn.Module):
    """
    Generator class for GAN.
    Takes a latent vector (size 100) as input and produces an image (size 784, i.e., 28x28) as output.
    Aims to generate fake images indistinguishable from real MNIST digits.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # Outputs in range [-1, 1] to match the preprocessed MNIST images
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # Reshape to image dimensions

class Discriminator(nn.Module):
    """
    Discriminator class for GAN.
    Takes an image (size 784, i.e., 28x28) as input and predicts whether the image is real or generated.
    Outputs a probability value between 0 and 1.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid activation to output probabilities
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

#Initialization
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Number of epochs
num_epochs = 100

# For visualizing the progress
fixed_noise = torch.randn(64, 100, device=device)

os.makedirs('./images', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        if epoch == 0 and i == 0:
            sample_real_images = real_images # to keep batch_size

        batch_size = real_images.size(0)

        # Real labels are 1, fake labels are 0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator with real images
        discriminator.zero_grad()
        outputs = discriminator(real_images.to(device))
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Train Discriminator with fake images
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize for discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize for generator
        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

    # Save real images once for comparison
    if epoch == 0:
        save_image(sample_real_images, './images/real_images.png')

    # Save sampled images every epoch
    fake_images = generator(fixed_noise)
    save_image(fake_images, f'./images/fake_images_epoch_{epoch+1:04d}.png')

    # Save discriminator results for each image in the grid
    discriminator_results = discriminator(fake_images).cpu().detach().numpy().reshape(8, 8)
    np.save(f'./results/discriminator_outputs_epoch_{epoch+1:04d}.npy', discriminator_results)

print('Training finished.')