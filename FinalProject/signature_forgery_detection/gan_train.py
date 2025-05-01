# gan_train.py - Train a GAN to generate synthetic signature images
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 128 * 128),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 128, 128)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

import os
print("Looking in:", os.path.abspath("data/genuine"))

#check
dataset = datasets.ImageFolder("C:/Users/user/Desktop/project_root/signature_forgery/data/genuine", transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(10):
    for real, _ in dataloader:
        real = real.to(device)
        b_size = real.size(0)
        real_labels = torch.ones(b_size, 1).to(device)
        fake_labels = torch.zeros(b_size, 1).to(device)

        noise = torch.randn(b_size, 100).to(device)
        fake = G(noise)

        d_real_loss = loss_fn(D(real), real_labels)
        d_fake_loss = loss_fn(D(fake.detach()), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        g_loss = loss_fn(D(fake), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    if not os.path.exists("generated"):
        os.makedirs("generated")
    for i in range(5):
        torch.save(fake[i], f"generated/sample_{epoch+1}_{i}.pt")

torch.save(G.state_dict(), "generator.pth")
