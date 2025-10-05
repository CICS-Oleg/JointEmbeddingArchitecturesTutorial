import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
z_dim = 64
lr = 1e-3
# batch_size = 128
batch_size = 4096
# n_epochs = 20
n_epochs = 120
# ema_decay = 0.99
ema_decay = 0.9
# mask_size = (14, 14)  # mask a 14x14 square in 28x28 MNIST
mask_size = (10, 10)
img_size = 28

# --- Datasets & Dataloader ---
transform = transforms.Compose([
    transforms.ToTensor(),  # yields [0,1]
])
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test, batch_size=256, shuffle=False)

# --- Models ---

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, z_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, z):
        return self.net(z)

class SimpleDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, out_dim=img_size * img_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()  # for pixel values [0,1]
        )

    def forward(self, z):
        return self.net(z)

# Instantiate
input_dim = img_size * img_size
encoder_online = MLPEncoder(input_dim, hidden_dim=256, z_dim=z_dim).to(device)
encoder_target = MLPEncoder(input_dim, hidden_dim=256, z_dim=z_dim).to(device)
predictor = Predictor(z_dim, hidden_dim=128).to(device)
decoder = SimpleDecoder(z_dim, hidden_dim=256, out_dim=input_dim).to(device)

# Freeze target encoder grad
for p in encoder_target.parameters():
    p.requires_grad = False

optimizer = optim.Adam(list(encoder_online.parameters()) + list(predictor.parameters()) + list(decoder.parameters()), lr=lr)

# EMA update
@torch.no_grad()
def update_ema(online, target, beta):
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data = beta * p_target.data + (1 - beta) * p_online.data

# --- Masking / Occlusion function ---

def random_occlusion_mask(batch_tensor, mask_size, fill_value=0.0):
    # batch_tensor: (B, 1, H, W)
    B, C, H, W = batch_tensor.shape
    mh, mw = mask_size
    out = batch_tensor.clone()
    # For each in batch, choose a random top-left corner
    for i in range(B):
        top = np.random.randint(0, H - mh + 1)
        left = np.random.randint(0, W - mw + 1)
        out[i, :, top:top+mh, left:left+mw] = fill_value
    return out

# --- Training Loop ---

for epoch in range(n_epochs):
    encoder_online.train()
    predictor.train()
    decoder.train()

    total_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)  # (B, 1, 28, 28)
        B = images.size(0)
        flat = images.view(B, -1)

        # Create occluded context
        occluded = random_occlusion_mask(images, mask_size, fill_value=0.0)
        flat_occ = occluded.view(B, -1)

        # Embeddings
        z_context = encoder_online(flat_occ)  # embedding of occluded input
        # Target embedding from full image via target encoder
        with torch.no_grad():
            z_target = encoder_target(flat)  # (B, z_dim)

        # Prediction
        z_pred = predictor(z_context)

        # Loss: embedding alignment
        loss_embed = nn.functional.mse_loss(z_pred, z_target)

        # Optionally, reconstruction: reconstruct full image from predicted embedding
        # We can reconstruct the masked part or full; here full
        recon = decoder(z_pred)  # (B, img_size*img_size)
        loss_recon = nn.functional.mse_loss(recon, flat)

        # Total loss = embed + recon (with weight)
        loss = loss_embed + 0.1 * loss_recon

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema(encoder_online, encoder_target, beta=ema_decay)

        total_loss += loss.item() * B

    avg_loss = total_loss / len(train)
    print(f"Epoch {epoch}/{n_epochs} — Loss: {avg_loss:.6f}")

# --- After Training: Embedding Visualization ---

encoder_online.eval()
all_z = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        flat = images.view(images.size(0), -1)
        z = encoder_online(flat)  # get embedding
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())
all_z = np.concatenate(all_z, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Use PCA to reduce to 2D
pca = PCA(n_components=2)
z2 = pca.fit_transform(all_z)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z2[:, 0], z2[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("2D PCA of JEPA Embeddings on MNIST")
plt.show()

# --- Visualize reconstructions ---
# Pick some samples
n_show = 8
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    flat = images.view(images.size(0), -1)
    occl = random_occlusion_mask(images, mask_size, fill_value=0.0).to(device)
    flat_occ = occl.view(occl.size(0), -1)
    z_context = encoder_online(flat_occ)
    z_pred = predictor(z_context)
    recons = decoder(z_pred).view(-1, 1, img_size, img_size).cpu()

# Plot original, occluded, reconstruction
fig, axs = plt.subplots(n_show, 3, figsize=(6, 2*n_show))
for i in range(n_show):
    axs[i, 0].imshow(images[i, 0].cpu(), cmap='gray')
    axs[i, 0].set_title("Original")
    axs[i, 1].imshow(occl[i, 0].cpu(), cmap='gray')
    axs[i, 1].set_title("Occluded")
    axs[i, 2].imshow(recons[i, 0], cmap='gray')
    axs[i, 2].set_title("Reconstructed")
    for j in range(3):
        axs[i, j].axis('off')
plt.tight_layout()
plt.show()
