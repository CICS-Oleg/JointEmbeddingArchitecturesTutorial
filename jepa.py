import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----- Step 1: Sample 2D Gaussian Mixture Data -----
def sample_mog(n_samples=1000):
    means = [np.array([0, 0]), np.array([5, 5]), np.array([-5, 5])]
    covs = [
        np.array([[1, 0.8], [0.8, 1]]),
        np.array([[1, -0.6], [-0.6, 1]]),
        np.array([[0.5, 0], [0, 0.5]])
    ]
    weights = [0.4, 0.35, 0.25]
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    samples = np.array([
        np.random.multivariate_normal(means[k], covs[k])
        for k in components
    ])
    return samples.astype(np.float32)

# ----- Step 2: Define JEPA-style Encoder -----
class Encoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Step 3: Define JEPA Loss (Embedding Alignment) -----
def jepa_loss(encoder, x, x_tilde, normalize=True):
    z1 = encoder(x)
    z2 = encoder(x_tilde)

    if normalize:
        z1 = z1 / z1.norm(dim=1, keepdim=True)
        z2 = z2 / z2.norm(dim=1, keepdim=True)

    loss = torch.mean((z1 - z2) ** 2)  # L2 distance
    return loss

# ----- Step 4: Training -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(emb_dim=4).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

# Dataset
data = sample_mog(5000)
data_tensor = torch.tensor(data).to(device)

# Training loop
batch_size = 128
n_epochs = 1000
noise_std = 0.1

for epoch in range(n_epochs):
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,))
    x = data_tensor[idx]
    x_tilde = x + torch.randn_like(x) * noise_std  # small perturbation

    loss = jepa_loss(encoder, x, x_tilde)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ----- Step 5: Visualize Embedding Space -----
@torch.no_grad()
def plot_embeddings(encoder, data):
    encoder.eval()
    data_tensor = torch.tensor(data).to(device)
    embeddings = encoder(data_tensor).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', alpha=0.5, s=10)
    plt.title("JEPA-style Embedding of 2D Gaussian Mixture")
    plt.xlabel("Embedding dim 1")
    plt.ylabel("Embedding dim 2")
    plt.grid(True)
    plt.show()

# plot_embeddings(encoder, data)

# ----- Energy Function (Negative Cosine Similarity) -----
@torch.no_grad()
def energy(x, x_ref, encoder):
    # x: (N, 2)
    # x_ref: (1, 2)
    z = encoder(x)
    z_ref = encoder(x_ref)

    # Normalize
    z = z / z.norm(dim=1, keepdim=True)
    z_ref = z_ref / z_ref.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    sim = torch.sum(z * z_ref, dim=1)
    return -sim  # Energy = negative similarity

# ----- Visualize Energy Landscape w.r.t. One Reference Point -----
@torch.no_grad()
def plot_energy_landscape(encoder, reference_point, x_range=(-10, 10), y_range=(-5, 15), resolution=200):
    encoder.eval()
    x_vals = torch.linspace(*x_range, resolution)
    y_vals = torch.linspace(*y_range, resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    reference_tensor = torch.tensor(reference_point, dtype=torch.float32).unsqueeze(0).to(device)

    energies = energy(grid_points, reference_tensor, encoder)
    energies = energies.reshape(resolution, resolution).cpu()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu(), Y.cpu(), energies, levels=50, cmap='inferno')
    plt.colorbar(label='Energy (neg cosine)')
    plt.scatter(reference_point[0], reference_point[1], color='cyan', s=80, label='Reference point')
    plt.title('JEPA Energy Landscape (Negative Cosine Similarity)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----- Example Usage -----
# Pick a reference point from the dataset
# reference_point = data[0]  # could also use mean or any random point
#
# plot_energy_landscape(encoder, reference_point)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import os

# Use the same encoder and energy functions from before
# Ensure encoder is already trained

# ----- Energy Function -----
@torch.no_grad()
def energy(x, x_ref, encoder):
    z = encoder(x)
    z_ref = encoder(x_ref)

    z = z / z.norm(dim=1, keepdim=True)
    z_ref = z_ref / z_ref.norm(dim=1, keepdim=True)

    sim = torch.sum(z * z_ref, dim=1)
    return -sim  # Negative cosine similarity

# ----- Create Energy Frames for Animation -----
def generate_energy_frames(encoder, reference_points, x_range=(-10, 10), y_range=(-5, 15), resolution=200):
    encoder.eval()
    x_vals = torch.linspace(*x_range, resolution)
    y_vals = torch.linspace(*y_range, resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    frames = []
    for i, ref in enumerate(reference_points):
        ref_tensor = torch.tensor(ref, dtype=torch.float32).unsqueeze(0).to(device)
        energies = energy(grid_points, ref_tensor, encoder)
        energies = energies.reshape(resolution, resolution).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X.cpu(), Y.cpu(), energies, levels=50, cmap='inferno')
        ax.scatter(ref[0], ref[1], color='cyan', s=80, label=f'Ref {i}')
        ax.set_title(f'JEPA Energy - Frame {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.legend()
        plt.tight_layout()

        # Save frame to in-memory buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    return frames

# ----- Save Frames as GIF -----
def save_gif(frames, filename='jepa_energy_animation.gif', fps=10):
    import imageio
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved animation to {filename}")

# ----- Pick Reference Points -----
# Example: use first N points from dataset
N_FRAMES = 30
reference_points = data[np.linspace(0, len(data) - 1, N_FRAMES, dtype=int)]

# ----- Run Animation -----
frames = generate_energy_frames(encoder, reference_points)
save_gif(frames, 'jepa_energy.gif', fps=5)