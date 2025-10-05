import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import imageio
import itertools

# ----- Gaussian Mixture Data -----
def sample_mog(n_samples=1000):
    means = [np.array([0, 0]), np.array([5, 3.5]), np.array([-5, 7])]
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

# ----- Encoder -----
class Encoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Output: scalar energy
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Predictor -----
class Predictor(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Output: scalar energy
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Masked JEPA Loss with predictor -----
def jepa_masked_loss(encoder_online, encoder_target, predictor, x, mask_coord=1, normalize=True):
    """
    mask_coord: 0 -> mask x, predict x from y
                1 -> mask y, predict y from x
    """
    x_context = x.clone()
    x_target = x.clone()

    x_context[:, mask_coord] = 0.0  # Masked input to online encoder
    x_target[:, 1 - mask_coord] = 0.0  # Masked input to target encoder

    z_context = encoder_online(x_context)
    z_target = encoder_target(x_target).detach()

    z_predict = predictor(z_context)

    if normalize:
        z_predict = z_predict / z_predict.norm(dim=1, keepdim=True)
        z_target = z_target / z_target.norm(dim=1, keepdim=True)

    return ((z_predict - z_target) ** 2).mean()

# ----- Masked pairwise JEPA Loss with predictor -----
def jepa_masked_pairwise_loss(encoder_online, encoder_target, predictor, x, mask_coord=1, normalize=True):
    """
    mask_coord: 0 = mask x (predict x from y), 1 = mask y (predict y from x)
    """
    x_context = x.clone()
    x_target = x.clone()

    x_context[:, mask_coord] = 0.0
    # x_target[:, 1 - mask_coord] = 0.0

    z_context = encoder_online(x_context)  # (B, D)
    z_target = encoder_target(x_target).detach()  # (B, D)

    z_predict = predictor(z_context)

    if normalize:
        z_predict = z_predict / z_predict.norm(dim=1, keepdim=True)
        z_target = z_target / z_target.norm(dim=1, keepdim=True)

    # Pairwise squared distance matrix
    diff = z_predict.unsqueeze(1) - z_target.unsqueeze(0)  # (B, B, D)
    dist_matrix = (diff ** 2).sum(dim=2)  # (B, B)
    return dist_matrix.mean()

# ----- EMA Update Function -----
@torch.no_grad()
def update_ema(online_model, target_model, beta=0.99):
    for param_online, param_target in zip(online_model.parameters(), target_model.parameters()):
        param_target.data = beta * param_target.data + (1 - beta) * param_online.data

# ----- Training Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = sample_mog(5000)
data_tensor = torch.tensor(data).to(device)

# Models
emb_dim = 64
encoder_online = Encoder(emb_dim=emb_dim).to(device)
encoder_target = copy.deepcopy(encoder_online).to(device)
predictor      = Predictor(emb_dim=emb_dim).to(device)

# Ensure target encoder doesn't require gradients
for p in encoder_target.parameters():
    p.requires_grad = False

optimizer = optim.Adam(itertools.chain(encoder_online.parameters(), predictor.parameters()), lr=1e-3)

# Training loop
batch_size = 128
n_epochs = 5000
noise_std = 0.01
ema_decay = 0.99

for epoch in range(n_epochs):
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,))
    x = data_tensor[idx]
    x_tilde = x + torch.randn_like(x) * noise_std
    
    mask_coord = torch.randint(0, 2, (1,)).item()
    #loss = jepa_masked_loss2(encoder_online, encoder_target, predictor, x, mask_coord=mask_coord)
    loss = jepa_masked_pairwise_loss(encoder_online, encoder_target, predictor, x, mask_coord=mask_coord)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # EMA update for target encoder
    update_ema(encoder_online, encoder_target, beta=ema_decay)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ----- Energy Function (JEPA-style) -----
@torch.no_grad()
def energy_grid(encoder_online, encoder_target, grid_points, reference_point):
    z = encoder_online(grid_points)
    z_ref = encoder_target(reference_point)

    z = z / z.norm(dim=1, keepdim=True)
    z_ref = z_ref / z_ref.norm(dim=1, keepdim=True)

    cos_sim = torch.sum(z * z_ref, dim=1)
    return -cos_sim  # Negative cosine similarity = energy

# ----- Frame Generation -----
@torch.no_grad()
def generate_energy_frames(encoder_online, encoder_target, reference_points,
                           data=None, x_range=(-10, 10), y_range=(-5, 15), resolution=200):
    x_vals = torch.linspace(*x_range, resolution)
    y_vals = torch.linspace(*y_range, resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    frames = []

    for i, ref in enumerate(reference_points):
        ref_tensor = torch.tensor(ref, dtype=torch.float32).unsqueeze(0).to(device)
        energy_vals = energy_grid(encoder_online, encoder_target, grid_points, ref_tensor)
        energy_map = energy_vals.reshape(resolution, resolution).cpu().numpy()

        # Plot energy contour
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contourf(X.cpu(), Y.cpu(), energy_map, levels=50, cmap='inferno')
        if data is not None:
            ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, color='white', label="Data")
        ax.scatter(ref[0], ref[1], color='cyan', s=80, label='Reference')
        ax.set_title(f"JEPA Energy Frame {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.legend()
        plt.tight_layout()

        # Convert to image buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    return frames

# ----- Save Frames as GIF -----
def save_gif(frames, filename="jepa_energy.gif", fps=5):
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved animation to {filename}")

# ----- Reference Points -----
# Sample a smooth path through the data space
n_frames = 30
reference_points = data[np.linspace(0, len(data) - 1, n_frames, dtype=int)]

# ----- Generate and Save Animation -----
frames = generate_energy_frames(
    encoder_online=encoder_online,
    encoder_target=encoder_online,
    reference_points=reference_points,
    data=data,  # optional: plot real data in background
    x_range=(-10, 10),
    y_range=(-5, 15),
    resolution=200
)

save_gif(frames, filename="jepa_energy.gif", fps=5)