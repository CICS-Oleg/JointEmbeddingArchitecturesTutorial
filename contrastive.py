import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----- Step 1: Generate Mixture of Gaussians Data -----
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

# ----- Step 2: Define Energy MLP -----
class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Output: scalar energy
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()  # shape: (batch,)

# ----- Step 3: Contrastive Loss (Joint Embedding Training Style) -----
def energy_contrastive_loss(model, real_data, neg_data):
    E_real = model(real_data)  # low energy
    E_neg = model(neg_data)    # high energy

    # Simple contrastive loss: margin between positive and negative energy
    loss = torch.mean(E_real) + torch.mean(torch.exp(-E_neg))
    return loss

# ----- Step 4: Training Loop -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EnergyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Generate dataset
data = sample_mog(5000)
data_tensor = torch.tensor(data).to(device)

# Training
batch_size = 128
n_epochs = 1000

for epoch in range(n_epochs):
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,))
    real_batch = data_tensor[idx]

    # Negative samples: Uniform noise
    neg_batch = torch.empty_like(real_batch).uniform_(-10, 10)

    loss = energy_contrastive_loss(model, real_batch, neg_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ----- Step 5: Visualize Energy Landscape -----
def plot_energy(model):
    model.eval()
    x = torch.linspace(-10, 10, 200)
    y = torch.linspace(-5, 15, 200)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        energies = model(coords).reshape(200, 200).cpu()

    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu(), Y.cpu(), energies, levels=50, cmap='inferno')
    plt.colorbar(label='Energy')
    plt.title('Learned Energy Function by contrastive training')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(data[:, 0], data[:, 1], s=1, color='cyan', alpha=0.3, label='Data')
    plt.legend()
    plt.show()

plot_energy(model)