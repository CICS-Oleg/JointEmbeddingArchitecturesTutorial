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

# ----- Step 3: Energy Loss with Regularization -----
def energy_regularized_loss(model, data_batch, lambda_reg=1.0):
    data_batch.requires_grad_(True)  # enable gradients w.r.t inputs
    energy = model(data_batch)

    # Compute gradients of energy w.r.t inputs
    grad_outputs = torch.ones_like(energy)
    grads = torch.autograd.grad(outputs=energy, inputs=data_batch,
                                grad_outputs=grad_outputs,
                                create_graph=True, retain_graph=True)[0]

    # Gradient norm squared (||∇E||²)
    grad_norm_sq = torch.sum(grads ** 2, dim=1)

    # Loss: energy on data + lambda * gradient penalty
    loss = torch.mean(energy) + lambda_reg * torch.mean(grad_norm_sq)
    return loss

def dsm_loss(model, data_batch, sigma=0.5):
    noise = torch.randn_like(data_batch) * sigma
    noisy_data = data_batch + noise
    noisy_data.requires_grad_(True)

    energy = model(noisy_data)
    grads = torch.autograd.grad(outputs=energy, inputs=noisy_data,
                                grad_outputs=torch.ones_like(energy),
                                create_graph=True)[0]
    target = -noise / (sigma ** 2)
    loss = torch.mean((grads - target) ** 2)
    return loss

# ----- Step 4: Training Loop -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnergyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dataset
data = sample_mog(5000)
data_tensor = torch.tensor(data).to(device)

# Training
batch_size = 128
n_epochs = 1000
# lambda_reg = 1.0  # gradient regularization strength
lambda_reg = 10

for epoch in range(n_epochs):
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,))
    batch = data_tensor[idx]

    # loss = energy_regularized_loss(model, batch, lambda_reg)
    loss = dsm_loss(model, batch, sigma=1)

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
    plt.title('Energy Function with Regularization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(data[:, 0], data[:, 1], s=1, color='cyan', alpha=0.3, label='Data')
    plt.legend()
    plt.show()

plot_energy(model)