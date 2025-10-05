import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# --- Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
# z_dim = 64
z_dim = 64
lr = 1e-3
# batch_size = 128
batch_size = 256
# n_epochs = 20
n_epochs = 100
ema_decay = 0.99
# ema_decay = 0.9
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
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

# for epoch in range(n_epochs):
#     encoder_online.train()
#     predictor.train()
#     decoder.train()
#
#     total_loss = 0.0
#     for images, _ in train_loader:
#         images = images.to(device)  # (B, 1, 28, 28)
#         B = images.size(0)
#         flat = images.view(B, -1)
#
#         # Create occluded context
#         occluded = random_occlusion_mask(images, mask_size, fill_value=0.0)
#         flat_occ = occluded.view(B, -1)
#
#         # Embeddings
#         z_context = encoder_online(flat_occ)  # embedding of occluded input
#         # Target embedding from full image via target encoder
#         with torch.no_grad():
#             z_target = encoder_target(flat)  # (B, z_dim)
#
#         # Prediction
#         z_pred = predictor(z_context)
#
#         # Loss: embedding alignment
#         loss_embed = nn.functional.mse_loss(z_pred, z_target)
#
#         # Optionally, reconstruction: reconstruct full image from predicted embedding
#         # We can reconstruct the masked part or full; here full
#         recon = decoder(z_pred)  # (B, img_size*img_size)
#         loss_recon = nn.functional.mse_loss(recon, flat)
#
#         # Total loss = embed + recon (with weight)
#         loss = loss_embed + 0.1 * loss_recon
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         update_ema(encoder_online, encoder_target, beta=ema_decay)
#
#         total_loss += loss.item() * B
#
#     avg_loss = total_loss / len(train)
#     print(f"Epoch {epoch}/{n_epochs} — Loss: {avg_loss:.6f}, Loss recon: {loss_recon:.6f}, Loss embed: {loss_embed:.6f}")

# --- Updated Training Loop: 20 epochs embed-only + 10 with reconstruction ---

# Updated hyperparams for stability
n_epochs_embed_only = 60
n_epochs_recon = 40
total_epochs = n_epochs_embed_only + n_epochs_recon

print_interval = 1  # print every N epochs

for epoch in range(total_epochs):
    encoder_online.train()
    predictor.train()
    decoder.train()

    total_loss = 0.0
    total_embed_loss = 0.0
    total_recon_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)  # (B, 1, 28, 28)
        B = images.size(0)
        flat = images.view(B, -1)

        # Create occluded context
        occluded = random_occlusion_mask(images, mask_size, fill_value=0.0)
        flat_occ = occluded.view(B, -1)

        # Embeddings
        z_context = encoder_online(flat_occ)
        with torch.no_grad():
            z_target = encoder_target(flat)

        # Prediction
        z_pred = predictor(z_context)

        # Embedding loss
        loss_embed = nn.functional.mse_loss(z_pred, z_target)

        # Optional: reconstruction from predicted embedding
        if epoch >= n_epochs_embed_only:
            recon = decoder(z_pred)
            loss_recon = nn.functional.mse_loss(recon, flat)
        else:
            loss_recon = torch.tensor(0.0, device=device)

        # Total loss
        loss = loss_embed + (0.1 * loss_recon if epoch >= n_epochs_embed_only else 0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema(encoder_online, encoder_target, beta=ema_decay)

        total_loss += loss.item() * B
        total_embed_loss += loss_embed.item() * B
        total_recon_loss += loss_recon.item() * B

    avg_loss = total_loss / len(train)
    avg_embed_loss = total_embed_loss / len(train)
    avg_recon_loss = total_recon_loss / len(train)

    if (epoch + 1) % print_interval == 0:
        print(f"[Epoch {epoch+1}/{total_epochs}] "
              f"Total Loss: {avg_loss:.6f} | "
              f"Embed: {avg_embed_loss:.6f} | "
              f"Recon: {avg_recon_loss:.6f}")


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


# --- Extract embeddings from train and test sets ---
def get_embeddings_and_labels(encoder, dataloader):
    encoder.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            flat = images.view(images.size(0), -1)
            z = encoder(flat)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)

# Get embeddings
print("\nExtracting embeddings for k-NN evaluation...")

z_train, y_train = get_embeddings_and_labels(encoder_online, train_loader)
z_test, y_test = get_embeddings_and_labels(encoder_online, test_loader)

# --- Fit k-NN classifier ---
k = 5
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
knn.fit(z_train, y_train)

# Predict on test set
y_pred = knn.predict(z_test)

# --- Evaluation Metrics ---
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"\n🔍 k-NN Evaluation (k={k}):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix — k-NN Classifier (k={k})")
plt.tight_layout()
plt.show()