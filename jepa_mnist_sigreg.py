import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import warnings
from skimage.morphology import skeletonize
import lejepa


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

# ---------------- DATA PERTURBATION ----------------
def prepare_transforms(device, batch_size, coeff=1.0):
    H = W =        28
    rotation_deg = coeff * 30
    shear_deg =    coeff * 15
    translate =   (coeff * 0.15, coeff * 0.15)
    scale_range = (1.0 - coeff * 0.25, 1.0 + coeff * 0.1)
    
    angles = torch.empty(batch_size, device=device).uniform_(-rotation_deg, rotation_deg) * (3.14159 / 180)
    tx = torch.empty(batch_size, device=device).uniform_(-translate[0], translate[0]) * W
    ty = torch.empty(batch_size, device=device).uniform_(-translate[1], translate[1]) * H
    s = torch.empty(batch_size, device=device).uniform_(1/scale_range[1], 1/scale_range[0])
    sh = torch.empty(batch_size, device=device).uniform_(-shear_deg, shear_deg) * (3.14159 / 180)
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = s * torch.cos(angles)
    theta[:, 0, 1] = -s * torch.sin(angles + sh)
    theta[:, 1, 0] = s * torch.sin(angles)
    theta[:, 1, 1] = s * torch.cos(angles + sh)
    theta[:, 0, 2] = tx / (W - 1)
    theta[:, 1, 2] = ty / (H - 1)
    grid = F.affine_grid(theta, (batch_size, 1, 28, 28), align_corners=True)
    return grid
    
def morph_perturb_image(img, coeff=1.0):
    B, C, H, W = img.shape
    device = img.device

    kernel_size = 3
    pad = kernel_size // 2
    perturbed = torch.empty_like(img)

    # ---- Affine transform ----
    tforms = prepare_transforms(device, B, coeff)
    perturbed = F.grid_sample(img, tforms, padding_mode="zeros", align_corners=True)
    # ---- Randomly choose none/erosion/dilation ----
    choice = torch.randint(0, 3, (B,), device=device)
    # ---- Randomly choose inversion ----
    choice2 = torch.randint(0, 2, (B,), device=device) == 1
    for i in range(B):
        if choice[i] == 1:  # erosion            
            #-F.max_pool2d(-x, 2, stride=1, padding=1)
            xn = perturbed[i].cpu().numpy()
            x_bin = (xn > 0.5).astype(np.uint8)
            x_skel=skeletonize(x_bin).astype(np.float32)
            perturbed[i:i+1,0] = torch.tensor(x_skel).to(device)
        elif choice[i] == 2:  # dilation
            x_morph = F.max_pool2d(perturbed[i], kernel_size, stride=1, padding=pad)
            perturbed[i:i+1] = x_morph
        if choice2[i]:
            perturbed[i:i+1] = 1 - perturbed[i:i+1]

    # ---- Add noise ----
    noise_std = 0.3
    noise = torch.randn_like(perturbed) * noise_std
    perturbed = torch.clamp(perturbed + noise, 0.0, 1.0)

    return perturbed
    
# ---------------- ARCHITECTURE ----------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1,1)),   # global pooling
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.ln = nn.LayerNorm(latent_dim, eps=1e-6)

    def forward(self, x):
        z = self.net(x)
        z = self.ln(z)
        return z
    
class Predictor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z): return self.net(z)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        return self.deconv(x)

class JEPA(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target(self, tau=0.99):
        for o, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            t.data = tau * t.data + (1 - tau) * o.data

    def forward(self, img, masked_img):
        z_online = self.encoder(masked_img)
        z_pred = z_online# self.predictor(z_online)
        #with torch.no_grad():
        z_target = self.encoder(img)
        return z_online, z_pred, z_target

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRAINING PARAMS ----------------
latent_dim = 128
lr = 1e-3
tau = 0.999
model = JEPA(latent_dim=latent_dim).to(device)
epochs_phase1 = 25
epochs_phase2 = 15
batch_size = 32

# ---------------- DATA ----------------
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# ---- Phase 1: JEPA Embedding Training ----

# Choose a univariate test (Epps-Pulley in this example)
univariate_test = lejepa.univariate.EppsPulley(n_points=17)

# Create the multivariate slicing test
sigreg = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test, 
    num_slices=32
)

opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=lr)

for epoch in range(epochs_phase1):
    model.train()
    total_loss = 0
    iters = 0

    for img, _ in train_loader:
        if img.shape[0] != batch_size:
            continue
        img = img.to(device)
        coeff = epoch / epochs_phase1
        perturbed_img = morph_perturb_image(img, 1.0).to(device)

        emb, z_pred, z_target = model(img, perturbed_img)
        z_target = z_target.detach()

        emb_n = F.normalize(emb + 1e-8, dim=1)
        z_pred_n = F.normalize(z_pred + 1e-8, dim=1)
        z_target_n = F.normalize(z_target + 1e-8, dim=1)
        loss = (1.0 - F.cosine_similarity(z_pred_n, z_target_n, dim=1)).mean() + sigreg(emb) * 0.5

        opt.zero_grad()
        loss.backward()
        opt.step()
        #model.update_target(tau)
        
        total_loss += loss.item()
        iters += 1

    avg_loss = total_loss / max(1, iters)
    
    print(f"[Phase 1: Embedding] Epoch {epoch+1}/{epochs_phase1} | Avg Loss: {avg_loss:.6f}")

# --- EMBEDDING VISUALIZATION ---
model.encoder.eval()
all_z = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        z = model.encoder(images)
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())
all_z = np.concatenate(all_z, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
pca = PCA(n_components=2)
z2 = pca.fit_transform(all_z)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(z2[:, 0], z2[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("2D PCA of JEPA Embeddings on MNIST")
plt.show()

# --- K-NN EVALUATION ---
print("\nExtracting embeddings for k-NN evaluation...")
def get_embeddings_and_labels(encoder, dataloader):
    encoder.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            z = encoder(images)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)
z_train, y_train = get_embeddings_and_labels(model.encoder, train_loader)
z_test, y_test = get_embeddings_and_labels(model.encoder, test_loader)
k = 5
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
knn.fit(z_train, y_train)
y_pred = knn.predict(z_test)
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"\n🔍 k-NN Evaluation (k={k}):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ---- Phase 2: Reconstruction Fine-tuning ----
print("🔧 Switching to reconstruction fine-tuning phase...")

opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=lr)

for epoch in range(epochs_phase2):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    iters = 0

    for img, _ in train_loader:
        if img.shape[0] != batch_size:
            continue
        img = img.to(device)
        perturbed_img = morph_perturb_image(img).to(device)

        emb, z_pred, z_target = model(img, perturbed_img)
        decoded = model.decoder(z_pred)
        loss = F.mse_loss(decoded, img)

        opt_dec.zero_grad()
        loss.backward()
        opt_dec.step()

        total_loss += loss.item()
        iters += 1

    avg_loss = total_loss / iters
    print(f"[Phase 2: Reconstruction] Epoch {epoch+1}/{epochs_phase2} | Loss: {avg_loss:.4f}")

# ---- RECONSTRUCTION EXAMPLES ----
model.eval()
with torch.no_grad():
    img, _ = next(iter(test_loader))
    img = img.to(device)
    perturbed_img = morph_perturb_image(img).to(device)
    emb, z_pred, _, = model(img, perturbed_img)
    decoded = model.decoder(z_pred)
n = 6
plt.figure(figsize=(n * 2, 6))
for i in range(n):
    plt.subplot(3, n, i+1)
    plt.imshow(perturbed_img[i].cpu().squeeze(), cmap="gray")
    plt.axis("off"); 
    if i == 0: plt.ylabel("Perturbed")

    plt.subplot(3, n, i+n+1)
    plt.imshow(decoded[i].cpu().squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0: plt.ylabel("Reconstructed")

    plt.subplot(3, n, i+2*n+1)
    plt.imshow(img[i].cpu().squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0: plt.ylabel("Original")
plt.suptitle("JEPA Reconstruction", fontsize=14)
plt.tight_layout()
plt.show()