import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import MNIST
import timm
import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=0,         
            drop_path_rate=0.1,
            img_size=128,          
        )
        self.proj = MLP(384, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        return emb, proj

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

    tforms = prepare_transforms(device, B, coeff)
    perturbed = F.grid_sample(img, tforms, padding_mode="zeros", align_corners=True)
    choice = torch.randint(0, 3, (B,), device=device)
    choice2 = torch.randint(0, 2, (B,), device=device) == 1
    for i in range(B):
        if choice[i] == 1:  # erosion            
            xn = perturbed[i].cpu().numpy()
            x_bin = (xn > 0.5).astype(np.uint8)
            x_skel=skeletonize(x_bin).astype(np.float32)
            perturbed[i:i+1,0] = torch.tensor(x_skel).to(device)
        elif choice[i] == 2:  # dilation
            x_morph = F.max_pool2d(perturbed[i], kernel_size, stride=1, padding=pad)
            perturbed[i:i+1] = x_morph
        if choice2[i]:
            perturbed[i:i+1] = 1 - perturbed[i:i+1]

    noise_std = 0.3
    noise = torch.randn_like(perturbed) * noise_std
    perturbed = torch.clamp(perturbed + noise, 0.0, 1.0)

    return perturbed

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, V=2, coeff=1.0):
        self.V = V
        self.coeff = coeff
        self.train = train
        self.ds = MNIST(root="./mnist", train=train, download=True)
        self.mean = torch.tensor([0.1307])
        self.std  = torch.tensor([0.3081])
        self.resize = v2.Resize((128, 128))
        self.to3 = v2.Grayscale(num_output_channels=3)

    def __getitem__(self, i):
        img, label = self.ds[i]
        img = v2.ToImage()(img)          # -> (1,28,28)
        img = img.float() / 255.0
        views = []
        if self.train:
            for _ in range(self.V):
                aug = morph_perturb_image(img.unsqueeze(0), coeff=self.coeff)[0]  # (1,28,28)
                aug = self.resize(aug)
                aug = self.to3(aug)
                aug = (aug - self.mean) / self.std
                views.append(aug)
        else:
            aug = self.resize(img)
            aug = self.to3(aug)
            aug = (aug - self.mean) / self.std
            views.append(aug)

        return torch.stack(views), label

    def __len__(self):
        return len(self.ds)

def get_proj_embeddings(model, loader):
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader):
            images = images.to("cuda")
            _, proj = model(images)
            z = proj.mean(0).cpu()
            all_z.append(z)
            all_labels.append(labels)
    return torch.cat(all_z).numpy(), torch.cat(all_labels).numpy()

def visualize_pca_embeddings(encoder, dataloader, save_path="pca_test_embeddings.png"):
    encoder.eval()    
    X, y = get_proj_embeddings(encoder, dataloader)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("2D PCA of Test Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] PCA visualization saved to {save_path}")

class CFG:
    def __init__(self):
        self.V = 4
        self.bs = 64
        self.proj_dim = 32
        self.epochs = 50
        self.lr = 1e-3
        self.lamb = 0.1

def main():
    torch.manual_seed(0)
    cfg = CFG()

    train_ds = MNISTDataset(train=True, V=cfg.V)
    test_ds = MNISTDataset(train=False, V=1)

    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=4)
    test  = DataLoader(test_ds,  batch_size=256, num_workers=4)

    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")

    probe = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 10)).to("cuda")

    sigreg = SIGReg().to("cuda")

    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, [s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=True)

    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            vs = vs.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)

                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

                y_rep = y.repeat_interleave(cfg.V)
                yhat = probe(emb.detach())

                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        net.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to("cuda")
                y = y.to("cuda")
                with autocast("cuda", dtype=torch.bfloat16):
                    logits = probe(net(vs)[0])
                correct += (logits.argmax(1) == y).sum().item()

        print(f"Epoch {epoch}: probe_loss={probe_loss.item():.4f}, "
              f"acc={correct/len(test_ds):.4f}")

    print("\nExtracting embeddings for k-NN evaluation...")

    z_train, y_train = get_proj_embeddings(net, train)
    z_test,  y_test  = get_proj_embeddings(net, test)

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(z_train, y_train)
    y_pred = knn.predict(z_test)

    print(f"\nKNN Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    visualize_pca_embeddings(net, test)
    pass


if __name__ == "__main__":
    main()
