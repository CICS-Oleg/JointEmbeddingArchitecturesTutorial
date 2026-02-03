import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 111):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def column_unit_norm_(W: torch.Tensor, eps: float = 1e-12):
    col_norm = torch.sqrt((W * W).sum(dim=0, keepdim=True)).clamp_min(eps)
    return W / col_norm

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def ols_fit_predict(X_train, Y_train, X_test):
    Xtr = np.asarray(X_train)
    Ytr = np.asarray(Y_train)
    Xte = np.asarray(X_test)

    Xtr_aug = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte_aug = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)

    W = np.linalg.lstsq(Xtr_aug, Ytr, rcond=None)[0]
    return Xte_aug @ W

def env_snr_db(x_signal: torch.Tensor, x_noise: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Environment SNR in observation space (dB):
      SNR = Var(signal part of x) / Var(noise part of x)
    """
    sig = x_signal.detach().cpu().numpy().reshape(-1)
    noi = x_noise.detach().cpu().numpy().reshape(-1)
    return float(10.0 * np.log10((np.var(sig) + eps) / (np.var(noi) + eps)))


# ----------------------------
# Environment: Noisy-TV Linear-Gaussian system (with stable signal)
# ----------------------------
def make_rotation_matrix(Ds, omega, device):
    """
    Build a block-diagonal rotation matrix with frequency omega.
    Ds must be even.
    """
    assert Ds % 2 == 0
    R = torch.zeros(Ds, Ds, device=device)

    for i in range(0, Ds, 2):
        c = torch.cos(torch.tensor(omega, device=device))
        s = torch.sin(torch.tensor(omega, device=device))
        R[i:i+2, i:i+2] = torch.tensor([[c, -s], [s, c]], device=device)

    return R

@torch.no_grad()
def rollout_noisy_tv(
    Dx=20, Ds=4, Dd=4, T=8000, sigma=0.0,
    a_scale=0.99,        # NEW: contraction factor to stabilize variance
    w_std=0.3,           # NEW: increase signal process noise (paper used 0.1)
    v_std=0.3,           # distractor process noise (paper: 0.3)
    eps_std=0.01,        # observation noise (paper: 0.01)
    device="cpu", seed=111
):
    """
    s_{t+1} = (a_scale * Q) s_t + w_t
    d_{t+1} = 0.9 d_t + v_t
    x_t     = C s_t + D (sigma d_t) + eps_t
    C, D column-wise unit norm
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # orthogonal base rotation Q
    M = torch.randn(Ds, Ds, generator=g, device=device)
    #Q, _ = torch.linalg.qr(M)
    #if torch.linalg.det(Q) < 0:
    #    Q[:, 0] *= -1
    #A = a_scale * Q  # contraction * rotation
    
    omega = 0.1 * math.pi    # LOW frequency
    # omega = 0.5 * math.pi   # MEDIUM
    # omega = 0.8 * math.pi   # HIGH

    Q = make_rotation_matrix(Ds, omega, device)
    A = a_scale * Q

    C = column_unit_norm_(torch.randn(Dx, Ds, generator=g, device=device))
    D = column_unit_norm_(torch.randn(Dx, Dd, generator=g, device=device))

    s = torch.zeros(T, Ds, device=device)
    d = torch.zeros(T, Dd, device=device)
    x = torch.zeros(T, Dx, device=device)

    # Keep the parts of x for environment SNR measurement
    x_sig = torch.zeros(T, Dx, device=device)   # C s_t
    x_noi = torch.zeros(T, Dx, device=device)   # D(sigma d_t) + eps

    s_prev = torch.randn(Ds, generator=g, device=device)
    d_prev = torch.randn(Dd, generator=g, device=device)

    for t in range(T):
        eps = eps_std * torch.randn(Dx, generator=g, device=device)
        sig_part = C @ s_prev
        noi_part = D @ (sigma * d_prev) + eps

        x[t] = sig_part + noi_part
        x_sig[t] = sig_part
        x_noi[t] = noi_part

        s[t] = s_prev
        d[t] = d_prev

        w = w_std * torch.randn(Ds, generator=g, device=device)
        v = v_std * torch.randn(Dd, generator=g, device=device)

        s_prev = A @ s_prev + w
        d_prev = 0.9 * d_prev + v

    return x, s, d, x_sig, x_noi, (A, C, D)

def make_train_test_split(x, s, train_T=6000, test_T=2000):
    assert x.shape[0] >= train_T + test_T
    x_tr = x[:train_T]
    s_tr = s[:train_T]
    x_te = x[train_T:train_T + test_T]
    s_te = s[train_T:train_T + test_T]
    return x_tr, s_tr, x_te, s_te


# ----------------------------
# Models (linear, bias=False)
# ----------------------------
class LinearVAE(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.mu = nn.Linear(Dx, Dz, bias=False)
        self.logvar = nn.Linear(Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def encode(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-20, 20)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar

class LinearAR(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def forward(self, x_t):
        z_t = self.enc(x_t)
        xhat_next = self.dec(z_t)
        return xhat_next, z_t

class LinearJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.pred = nn.Linear(Dz, Dz, bias=False)
        self.tgt = nn.Linear(Dx, Dz, bias=False)
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        self.tgt.load_state_dict(self.enc.state_dict())

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for p_tgt, p_enc in zip(self.tgt.parameters(), self.enc.parameters()):
            p_tgt.data.mul_(tau).add_(p_enc.data, alpha=(1 - tau))

    def forward(self, x_t, x_next):
        z_t = self.enc(x_t)
        zhat_next = self.pred(z_t)
        with torch.no_grad():
            ztgt_next = self.tgt(x_next)
        return zhat_next, z_t, ztgt_next

def vicreg_loss(z1, z2, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    inv = F.mse_loss(z1, z2)

    def var_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))
    var = var_term(z1) + var_term(z2)

    def cov_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        N, D = z.shape
        cov = (z.T @ z) / (N - 1 + 1e-12)
        offdiag = cov - torch.diag(torch.diag(cov))
        return (offdiag ** 2).sum() / D
    cov = cov_term(z1) + cov_term(z2)

    return inv_coeff * inv + var_coeff * var + cov_coeff * cov

class LinearVJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.mu_pred = nn.Linear(Dz, Dz, bias=False)
        self.logvar_pred = nn.Linear(Dz, Dz, bias=False)

        self.mu_tgt = nn.Linear(Dx, Dz, bias=False)
        self.logvar_tgt = nn.Linear(Dx, Dz, bias=False)
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        self.mu_tgt.load_state_dict(self.enc.state_dict())
        nn.init.zeros_(self.logvar_tgt.weight)

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for p_tgt, p_enc in zip(self.mu_tgt.parameters(), self.enc.parameters()):
            p_tgt.data.mul_(tau).add_(p_enc.data, alpha=(1 - tau))

    def forward(self, x_t, x_next):
        z_t = self.enc(x_t)
        mu_p = self.mu_pred(z_t)
        logvar_p = self.logvar_pred(z_t).clamp(-20, 20)

        with torch.no_grad():
            mu_q = self.mu_tgt(x_next)
            logvar_q = self.logvar_tgt(x_next).clamp(-20, 20)

        return mu_p, logvar_p, mu_q, logvar_q

def diag_gaussian_kl(mu_q, logvar_q, mu_p=None, logvar_p=None):
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-12) - 1.0)
    return kl.sum(dim=-1).mean()

def diag_gaussian_nll(sample_z, mu, logvar):
    var = torch.exp(logvar)
    nll = 0.5 * ((sample_z - mu) ** 2 / (var + 1e-12) + logvar + math.log(2 * math.pi))
    return nll.sum(dim=-1).mean()

class LinearBJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.mu_dyn = nn.Linear(Dz, Dz, bias=False)
        self.logvar_dyn = nn.Linear(Dz, Dz, bias=False)

        self.mu_prior = nn.Parameter(torch.zeros(Dz))
        self.logvar_prior = nn.Parameter(torch.zeros(Dz))

        self.mu_tgt = nn.Linear(Dx, Dz, bias=False)
        self.logvar_tgt = nn.Linear(Dx, Dz, bias=False)
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        self.mu_tgt.load_state_dict(self.enc.state_dict())
        nn.init.zeros_(self.logvar_tgt.weight)

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for p_tgt, p_enc in zip(self.mu_tgt.parameters(), self.enc.parameters()):
            p_tgt.data.mul_(tau).add_(p_enc.data, alpha=(1 - tau))

    def forward(self, x_t, x_next):
        z_t = self.enc(x_t)
        mu_dyn = self.mu_dyn(z_t)
        logvar_dyn = self.logvar_dyn(z_t).clamp(-20, 20)

        with torch.no_grad():
            mu_q = self.mu_tgt(x_next)
            logvar_q = self.logvar_tgt(x_next).clamp(-20, 20)

        return mu_dyn, logvar_dyn, mu_q, logvar_q

    def fused_posterior_mean(self, mu_dyn, logvar_dyn):
        var_dyn = torch.exp(logvar_dyn)
        var_prior = torch.exp(self.logvar_prior).unsqueeze(0)
        mu_prior = self.mu_prior.unsqueeze(0)
        return (var_prior * mu_dyn + var_dyn * mu_prior) / (var_dyn + var_prior + 1e-12)


# ----------------------------
# Training helper (full-batch)
# ----------------------------
def train_full_batch(model, loss_fn, steps=6000, lr=1e-3, ema_tau=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model)
        loss.backward()
        opt.step()
        if ema_tau is not None and hasattr(model, "ema_update"):
            model.ema_update(tau=ema_tau)
    return model


# ----------------------------
# Paper-style time series plot (no scatter/PCA)
# ----------------------------
def plot_time_series_paper_style(
    sigma, env_snr, t, y_true, yhat_dict,
    dim=0, max_points=2000
):
    """
    Style roughly like the paper: long time axis, title shows Scale + env SNR.
    """
    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).astype(int)
        t = t[idx]
        y_true = y_true[idx]
        yhat_dict = {k: v[idx] for k, v in yhat_dict.items()}

    plt.figure(figsize=(12, 4.8))
    plt.plot(t, y_true[:, dim], linewidth=3, label="True")

    styles = {
        "AR":    dict(linestyle="--", linewidth=2.0, alpha=0.65),
        "JEPA":  dict(linestyle="-",  linewidth=2.6, alpha=0.95),
        "VJEPA": dict(linestyle="--", linewidth=2.2, alpha=0.85),
        "BJEPA": dict(linestyle="-",  linewidth=2.6, alpha=0.90),
        "VAE":   dict(linestyle="-",  linewidth=2.2, alpha=0.85),
    }

    for name, yhat in yhat_dict.items():
        kw = styles.get(name, dict(linestyle="-", linewidth=2.0, alpha=0.85))
        plt.plot(t, yhat[:, dim], label=name, **kw)

    plt.title(f"Scale {sigma:.1f} (SNR: {env_snr:.1f} dB)")
    plt.xlabel("Time Step (t)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=3, loc="lower left")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"series_{sigma}.png")


# ----------------------------
# One sigma run (train/test split + R² + optional time series)
# ----------------------------
def run_sigma(
    sigma,
    device="cpu",
    seed=111,
    train_T=6000,
    test_T=2000,
    a_scale=0.98,
    w_std=0.3,
    steps=6000,
    lr=1e-3,
    make_timeseries=False,
    ts_points=2000,
    ts_dim=0,
):
    # IMPORTANT: keep model init consistent per sigma (removes weird dips)
    set_seed(seed)

    Dx, Ds, Dd, Dz = 20, 4, 4, 4

    # Rollout once per sigma, then slice into train/test
    x, s, d, x_sig, x_noi, _ = rollout_noisy_tv(
        Dx=Dx, Ds=Ds, Dd=Dd, T=train_T + test_T, sigma=sigma,
        a_scale=a_scale, w_std=w_std, device=device, seed=seed
    )
    env_snr = env_snr_db(x_sig, x_noi)

    x_tr, s_tr, x_te, s_te = make_train_test_split(x, s, train_T=train_T, test_T=test_T)

    # Predictive pairs
    x_t_tr, x_n_tr = x_tr[:-1], x_tr[1:]
    s_t_tr, s_n_tr = s_tr[:-1], s_tr[1:]

    x_t_te, x_n_te = x_te[:-1], x_te[1:]
    s_t_te, s_n_te = s_te[:-1], s_te[1:]

    # --- VAE ---
    vae = LinearVAE(Dx=Dx, Dz=Dz).to(device)
    beta_vae = 1.0
    def vae_loss(m):
        xhat, mu, logvar = m(x_tr)
        recon = F.mse_loss(xhat, x_tr)
        kl = diag_gaussian_kl(mu, logvar)
        return recon + beta_vae * kl
    train_full_batch(vae, vae_loss, steps=steps, lr=lr)

    with torch.no_grad():
        mu_tr, _ = vae.encode(x_tr[:-1])
        mu_te, _ = vae.encode(x_te[:-1])
    r2_vae = r2_score(
        s_t_te.cpu().numpy(),
        ols_fit_predict(mu_tr.cpu().numpy(), s_t_tr.cpu().numpy(), mu_te.cpu().numpy())
    )

    # --- AR ---
    ar = LinearAR(Dx=Dx, Dz=Dz).to(device)
    def ar_loss(m):
        xhat_next, _ = m(x_t_tr)
        return F.mse_loss(xhat_next, x_n_tr)
    train_full_batch(ar, ar_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, ztr = ar(x_t_tr)
        _, zte = ar(x_t_te)
    r2_ar = r2_score(
        s_n_te.cpu().numpy(),
        ols_fit_predict(ztr.cpu().numpy(), s_n_tr.cpu().numpy(), zte.cpu().numpy())
    )

    # --- JEPA (VICReg) ---
    jepa = LinearJEPA(Dx=Dx, Dz=Dz).to(device)
    def jepa_loss(m):
        zhat_next, _, ztgt_next = m(x_t_tr, x_n_tr)
        return vicreg_loss(zhat_next, ztgt_next, 25.0, 25.0, 1.0)
    train_full_batch(jepa, jepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        zhat_tr, _, _ = jepa(x_t_tr, x_n_tr)
        zhat_te, _, _ = jepa(x_t_te, x_n_te)
    r2_jepa = r2_score(
        s_n_te.cpu().numpy(),
        ols_fit_predict(zhat_tr.cpu().numpy(), s_n_tr.cpu().numpy(), zhat_te.cpu().numpy())
    )

    # --- VJEPA ---
    vjepa = LinearVJEPA(Dx=Dx, Dz=Dz).to(device)
    beta_v = 0.01
    def vjepa_loss(m):
        mu_p, logvar_p, mu_q, logvar_q = m(x_t_tr, x_n_tr)
        std_q = torch.exp(0.5 * logvar_q)
        z_samp = mu_q + torch.randn_like(std_q) * std_q
        nll = diag_gaussian_nll(z_samp, mu_p, logvar_p)
        kl = diag_gaussian_kl(mu_q, logvar_q)
        return nll + beta_v * kl
    train_full_batch(vjepa, vjepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        mu_p_tr, _, _, _ = vjepa(x_t_tr, x_n_tr)
        mu_p_te, _, _, _ = vjepa(x_t_te, x_n_te)
    r2_vjepa = r2_score(
        s_n_te.cpu().numpy(),
        ols_fit_predict(mu_p_tr.cpu().numpy(), s_n_tr.cpu().numpy(), mu_p_te.cpu().numpy())
    )

    # --- BJEPA ---
    bjepa = LinearBJEPA(Dx=Dx, Dz=Dz).to(device)
    beta_b = 0.01
    gamma = 0.1
    def bjepa_loss(m):
        mu_dyn, logvar_dyn, mu_q, logvar_q = m(x_t_tr, x_n_tr)
        std_q = torch.exp(0.5 * logvar_q)
        z_samp = mu_q + torch.randn_like(std_q) * std_q
        nll = diag_gaussian_nll(z_samp, mu_dyn, logvar_dyn)
        kl_target = diag_gaussian_kl(mu_q, logvar_q)
        kl_prior = diag_gaussian_kl(m.mu_prior.unsqueeze(0), m.logvar_prior.unsqueeze(0))
        return nll + beta_b * kl_target + gamma * kl_prior
    train_full_batch(bjepa, bjepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        mu_dyn_tr, logvar_dyn_tr, _, _ = bjepa(x_t_tr, x_n_tr)
        mu_dyn_te, logvar_dyn_te, _, _ = bjepa(x_t_te, x_n_te)
        mu_post_tr = bjepa.fused_posterior_mean(mu_dyn_tr, logvar_dyn_tr)
        mu_post_te = bjepa.fused_posterior_mean(mu_dyn_te, logvar_dyn_te)
    r2_bjepa = r2_score(
        s_n_te.cpu().numpy(),
        ols_fit_predict(mu_post_tr.cpu().numpy(), s_n_tr.cpu().numpy(), mu_post_te.cpu().numpy())
    )

    # Optional: paper-style time series for this sigma (predictive models, s[t+1])
    if make_timeseries:
        # Use a consistent readout target: s_{t+1}
        # Compute predictions on the FIRST part of test (for the plot)
        t_np = np.arange(s_n_te.shape[0])

        # Fit readouts on train
        vae_yhat = ols_fit_predict(mu_tr.cpu().numpy(), s_n_tr.cpu().numpy(), mu_te.cpu().numpy())

        # AR
        ar_yhat = ols_fit_predict(ztr.cpu().numpy(), s_n_tr.cpu().numpy(), zte.cpu().numpy())

        # JEPA (predictor output)
        jepa_yhat = ols_fit_predict(zhat_tr.cpu().numpy(), s_n_tr.cpu().numpy(), zhat_te.cpu().numpy())

        # VJEPA (mu_pred)
        vjepa_yhat = ols_fit_predict(mu_p_tr.cpu().numpy(), s_n_tr.cpu().numpy(), mu_p_te.cpu().numpy())

        # BJEPA (fused mean)
        bjepa_yhat = ols_fit_predict(mu_post_tr.cpu().numpy(), s_n_tr.cpu().numpy(), mu_post_te.cpu().numpy())

        preds = {
            "VAE": vae_yhat,
            "AR": ar_yhat,
            "JEPA": jepa_yhat,
            "VJEPA": vjepa_yhat,
            "BJEPA": bjepa_yhat,
        }

        plot_time_series_paper_style(
            sigma=sigma,
            env_snr=env_snr,
            t=t_np,
            y_true=s_n_te.cpu().numpy(),
            yhat_dict=preds,
            dim=ts_dim,
            max_points=ts_points,
        )

    return {
        "sigma": float(sigma),
        "env_snr_db": float(env_snr),
        "R2_VAE": float(r2_vae),
        "R2_AR": float(r2_ar),
        "R2_JEPA": float(r2_jepa),
        "R2_VJEPA": float(r2_vjepa),
        "R2_BJEPA": float(r2_bjepa),
    }


# ----------------------------
# Plots: R² vs sigma
# ----------------------------
def plot_r2_vs_sigma(df: pd.DataFrame, out_path: str = None):
    plt.figure()
    x = df["sigma"].to_numpy()
    for col, label in [
        ("R2_VAE", "VAE"),
        ("R2_AR", "AR"),
        ("R2_JEPA", "JEPA"),
        ("R2_VJEPA", "VJEPA"),
        ("R2_BJEPA", "BJEPA"),
    ]:
        plt.plot(x, df[col].to_numpy(), marker="o", label=label)

    plt.xlabel("sigma")
    plt.ylabel("R² (linear probe)")
    plt.title("Noisy-TV Toy Experiment: R² vs sigma")
    plt.grid(True)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    #plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 111

    # sigma sweep (paper: 0..8 in 9 steps)
    sigmas = np.linspace(0.0, 10.0, 11)

    # Tweaks to make dynamics less trivially predictable and more paper-like
    a_scale = 0.99
    w_std = 0.1

    # Training
    steps = 10000
    lr = 1e-3

    rows = []
    for s in sigmas:
        row = run_sigma(
            sigma=float(s),
            device=device,
            seed=seed,
            train_T=5000,
            test_T=300,
            a_scale=a_scale,
            w_std=w_std,
            steps=steps,
            lr=lr,
            make_timeseries=True,   # set True to render per-sigma time series
        )
        print(row)
        rows.append(row)
        df = pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)
        plot_r2_vs_sigma(df, out_path="r2_vs_sigma.png")

    df = pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)
    print("\nSummary:\n", df)
    plot_r2_vs_sigma(df, out_path="r2_vs_sigma.png")

    # Make a few paper-style time series plots like the paper (choose 0, 4, 8)
    #for s in [0.0, 4.0, 8.0]:
    #    _ = run_sigma(
    #        sigma=float(s),
    #        device=device,
    #        seed=seed,
    #        train_T=750,
    #        test_T=250,
    #        a_scale=a_scale,
    #        w_std=w_std,
    #        steps=steps,
    #        lr=lr,
    #        make_timeseries=True,
    #        ts_points=1000,  # paper-like long plot
    #        ts_dim=0
    #    )


if __name__ == "__main__":
    main()
