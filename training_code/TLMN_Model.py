"""
================================================================================
Copyright (c) 2026, Eng. Mohammed Ezzeldin Babiker Abdullah. All rights reserved.
Official Project & Research Work by م. محمد عزالدين بابكر عبدالله
================================================================================
Thermodynamic Liquid Manifold Network (TLMN) — v3 (Architecture Rebuild)
================================================================================

Author : Prepared by: Eng. Mohammed Izzaldeen Babeker Abdullah
Data   : NASA POWER Hourly (2010-2015), Khartoum, Sudan

v3 Core Architectural Changes:
  [A] 1D-CNN Temporal Encoder replaces LiquidNeuralODE to capture sharp irradiance
      peaks and high-frequency transients without temporal smoothing lag.
  [B] Physics Gating Output: final prediction = sigmoid_output * ClearSky_Guide
      Forces physical correctness (night=0, daytime bounded) WITHOUT loss penalties.
  [C] Log-Cosh Loss replaces MSE+penalties: pursues peaks aggressively, robust to
      outliers, no penalty-induced "cowardly" sub-peak predictions.
  [D] Symplectic Cross-Attention retained — now using just CLRSKY+SZA as K/V
      (clean, no redundant HR channels).
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ==============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# ==============================================================================
DATA_PATH = "Hourly_2010_2015.csv"

with open(DATA_PATH, "r") as f:
    for idx, line in enumerate(f):
        if "-END HEADER-" in line:
            skip_rows = idx + 1
            break

print(f"[INFO] Skipping {skip_rows} header lines.")
df = pd.read_csv(DATA_PATH, skiprows=skip_rows)
print(f"[INFO] Raw data shape: {df.shape}")

df.replace(-999.0, np.nan, inplace=True)
df.replace(-999,   np.nan, inplace=True)
df.interpolate(method="linear", inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

# Dynamic feature engineering
df['T2M_diff']   = df['T2M'].diff()
df['RH2M_diff']  = df['RH2M'].diff()
df['WS10M_diff'] = df['WS10M'].diff()
df['PS_diff']    = df['PS'].diff()

decay_window = 3
weights_w = np.arange(1, decay_window + 1)
def wma(s):
    return s.rolling(window=decay_window).apply(
        lambda x: np.dot(x, weights_w) / weights_w.sum(), raw=True)

df['DNI_frac_mem']  = wma(df['ALLSKY_SFC_SW_DNI'])
df['DIFF_frac_mem'] = wma(df['ALLSKY_SFC_SW_DIFF'])
df.bfill(inplace=True)
df['TSI'] = df['PS'] / (df['T2M'] + 273.15) * df['WS10M']

TARGET_COL = "ALLSKY_SFC_SW_DWN"
PHYSICAL_COLS = [
    "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "ALLSKY_KT",
    "SZA", "T2M", "RH2M", "WS10M", "PS",
    "T2M_diff", "RH2M_diff", "WS10M_diff", "PS_diff",
    "DNI_frac_mem", "DIFF_frac_mem", "TSI"
]
TIME_COLS = ["MO", "DY", "HR"]
KEEP_COLS = [TARGET_COL] + PHYSICAL_COLS + TIME_COLS
df = df[KEEP_COLS].copy()

# Night mask
RADIATION_COLS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
                  "ALLSKY_SFC_SW_DNI",  "ALLSKY_SFC_SW_DIFF"]
night_mask = df["CLRSKY_SFC_SW_DWN"] <= 0.0
for col in RADIATION_COLS:
    df.loc[night_mask, col] = 0.0

# Log transform
LOG_COLS = ["ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M"]
if df["T2M"].min() < 0:
    df["T2M"] = df["T2M"] + abs(df["T2M"].min()) + 1.0
for col in LOG_COLS:
    df[col] = np.log1p(df[col].clip(lower=0))

# Cyclical time encoding
df["MO_sin"] = np.sin(2 * np.pi * df["MO"] / 12.0)
df["MO_cos"] = np.cos(2 * np.pi * df["MO"] / 12.0)
df["DY_sin"] = np.sin(2 * np.pi * df["DY"] / 31.0)
df["DY_cos"] = np.cos(2 * np.pi * df["DY"] / 31.0)
df["HR_sin"] = np.sin(2 * np.pi * df["HR"] / 24.0)
df["HR_cos"] = np.cos(2 * np.pi * df["HR"] / 24.0)

CYCLICAL_COLS = ["MO_sin", "MO_cos", "DY_sin", "DY_cos", "HR_sin", "HR_cos"]
df.drop(columns=TIME_COLS, inplace=True)

FEATURE_COLS = PHYSICAL_COLS + CYCLICAL_COLS
NUM_FEATURES = len(FEATURE_COLS)
print(f"[INFO] Total features: {NUM_FEATURES}")

# ==============================================================================
# SECTION 2 — CHRONOLOGICAL SPLIT (80/20)
# ==============================================================================
n_train  = int(len(df) * 0.80)
df_train = df.iloc[:n_train].copy()
df_test  = df.iloc[n_train:].copy()

# ==============================================================================
# SECTION 3 — HYBRID NORMALIZATION (fit on train only)
# ==============================================================================
scaler_features = StandardScaler()
scaler_features.fit(df_train[FEATURE_COLS].values)

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(df_train[[TARGET_COL]].values)

train_features      = scaler_features.transform(df_train[FEATURE_COLS].values)
test_features       = scaler_features.transform(df_test[FEATURE_COLS].values)
train_target_scaled = scaler_target.transform(df_train[[TARGET_COL]].values).flatten()
test_target_scaled  = scaler_target.transform(df_test[[TARGET_COL]].values).flatten()

# Raw unscaled values for the Physics Gate
train_clrsky_unscaled = df_train["CLRSKY_SFC_SW_DWN"].values
test_clrsky_unscaled  = df_test["CLRSKY_SFC_SW_DWN"].values

# Indices
SZA_IDX    = FEATURE_COLS.index("SZA")
CLRSKY_IDX = FEATURE_COLS.index("CLRSKY_SFC_SW_DWN")

# ==============================================================================
# SECTION 4 — SLIDING WINDOW DATASET
# ==============================================================================
INPUT_WINDOW = 24
HORIZON      = 1
SUB_WINDOW   = 5

def create_sequences(features, target_scaled, clrsky_raw, input_window, horizon):
    X_list, y_list, clr_list = [], [], []
    for i in range(len(features) - input_window - horizon + 1):
        X_list.append(features[i : i + input_window])
        ti = i + input_window + horizon - 1
        y_list.append(target_scaled[ti])
        clr_list.append(clrsky_raw[ti])
    return np.array(X_list), np.array(y_list), np.array(clr_list)

X_tr, y_tr, clr_tr = create_sequences(
    train_features, train_target_scaled, train_clrsky_unscaled,
    INPUT_WINDOW, HORIZON)
X_te, y_te, clr_te = create_sequences(
    test_features, test_target_scaled, test_clrsky_unscaled,
    INPUT_WINDOW, HORIZON)

def to_t(a): return torch.tensor(a, dtype=torch.float32)

BATCH_SIZE = 128
train_loader = DataLoader(
    TensorDataset(to_t(X_tr), to_t(y_tr), to_t(clr_tr)),
    batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(
    TensorDataset(to_t(X_te), to_t(y_te), to_t(clr_te)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ==============================================================================
# SECTION 5 — TLMN v3 MODEL ARCHITECTURE
# ==============================================================================

class HankelEmbeddingLayer(nn.Module):
    """Unrolls 1D sequence into Hankel state-space tensor."""
    def __init__(self, sub_window: int):
        super().__init__()
        self.sub_window = sub_window

    def forward(self, x):
        b, s, f = x.shape
        xu = x.permute(0, 2, 1).unfold(2, self.sub_window, 1)
        nw = xu.size(2)
        return xu.permute(0, 2, 1, 3).reshape(b, nw, f * self.sub_window)


class CNN1D_TemporalEncoder(nn.Module):
    """
    [ARCH A] Replaces LiquidNeuralODE.
    3-layer dilated 1D-CNN to capture multi-scale temporal patterns
    (sharp peaks, slow diurnal trend, inter-hour transitions) without lag.
    Dilations [1, 2, 4] give receptive field of 13 steps with only 3 layers.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3,
                               padding=1,  dilation=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3,
                               padding=2,  dilation=2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3,
                               padding=4,  dilation=4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.act   = nn.GELU()

    def forward(self, x):
        # x: (B, T, d_model)  — conv expects (B, C, T)
        xc = x.permute(0, 2, 1)
        xc = self.act(self.norm1((self.conv1(xc) + xc).permute(0, 2, 1))).permute(0, 2, 1)
        xc = self.act(self.norm2((self.conv2(xc) + xc).permute(0, 2, 1))).permute(0, 2, 1)
        xc = self.act(self.norm3((self.conv3(xc) + xc).permute(0, 2, 1)))
        return xc  # (B, T, d_model)


class SymplecticCrossAttention(nn.Module):
    """Physics-guided cross-attention: Q=meteorological, K/V=ClearSky+SZA."""
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.phys_proj    = nn.Linear(2, embed_dim)
        self.cross_attn   = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.gamma        = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, h, clrsky_seq, sza_seq):
        kv  = self.phys_proj(torch.cat([clrsky_seq, sza_seq], dim=-1))
        ao, _= self.cross_attn(query=h, key=kv, value=kv)
        return h + self.gamma * ao


class PhysicsGatedOutput(nn.Module):
    """
    [ARCH B] Physics Gating — replaces loss penalties entirely.
    Architecture:
        raw_logit  = KAN_RBF(h_last)          # unconstrained
        gate_alpha = sigmoid(raw_logit)        # in (0, 1)
        prediction = gate_alpha * clrsky_norm  # physically bounded

    Effect:
      - Night  (clrsky_norm = 0): prediction = 0   ALWAYS, no penalty needed.
      - Day    (clrsky_norm > 0): prediction ≤ clrsky_norm  ALWAYS.
      - MSE is now free to focus 100% on matching peaks — no compromise.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.kan_linear = nn.Linear(d_model, 32)
        self.kan_act    = nn.GELU()
        self.kan_out    = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.kan_linear.weight)
        nn.init.xavier_uniform_(self.kan_out.weight)

    def forward(self, h_last, clrsky_norm):
        """
        h_last    : (B, d_model) — last hidden state
        clrsky_norm: (B, 1)      — ClearSky in [0,1] normalized
        """
        raw     = self.kan_out(self.kan_act(self.kan_linear(h_last)))
        alpha   = torch.sigmoid(raw)           # (B, 1) in (0, 1)
        pred    = alpha * clrsky_norm          # (B, 1) physically gated
        return pred


class MathematicalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==============================================================================
# Main TLMN v3 Class
# ==============================================================================
class TLMN_v3(nn.Module):
    def __init__(self, num_features, sub_window: int = 5, d_model: int = 64,
                 clrsky_idx: int = 0, sza_idx: int = 4):
        super().__init__()
        self.clrsky_idx = clrsky_idx
        self.sza_idx    = sza_idx

        hankel_dim = num_features * sub_window

        # Layer 1: Hankel Embedding
        self.hankel      = HankelEmbeddingLayer(sub_window)
        self.proj        = nn.Linear(hankel_dim, d_model)
        nn.init.kaiming_normal_(self.proj.weight)
        self.proj_act    = nn.Tanh()
        self.pos_encoder = MathematicalPositionalEncoding(d_model)

        # Layer 2: 1D-CNN Temporal Encoder [ARCH A]
        self.cnn_encoder = CNN1D_TemporalEncoder(d_model)

        # Layer 3: Symplectic Physics Cross-Attention
        self.cross_attn  = SymplecticCrossAttention(d_model, num_heads=4)

        # Layer 4: Physics-Gated Output [ARCH B]
        self.phys_gate   = PhysicsGatedOutput(d_model)

        # ClearSky normalization parameters (set after scaler is known)
        self.clrsky_scale = 1.0   # will be overridden externally

    def forward(self, x, clrsky_unscaled_target=None):
        """
        x                     : (B, T, F) — full feature window
        clrsky_unscaled_target: (B, 1)    — raw ClearSky for the target step
        """
        clrsky_full = x[:, :, self.clrsky_idx].unsqueeze(-1)  # (B, T, 1)
        sza_full    = x[:, :, self.sza_idx].unsqueeze(-1)      # (B, T, 1)

        # Hankel + projection
        h  = self.hankel(x)                          # (B, nw, hankel_dim)
        nw = h.size(1)
        h  = self.proj_act(self.proj(h))             # (B, nw, d_model)
        h  = self.pos_encoder(h)                     # (B, nw, d_model)

        # 1D-CNN temporal encoding [ARCH A]
        h  = self.cnn_encoder(h)                     # (B, nw, d_model)

        # Physics cross-attention
        clrsky_seq = clrsky_full[:, -nw:, :]
        sza_seq    = sza_full[:, -nw:, :]
        h  = self.cross_attn(h, clrsky_seq, sza_seq) # (B, nw, d_model)

        h_last = h[:, -1, :]                          # (B, d_model)

        # Physics Gate [ARCH B]
        if clrsky_unscaled_target is not None:
            # Normalize ClearSky to [0, 1] using train max
            clrsky_norm = torch.clamp(
                clrsky_unscaled_target / self.clrsky_scale, 0.0, 1.0)
        else:
            # Fallback: use last step ClearSky from input window
            clrsky_norm = torch.clamp(
                clrsky_full[:, -1, :] / self.clrsky_scale, 0.0, 1.0)

        pred = self.phys_gate(h_last, clrsky_norm)   # (B, 1)
        return pred


# ==============================================================================
# SECTION 6 — MODEL INSTANTIATION
# ==============================================================================
# Compute ClearSky max from training data (for gate normalization)
CLRSKY_MAX = float(df_train["CLRSKY_SFC_SW_DWN"].max())
GHI_MAX    = float(df_train[TARGET_COL].max())

model = TLMN_v3(
    num_features=NUM_FEATURES,
    sub_window=SUB_WINDOW,
    d_model=64,
    clrsky_idx=CLRSKY_IDX,
    sza_idx=SZA_IDX
).to(DEVICE)
model.clrsky_scale = CLRSKY_MAX   # inject physics normalization constant

print(f"\n{'='*60}")
print(f"  TLMN v3 — Architecture Summary")
print(f"{'='*60}")
print(f"  Temporal Encoder  : 1D-CNN (Dilated, 3 layers)")
print(f"  Output Constraint : Physics Gate (α × ClearSky_norm)")
print(f"  Loss Function     : Log-Cosh")
print(f"  ClearSky Max      : {CLRSKY_MAX:.2f} Wh/m²")
print(f"  Total parameters  : {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*60}\n")


# ==============================================================================
# SECTION 7 — LOG-COSH LOSS [ARCH C]
# ==============================================================================
class LogCoshLoss(nn.Module):
    """
    [ARCH C] Log-Cosh Loss: smooth approximation of MAE.
      - Behaves like MSE for small errors (differentiable at 0).
      - Behaves like MAE for large errors (robust to outliers).
      - Does NOT punish peaks — aggressively pursues them.
    No physics penalty needed: Physics Gate enforces boundaries structurally.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff = pred - target
        # numerically stable: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        loss = torch.abs(diff) + torch.log1p(torch.exp(-2.0 * torch.abs(diff))) - math.log(2)
        return torch.mean(loss)


NUM_EPOCHS    = 60
LEARNING_RATE = 1e-3

loss_fn   = LogCoshLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=8)

def compute_rmse(p, t): return torch.sqrt(torch.mean((p - t)**2)).item()
def compute_mae(p, t):  return torch.mean(torch.abs(p - t)).item()

print(f"{'='*70}")
print(f"  Training TLMN v3 — {NUM_EPOCHS} Epochs | Log-Cosh | Physics Gate")
print(f"{'='*70}")

train_losses, val_losses = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss, num_batches = 0.0, 0

    for X_b, y_b, clr_b in train_loader:
        X_b   = X_b.to(DEVICE)
        y_b   = y_b.to(DEVICE).unsqueeze(1)
        clr_b = clr_b.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(X_b, clrsky_unscaled_target=clr_b)
        loss  = loss_fn(preds, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_rmse     = math.sqrt(max(avg_train_loss, 1e-10))
    train_losses.append(avg_train_loss)

    model.eval()
    vp_list, vt_list = [], []
    with torch.no_grad():
        for X_b, y_b, clr_b in test_loader:
            p = model(X_b.to(DEVICE), clr_b.to(DEVICE).unsqueeze(1))
            vp_list.append(p.cpu())
            vt_list.append(y_b.unsqueeze(1))

    val_preds_all   = torch.cat(vp_list)
    val_targets_all = torch.cat(vt_list)
    val_rmse = compute_rmse(val_preds_all, val_targets_all)
    val_mae  = compute_mae(val_preds_all, val_targets_all)
    val_losses.append(val_rmse ** 2)
    scheduler.step(val_rmse)

    print(f"  Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
          f"Train RMSE: {train_rmse:.5f} | "
          f"Val RMSE: {val_rmse:.5f} | Val MAE: {val_mae:.5f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

torch.save(model.state_dict(), "tlmn_saved_weights.pth")
print(f"\n[INFO] Weights saved: tlmn_saved_weights.pth")

# ==============================================================================
# SECTION 8 — FINAL EVALUATION
# ==============================================================================
preds_orig   = scaler_target.inverse_transform(
    val_preds_all.numpy()).flatten()
targets_orig = scaler_target.inverse_transform(
    val_targets_all.numpy()).flatten()
preds_orig   = np.maximum(preds_orig, 0.0)

final_rmse = np.sqrt(np.mean((preds_orig - targets_orig)**2))
final_mae  = np.mean(np.abs(preds_orig - targets_orig))
corr       = np.corrcoef(targets_orig, preds_orig)[0, 1]

print(f"\n{'='*60}")
print(f"  Final Metrics (Wh/m²)")
print(f"{'='*60}")
print(f"  RMSE        : {final_rmse:.4f}")
print(f"  MAE         : {final_mae:.4f}")
print(f"  Correlation : {corr:.6f}")
print(f"{'='*60}\n")

# ==============================================================================
# SECTION 9 — VISUALIZATION
# ==============================================================================
def plot_predictions(targets, predictions, rmse, mae, num_points=500):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Thermodynamic Liquid Manifold Network (TLMN) v3\n"
                 "Solar Irradiance Forecasting",
                 fontsize=16, fontweight="bold", y=0.98)

    n   = min(num_points, len(targets))
    idx = np.arange(n)
    t, p = targets[:n], predictions[:n]

    ax1 = axes[0]
    ax1.plot(idx, t, color="#1a73e8", lw=1.4, alpha=0.9, label="Actual")
    ax1.plot(idx, p, color="#ea4335", lw=1.1, ls="--", label="Predicted (TLMN v3)")
    ax1.fill_between(idx, t, p, alpha=0.15, color="#fbbc04")
    ax1.set_ylabel("Solar Irradiance (Wh/m²)", fontsize=13)
    ax1.set_title(f"RMSE = {rmse:.2f} Wh/m²   |   MAE = {mae:.2f} Wh/m²",
                  fontsize=12)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2    = axes[1]
    errors = t - p
    colors = np.where(errors >= 0, "#34a853", "#ea4335")
    ax2.bar(idx, errors, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color="#333333", lw=0.8)
    ax2.set_xlabel("Sample Index", fontsize=13)
    ax2.set_ylabel("Error (Wh/m²)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("tlmn_forecast_results.png", dpi=200)
    print("[INFO] Saved: tlmn_forecast_results.png")


def plot_training_curves(train_losses, val_losses):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = range(1, len(train_losses) + 1)
    ax.plot(epochs, [math.sqrt(max(l, 1e-10)) for l in train_losses],
            color="#1a73e8", marker="o", ms=4, label="Train RMSE")
    ax.plot(epochs, [math.sqrt(max(l, 1e-10)) for l in val_losses],
            color="#ea4335", marker="s", ms=4, label="Val RMSE")
    ax.set_ylabel("RMSE (Normalized)")
    ax.set_xlabel("Epoch")
    ax.set_title("TLMN v3 — Training Convergence", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("tlmn_training_curves.png", dpi=200)
    print("[INFO] Saved: tlmn_training_curves.png")


plot_training_curves(train_losses, val_losses)
plot_predictions(targets_orig, preds_orig, final_rmse, final_mae)
print("=" * 60 + "\n  TLMN v3 Pipeline Complete.\n" + "=" * 60)
