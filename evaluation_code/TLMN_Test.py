"""
================================================================================
Thermodynamic Liquid Manifold Network (TLMN) — Evaluation Suite v3
================================================================================

Author : Prepared by: Eng. Mohammed Izzaldeen Babeker Abdullah
Description : Loads trained TLMN v3 weights and generates 6 publication-ready plots.
              Architecture identically mirrors TLMN_Model.py v3 with CNN-1D 
              temporal encoder and Physics Gate.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size']   = 12
import warnings

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] TLMN Evaluation Suite v3 on: {DEVICE}")

# ==============================================================================
# SECTION 1 — DATA PIPELINE (Identical to TLMN_Model.py v3)
# ==============================================================================
DATA_PATH = "Hourly_2010_2015.csv"

with open(DATA_PATH, "r") as f:
    for idx, line in enumerate(f):
        if "-END HEADER-" in line:
            skip_rows = idx + 1
            break

df = pd.read_csv(DATA_PATH, skiprows=skip_rows)
df.replace(-999.0, np.nan, inplace=True)
df.replace(-999,   np.nan, inplace=True)
df.interpolate(method="linear", inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

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

TARGET_COL   = "ALLSKY_SFC_SW_DWN"
PHYSICAL_COLS = [
    "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "ALLSKY_KT",
    "SZA", "T2M", "RH2M", "WS10M", "PS",
    "T2M_diff", "RH2M_diff", "WS10M_diff", "PS_diff",
    "DNI_frac_mem", "DIFF_frac_mem", "TSI"
]
TIME_COLS = ["MO", "DY", "HR"]
KEEP_COLS = [TARGET_COL] + PHYSICAL_COLS + TIME_COLS
df = df[KEEP_COLS].copy()

RADIATION_COLS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
                  "ALLSKY_SFC_SW_DNI",  "ALLSKY_SFC_SW_DIFF"]
night_mask = df["CLRSKY_SFC_SW_DWN"] <= 0.0
for col in RADIATION_COLS:
    df.loc[night_mask, col] = 0.0

LOG_COLS = ["ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M"]
if df["T2M"].min() < 0:
    df["T2M"] = df["T2M"] + abs(df["T2M"].min()) + 1.0
for col in LOG_COLS:
    df[col] = np.log1p(df[col].clip(lower=0))

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
CLRSKY_IDX   = FEATURE_COLS.index("CLRSKY_SFC_SW_DWN")
SZA_IDX      = FEATURE_COLS.index("SZA")

n_train  = int(len(df) * 0.80)
df_train = df.iloc[:n_train].copy()
df_test  = df.iloc[n_train:].copy()

scaler_features = StandardScaler().fit(df_train[FEATURE_COLS].values)
scaler_target   = MinMaxScaler(feature_range=(0, 1)).fit(df_train[[TARGET_COL]].values)

test_features        = scaler_features.transform(df_test[FEATURE_COLS].values)
test_target_scaled   = scaler_target.transform(df_test[[TARGET_COL]].values).flatten()
test_target_unscaled = df_test[TARGET_COL].values
test_clrsky_unscaled = df_test["CLRSKY_SFC_SW_DWN"].values

INPUT_WINDOW, HORIZON, SUB_WINDOW = 24, 1, 5

def create_sequences(features, target_scaled, clrsky_raw, input_window, horizon):
    X_list, y_list, clr_list = [], [], []
    for i in range(len(features) - input_window - horizon + 1):
        X_list.append(features[i : i + input_window])
        ti = i + input_window + horizon - 1
        y_list.append(target_scaled[ti])
        clr_list.append(clrsky_raw[ti])
    return np.array(X_list), np.array(y_list), np.array(clr_list)

X_te, y_te, clr_te = create_sequences(
    test_features, test_target_scaled, test_clrsky_unscaled,
    INPUT_WINDOW, HORIZON)

def to_t(a): return torch.tensor(a, dtype=torch.float32)

test_loader = DataLoader(
    TensorDataset(to_t(X_te), to_t(y_te), to_t(clr_te)),
    batch_size=128, shuffle=False)

# ==============================================================================
# SECTION 2 — MODEL ARCHITECTURE (v3 Exact Mirror)
# ==============================================================================
class HankelEmbeddingLayer(nn.Module):
    def __init__(self, sub_window: int):
        super().__init__()
        self.sub_window = sub_window
    def forward(self, x):
        b, s, f = x.shape
        xu = x.permute(0, 2, 1).unfold(2, self.sub_window, 1)
        nw = xu.size(2)
        return xu.permute(0, 2, 1, 3).reshape(b, nw, f * self.sub_window)

class CNN1D_TemporalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.act   = nn.GELU()
    def forward(self, x):
        xc = x.permute(0, 2, 1)
        xc = self.act(self.norm1((self.conv1(xc) + xc).permute(0, 2, 1))).permute(0, 2, 1)
        xc = self.act(self.norm2((self.conv2(xc) + xc).permute(0, 2, 1))).permute(0, 2, 1)
        xc = self.act(self.norm3((self.conv3(xc) + xc).permute(0, 2, 1)))
        return xc

class SymplecticCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.phys_proj  = nn.Linear(2, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.gamma      = nn.Parameter(torch.ones(1) * 0.1)
    def forward(self, h, clrsky_seq, sza_seq):
        kv   = self.phys_proj(torch.cat([clrsky_seq, sza_seq], dim=-1))
        ao, _= self.cross_attn(query=h, key=kv, value=kv)
        return h + self.gamma * ao

class PhysicsGatedOutput(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.kan_linear = nn.Linear(d_model, 32)
        self.kan_act    = nn.GELU()
        self.kan_out    = nn.Linear(32, 1)
    def forward(self, h_last, clrsky_norm):
        raw   = self.kan_out(self.kan_act(self.kan_linear(h_last)))
        alpha = torch.sigmoid(raw)
        return alpha * clrsky_norm

class MathematicalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TLMN_v3(nn.Module):
    def __init__(self, num_features, sub_window=5, d_model=64, clrsky_idx=0, sza_idx=4):
        super().__init__()
        self.clrsky_idx  = clrsky_idx
        self.sza_idx     = sza_idx
        self.hankel      = HankelEmbeddingLayer(sub_window)
        self.proj        = nn.Linear(num_features * sub_window, d_model)
        self.proj_act    = nn.Tanh()
        self.pos_encoder = MathematicalPositionalEncoding(d_model)
        self.cnn_encoder = CNN1D_TemporalEncoder(d_model)
        self.cross_attn  = SymplecticCrossAttention(d_model, num_heads=4)
        self.phys_gate   = PhysicsGatedOutput(d_model)
        self.clrsky_scale = 1.0

    def forward(self, x, clrsky_unscaled_target=None):
        clr_f = x[:, :, self.clrsky_idx].unsqueeze(-1)
        sza_f = x[:, :, self.sza_idx].unsqueeze(-1)
        h  = self.hankel(x)
        nw = h.size(1)
        h  = self.pos_encoder(self.proj_act(self.proj(h)))
        h  = self.cnn_encoder(h)
        h  = self.cross_attn(h, clr_f[:, -nw:, :], sza_f[:, -nw:, :])
        if clrsky_unscaled_target is not None:
            clr_n = torch.clamp(clrsky_unscaled_target / self.clrsky_scale, 0.0, 1.0)
        else:
            clr_n = torch.clamp(clr_f[:, -1, :] / self.clrsky_scale, 0.0, 1.0)
        return self.phys_gate(h[:, -1, :], clr_n)

# ==============================================================================
# SECTION 3 — LOAD WEIGHTS & INFERENCE
# ==============================================================================
CLRSKY_MAX = float(df_train["CLRSKY_SFC_SW_DWN"].max())

model = TLMN_v3(NUM_FEATURES, clrsky_idx=CLRSKY_IDX, sza_idx=SZA_IDX).to(DEVICE)
model.clrsky_scale = CLRSKY_MAX
model.load_state_dict(torch.load("tlmn_saved_weights.pth", map_location=DEVICE))
model.eval()
print("[INFO] TLMN v3 weights loaded successfully.")

preds_sc, targ_sc = [], []
with torch.no_grad():
    for X_b, y_b, clr_b in test_loader:
        p = model(X_b.to(DEVICE), clr_b.to(DEVICE).unsqueeze(1))
        preds_sc.append(p.cpu().numpy())
        targ_sc.append(y_b.numpy())

preds_sc = np.concatenate(preds_sc)
targ_sc  = np.concatenate(targ_sc)

# Inverse transform
preds_wh  = scaler_target.inverse_transform(preds_sc).flatten()
actual_wh = scaler_target.inverse_transform(targ_sc.reshape(-1,1)).flatten()
preds_wh  = np.maximum(preds_wh, 0.0)

rmse = np.sqrt(np.mean((preds_wh - actual_wh)**2))
mae  = np.mean(np.abs(preds_wh - actual_wh))
r2   = r2_score(actual_wh, preds_wh)
corr = np.corrcoef(actual_wh, preds_wh)[0, 1]
errors = actual_wh - preds_wh

print(f"\n{'='*60}\n  TLMN v3 Final Metrics (Wh/m²)\n{'='*60}")
print(f"  RMSE        : {rmse:.4f}")
print(f"  MAE         : {mae:.4f}")
print(f"  R²          : {r2:.6f}")
print(f"  Correlation : {corr:.6f}")
print(f"{'='*60}\n")

# ==============================================================================
# SECTION 4 — 6 PUBLICATION-READY PLOTS
# ==============================================================================
def plot_1_temporal_trace():
    fig, ax = plt.subplots(figsize=(16, 5))
    n = min(500, len(actual_wh))
    idx = np.arange(n)
    ax.plot(idx, actual_wh[:n], color="#0369a1", lw=1.8, label="Actual GHI")
    ax.plot(idx, preds_wh[:n], color="#dc2626", lw=1.2, ls="--", label="TLMN v3 Predicted")
    ax.fill_between(idx, actual_wh[:n], preds_wh[:n], alpha=0.15, color="#f59e0b")
    ax.set_xlabel("Hourly Time Step", fontsize=13, fontweight='bold')
    ax.set_ylabel("Solar Irradiance (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_title("TLMN v3 — Temporal Trace: Ground Truth vs Prediction", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot1_TemporalTrace.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot1_TemporalTrace.png")

def plot_2_scatter_regression():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_wh, preds_wh, alpha=0.15, s=8, c="#6366f1", edgecolors='none')
    lim = max(actual_wh.max(), preds_wh.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.5, label="Ideal (y=x)")
    z  = np.polyfit(actual_wh, preds_wh, 1)
    xf = np.linspace(0, lim, 100)
    ax.plot(xf, np.poly1d(z)(xf), color="#dc2626", lw=2, label=f"Fit: y={z[0]:.3f}x+{z[1]:.2f}")
    ax.set_xlabel("Actual GHI (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted GHI (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_title(f"Scatter — R² = {r2:.4f},  Corr = {corr:.4f}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot2_ScatterRegression.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot2_ScatterRegression.png")

def plot_3_error_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=80, color="#6366f1", alpha=0.75, edgecolor='white', lw=0.5, density=True)
    ax.axvline(0, color='#dc2626', lw=2, ls='--', label="Zero Error")
    ax.axvline(np.mean(errors), color='#f59e0b', lw=2, ls='-.', label=f"Mean = {np.mean(errors):.2f}")
    ax.set_xlabel("Prediction Error (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Density", fontsize=13, fontweight='bold')
    ax.set_title("TLMN v3 — Error Distribution Analysis", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot3_ErrorDistribution.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot3_ErrorDistribution.png")

def plot_4_bar_error():
    fig, ax = plt.subplots(figsize=(16, 4))
    n   = min(400, len(errors))
    idx = np.arange(n)
    e   = errors[:n]
    ax.bar(idx, e, color=np.where(e >= 0, "#10b981", "#ef4444"), alpha=0.8, width=1.0)
    ax.axhline(y=0, color="#333", lw=1)
    ax.set_xlabel("Hourly Sample Index", fontsize=13, fontweight='bold')
    ax.set_ylabel("Error (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_title("TLMN v3 — Signed Prediction Error per Hour", fontsize=14, fontweight='bold')
    ax.set_xlim(0, n)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot4_BarError.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot4_BarError.png")

def plot_5_diurnal_pattern():
    nf       = len(actual_wh) - (len(actual_wh) % 24)
    act_24   = actual_wh[:nf].reshape(-1, 24)
    pred_24  = preds_wh[:nf].reshape(-1, 24)
    avg_act  = act_24.mean(0)
    avg_pred = pred_24.mean(0)
    std_act  = act_24.std(0)
    fig, ax  = plt.subplots(figsize=(10, 6))
    hours    = np.arange(24)
    ax.plot(hours, avg_act,  'o-', color="#0369a1", lw=2.5, ms=7, label="Actual Mean")
    ax.plot(hours, avg_pred, 's--', color="#dc2626", lw=2,   ms=6, label="TLMN Mean")
    ax.fill_between(hours, avg_act-std_act, avg_act+std_act, alpha=0.15, color="#0369a1")
    ax.set_xlabel("Hour of Day", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average GHI (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_title("TLMN v3 — Average Diurnal Irradiance Profile", fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot5_DiurnalPattern.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot5_DiurnalPattern.png")

def plot_6_cumulative_error():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.cumsum(np.abs(errors)), color="#7c3aed", lw=2)
    ax.set_xlabel("Sample Index", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cumulative |Error| (Wh/m²)", fontsize=13, fontweight='bold')
    ax.set_title("TLMN v3 — Cumulative Absolute Error Growth", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TLMN_Plot6_CumulativeError.png", dpi=300, facecolor='white')
    print("[SAVED] TLMN_Plot6_CumulativeError.png")

print("\n📊 Generating Complete Evaluation Suite...\n")
plot_1_temporal_trace()
plot_2_scatter_regression()
plot_3_error_distribution()
plot_4_bar_error()
plot_5_diurnal_pattern()
plot_6_cumulative_error()
print(f"\n{'='*60}")
print("  ✅ All 6 TLMN v3 evaluation plots generated.")
print(f"{'='*60}")
