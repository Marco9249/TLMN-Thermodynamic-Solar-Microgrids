<div align="center">

# 🌡️ Thermodynamic Liquid Manifold Networks (TLMN)
### *Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids*

[![arXiv](https://img.shields.io/badge/arXiv-2604.11909-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2604.11909)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Mohammed%20E.%20B.%20Abdullah-blue?style=for-the-badge)](https://github.com/Marco9249)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

</div>

---

## 📄 Abstract

Contemporary deep learning solar forecasting models consistently exhibit two critical failure modes:
1. ☁️ **Severe temporal phase lags** during cloud transients
2. 🌙 **Physically impossible nocturnal power generation**

**TLMN v3** resolves both by projecting **22 meteorological and geometric variables** into a Koopman-linearized Riemannian manifold, combined with a multiplicative **Thermodynamic Alpha-Gate** that structurally enforces celestial geometry compliance.

### Validated Results (5-Year Horizon, Semi-Arid Climate):

| Metric | Value |
|--------|-------|
| **RMSE** | **18.31 Wh/m²** |
| **Pearson Correlation** | **0.988** |
| **Nocturnal Error** | **Zero — across all 1,826 test days** |
| **Phase Response** | **< 30-minute lag during rapid transients** |
| **Parameters** | **63,458 (ultra-lightweight)** |

---

## 🏗️ Model Architecture (v3)

```
NASA POWER Input (22 Features, Window=24h)
            │
   ┌────────▼─────────┐
   │ Hankel Embedding  │  ← Koopman state-space linearization
   │  + Projection     │    (24→20 dynamic windows)
   └────────┬──────────┘
            │
   ┌────────▼──────────────────┐
   │  1D-CNN Temporal Encoder   │  ← 3-layer dilated conv (dilation: 1,2,4)
   │  (replaces LiquidNeuralODE)│    No temporal smoothing lag
   └────────┬──────────────────┘
            │
   ┌────────▼──────────────────────────────┐
   │  Symplectic Cross-Attention            │  ← Physics-guided: Q=meteo, K/V=ClearSky+SZA
   │  After CNN Encoder                     │  ← Attention positioned AFTER temporal encoder
   └────────┬──────────────────────────────┘
            │
   ┌────────▼──────────────────────────────────────┐
   │  Thermodynamic Alpha-Gate (Physics Output)     │
   │  pred = σ(KAN(h_last)) × ClearSky_norm        │  ← Structural zero guarantee at night
   └────────┬──────────────────────────────────────┘
            │
   [GHI Prediction — Wh/m², physically bounded]
```

### 🔬 Architecture Innovation vs. v2

| Component | v2 (LiquidNeuralODE) | v3 (TLMN) |
|-----------|---------------------|------------|
| Temporal model | ODE solver (slow) | Dilated 1D-CNN (fast, no lag) |
| Loss function | MSE + penalties | Log-Cosh (peak-aggressive) |
| Physics enforcement | Penalty terms | Structural gate (zero error, always) |
| **Data window** | **Fixed** | **Sliding window, 3-step stride** |
| **Data source** | **NASA POWER** | **NASA POWER (22 variables)** |

---

## 📂 Project Structure

```
📁 TLMN-Thermodynamic-Solar-Microgrids/
├── 📁 كود التدريب/         # Training pipeline
│   └── TLMN_Model.py       # Full TLMN v3 architecture
├── 📁 كود اختبارات/        # Test evaluation scripts
│   └── TLMN_Test.py
├── 📁 بيانات التدريب/       # NASA POWER datasets
│   └── Hourly_2010_2015.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone & install
git clone https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids.git
cd TLMN-Thermodynamic-Solar-Microgrids
pip install -r requirements.txt

# Run training (ensure NASA POWER CSV is in directory)
python "كود التدريب/TLMN_Model.py"

# Run evaluation
python "كود اختبارات/TLMN_Test.py"
```

---

## 🔗 Related Research by the Same Author

| # | Paper | Repository | arXiv |
|---|-------|------------|-------|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [Physics-Guided-CNN-BiLSTM-Solar](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) | [2604.13455](https://arxiv.org/abs/2604.13455) |
| 2 | Physics-Informed State Space Models (PISSM) | [PISSM-Solar-Forecasting](https://github.com/Marco9249/PISSM-Solar-Forecasting) | [2604.11807](https://arxiv.org/abs/2604.11807) |
| 3 | **Thermodynamic Liquid Manifold Networks (TLMN)** *(this repo)* | [Here](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) | [2604.11909](https://arxiv.org/abs/2604.11909) |
| 4 | Asymmetric-Loss RUL Prediction (Industrial AI) | [Industrial-RUL-Prediction-Architecture](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) | [2604.13459](https://arxiv.org/abs/2604.13459) |

---

## 📖 Citation

```bibtex
@misc{abdullah2026tlmn,
  title   = {Thermodynamic Liquid Manifold Networks: Physics-Bounded
             Deep Learning for Solar Forecasting in Autonomous
             Off-Grid Microgrids},
  author  = {Mohammed Ezzeldin Babiker Abdullah},
  year    = {2026},
  eprint  = {2604.11909},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2604.11909}
}
```

**APA 7th Edition:**
> Abdullah, M. E. B. (2026). *Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids*. arXiv. https://arxiv.org/abs/2604.11909

---

## 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-black?style=flat-square&logo=github)](https://github.com/Marco9249)

---

<div align="center">

© 2026 Mohammed Ezzeldin Babiker Abdullah. All rights reserved.

</div>
