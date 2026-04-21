<div align="center">

<img src="https://img.shields.io/badge/%F0%9F%8C%A1%EF%B8%8F-Thermodynamic%20AI-FF6B6B?style=for-the-badge&labelColor=1a1a2e" alt="Thermodynamic AI"/>

# Thermodynamic Liquid Manifold Networks (TLMN)

### 🌍 *Physics-Bounded Deep Learning for Autonomous Off-Grid Microgrids* 🌍

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2604.11909-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.11909)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-005288?style=for-the-badge&logo=nasa&logoColor=white)](https://power.larc.nasa.gov/)

<br/>

<img src="https://img.shields.io/badge/Author-Mohammed%20Ezzeldin%20Babiker%20Abdullah-4A90D9?style=flat-square&logo=google-scholar&logoColor=white" alt="Author"/>

---

*"Zero-magnitude nocturnal error across all 1,826 testing days."*

</div>

> [!IMPORTANT]
> **Implementation Note**: This repository contains the core architecture and settings as described in the associated research paper. However, some code structures and experimental configurations have been slightly adjusted to facilitate educational study, modification, and independent testing. The codebase will be fully synchronized with the exact methodology presented in the manuscript upon the paper's final formal publication.

---

## 🎯 The Problem We Solved

> Contemporary solar forecasting models fail in two catastrophic ways:

| Failure Mode | Impact | TLMN Solution |
|:------------:|:------:|:--------------:|
| ☁️ **Temporal Phase Lag** | Delayed cloud response | Dilated 1D-CNN (zero lag) |
| 🌙 **Phantom Nocturnal Generation** | Impossible predictions | Thermodynamic Alpha-Gate |

### 🏆 Results (5-Year Horizon, Semi-Arid Climate)

<div align="center">

| Metric | Value |
|:------:|:-----:|
| 📉 **RMSE** | **18.31 Wh/m²** |
| 📊 **Pearson Correlation** | **0.988** |
| 🌙 **Nocturnal Error** | **Zero** (all 1,826 days) |
| ⚡ **Phase Response** | **< 30 min** during rapid transients |
| 🧮 **Parameters** | **63,458** (ultra-lightweight) |

</div>

---

## 🏗️ Architecture (v3)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ☀️  NASA POWER Input (22 Variables, 24h Window)   │
│       Meteorological + Geometric + Derived          │
│                        │                            │
│              ┌─────────▼──────────┐                 │
│              │  Hankel Embedding  │  Koopman         │
│              │  + Tanh Projection │  linearization   │
│              │  + Positional Enc. │                  │
│              └─────────┬──────────┘                 │
│                        │                            │
│       ┌────────────────▼────────────────┐           │
│       │  🔥 Dilated 1D-CNN Encoder      │           │
│       │  3 layers × dilation [1,2,4]    │  Zero-lag │
│       │  Receptive field = 13 steps     │  temporal │
│       │  (replaces LiquidNeuralODE)     │  encoding │
│       └────────────────┬────────────────┘           │
│                        │                            │
│       ┌────────────────▼────────────────┐           │
│       │  🎯 Symplectic Cross-Attention  │  Physics  │
│       │  Q = meteorological features    │  guided   │
│       │  K/V = ClearSky + SZA           │  (after   │
│       │  γ-weighted residual            │  encoder) │
│       └────────────────┬────────────────┘           │
│                        │                            │
│    ┌───────────────────▼───────────────────┐        │
│    │  🌡️ Thermodynamic Alpha-Gate          │        │
│    │  pred = σ(KAN(h)) × ClearSky_norm    │  Night  │
│    │  Structural zero guarantee           │  = 0    │
│    └───────────────────┬───────────────────┘        │
│                        │                            │
│             📊 GHI Prediction (Wh/m²)               │
│             Physically bounded, always              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 🔬 v2 → v3 Evolution

| Component | v2 | v3 (Current) |
|:---------:|:--:|:------------:|
| Temporal Encoder | LiquidNeuralODE (slow) | Dilated 1D-CNN (fast, no lag) |
| Loss Function | MSE + penalty terms | Log-Cosh (peak-aggressive) |
| Physics Enforcement | Loss penalties | Structural gate (100% guarantee) |
| Sliding Window | Fixed | **3-step stride** |
| Input Features | 15 | **22 variables** |

---

## 📂 Repository Structure

```
📦 TLMN-Thermodynamic-Solar-Microgrids/
│
├── 📁 training_code/
│   └── 🧠 TLMN_Model.py                 # Full TLMN v3 architecture
│
├── 📁 evaluation_code/
│   └── 📊 TLMN_Test.py                   # Test evaluation pipeline
│
├── 📁 training_data/
│   ├── 📊 Hourly_2010_2015.csv           # NASA POWER hourly data
│   └── 📊 Hourly_2020_2025.csv
│
├── 📄 TLMN_Paper.pdf                     # Published paper
├── 📄 TLMN_Paper.docx
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids.git
cd TLMN-Thermodynamic-Solar-Microgrids
pip install -r requirements.txt

# Train TLMN v3
python training_code/TLMN_Model.py

# Evaluate
python evaluation_code/TLMN_Test.py
```

---

## 📚 Related Research Papers

<div align="center">

| # | Paper | Repository | arXiv |
|:-:|:------|:----------:|:-----:|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) | [![arXiv](https://img.shields.io/badge/-2604.13455-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13455) |
| 2 | Physics-Informed State Space Model (PI-SSM) | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-SSM-Solar-Forecasting) | [![arXiv](https://img.shields.io/badge/-2604.11807-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11807) |
| **3** | **TLMN** *(this repo)* 🌟 | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) | [![arXiv](https://img.shields.io/badge/-2604.11909-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11909) |
| 4 | Asymmetric-Loss Industrial RUL Prediction | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) | [![arXiv](https://img.shields.io/badge/-2604.13459-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13459) |
| 🎮 | Interactive 3D Architecture Visualization | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-Hybrid-3D-Viz) | — |

</div>

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

> **APA 7th Edition:**
> Abdullah, M. E. B. (2026). *Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids*. arXiv. https://arxiv.org/abs/2604.11909

---

<div align="center">

### 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-181717?style=for-the-badge&logo=github)](https://github.com/Marco9249)

---

© 2026 Mohammed Ezzeldin Babiker Abdullah — All rights reserved.

</div>

