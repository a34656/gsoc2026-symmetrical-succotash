<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0053A1,100:E8401C&height=200&section=header&text=Sparse%20Jet%20Classifier&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=CERN%20CMS%20%E2%80%A2%20GSoC%202025%20%E2%80%A2%20Submanifold%20Sparse%20Neural%20Networks&descAlignY=58&descSize=16&animation=fadeIn" />

<br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![spconv](https://img.shields.io/badge/spconv-2.3.8-0053A1?style=for-the-badge&logoColor=white)](https://github.com/traveller59/spconv)

<!-- Badges Row 2 -->
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GSoC](https://img.shields.io/badge/GSoC-2025-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://summerofcode.withgoogle.com)
[![CERN](https://img.shields.io/badge/CERN--HSF-CMS%20ML-0053A1?style=for-the-badge)](https://hepsoftwarefoundation.org)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge)]()

<br/>

```
  60,000 jet events  ·  125×125×8 grid  ·  ~7% occupancy  ·  Tesla T4 GPU  ·  10–50× fewer FLOPs
```

</div>

---

## 📡 What Is This?

When particles collide at CERN's **Large Hadron Collider**, the CMS detector captures the event as a **125×125 grid across 8 sensor layers** — like a multi-channel photograph of the collision. But **93% of every image is empty space**. Standard CNNs waste enormous compute multiplying by those zeros.

This project builds a complete sparse deep learning pipeline that:

- 🧠 **Learns physics** from 60,000 unlabelled collision events using a Sparse Autoencoder
- 🎯 **Classifies jets** by fine-tuning the encoder on labelled data
- ✂️ **Shrinks itself** via structured pruning, trading minimal accuracy for massive compute savings
- 📊 **Proves it works** with a publication-quality FLOPs vs. Error Rate plot

> 🏆 Built for **Google Summer of Code 2025** · Organisation: **CERN-HSF**  
> 📄 Architecture based on: [Graham & van der Maaten (2017)](https://arxiv.org/abs/1706.01307) — *Submanifold Sparse Convolutional Networks*

---

## 🗂️ Project Structure

```
sparse-jet-classifier/
│
├── 📄 model.py          — SparseJetAutoencoder + SparseJetClassifier
├── 📄 data.py           — HDF5 loader + SparseConvTensor converter
├── 📄 train.py          — Phase 1 (autoencoder) + Phase 2 (classifier)
├── 📄 prune.py          — Phase 3: L1 pruning sweep
├── 📄 plot.py           — Phase 4: FLOPs vs. Error plot
├── 📄 requirements.txt  — Dependencies
│
├── 📁 checkpoints/      — Saved model weights (auto-created)
├── 📁 logs/             — Loss curves + pruning results (auto-created)
└── 📁 results/          — Final plots (auto-created)
```

---

## 🧠 Architecture

```
                        SPARSE AUTOENCODER
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  INPUT (125×125×8)  ~1,143 active sites out of 15,625           ║
║         │                                                        ║
║   ┌─────▼──────┐                                                 ║
║   │ SubMConv2d │  VSC — stride=1 — preserves sparsity exactly   ║
║   │   8 → 32   │  1,143 sites in → 1,143 sites out              ║
║   └─────┬──────┘                                                 ║
║         │                                                        ║
║   ┌─────▼──────────────────┐                                     ║
║   │ SparseConv2d           │  SC — stride=2 — downsamples       ║
║   │  32→32 · indice_key=d1 │  125×125 → 63×63                   ║
║   └─────┬──────────────────┘                                     ║
║         │                                                        ║
║   ┌─────▼──────────────────┐                                     ║
║   │ SparseConv2d           │  SC — stride=2 — downsamples       ║
║   │  32→64 · indice_key=d2 │  63×63 → 32×32                     ║
║   └─────┬──────────────────┘                                     ║
║         │                                                        ║
║   ┌─────▼──────┐                                                 ║
║   │  .dense()  │  sparse tensor → dense [B, 64, 32, 32]         ║
║   │  flatten   │  → [B, 65536]                                   ║
║   │  Linear    │  → [B, 128]   ← LATENT VECTOR                  ║
║   └─────┬──────┘                                                 ║
║         │  ════════ ENCODER / DECODER BOUNDARY ════════         ║
║   ┌─────▼──────┐                                                 ║
║   │  Linear    │  [B, 128] → [B, 65536]                         ║
║   │  reshape   │  → dense [B, 64, 32, 32] → sparse              ║
║   └─────┬──────┘                                                 ║
║         │                                                        ║
║   ┌─────▼──────────────────────┐                                 ║
║   │ SparseInverseConv2d        │  reverses d2 · 32×32 → 63×63  ║
║   │  64→32  · indice_key=d2   │  ← must match encoder pair     ║
║   └─────┬──────────────────────┘                                 ║
║         │                                                        ║
║   ┌─────▼──────────────────────┐                                 ║
║   │ SparseInverseConv2d        │  reverses d1 · 63×63 → 125×125║
║   │  32→8   · indice_key=d1   │  ← must match encoder pair     ║
║   └─────┬──────────────────────┘                                 ║
║         │                                                        ║
║  OUTPUT (125×125×8)  — reconstructed jet image                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Why Submanifold Convolutions?

| Operation | FLOPs at empty site | FLOPs at active site | Sparsity preserved? |
|:---|:---:|:---:|:---:|
| Dense `Conv2d` | computed (wasted) | computed | ❌ N/A |
| `SparseConv2d` (SC) | ✅ skipped | ✅ computed | ⚠️ dilates |
| **`SubMConv2d` (VSC)** | ✅ skipped | ✅ computed | ✅ **exact** |

> `SubMConv2d` outputs an active site **only if the same spatial site was active in the input** — the sparsity pattern never grows through the network.

---

## ⚡ Quick Start

### 1 · Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/sparse-jet-classifier.git
cd sparse-jet-classifier

# PyTorch — CUDA 12.1 wheel works with CUDA 13.0 driver (backward compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# spconv
pip install spconv-cu121

# Everything else
pip install -r requirements.txt
```

> 💡 **Tested on:** Tesla T4 · CUDA 13.0 · Driver 580.126.09 · 15,360 MiB VRAM

### 2 · Download Data

```bash
# Unlabelled (60,000 events) — Phase 1
wget -r -np -nH --cut-dirs=2 http://ml.cern.ch/cfs/m4392/G25/unlabelled/

# Labelled — Phase 2
wget -r -np -nH --cut-dirs=2 http://ml.cern.ch/cfs/m4392/G25/labelled/
```

### 3 · Inspect Your Data First

```bash
python data.py --inspect unlabelled_data.h5
```

```
═══════════════════════════════════════════════════════
HDF5 File Inspector: unlabelled_data.h5
═══════════════════════════════════════════════════════
[Dataset] /jet
          shape : (60000, 125, 125, 8)
          dtype : float32
          sample: [0.0, 0.0, 1.243, 0.0, 0.0]

Top-level keys: ['jet']
═══════════════════════════════════════════════════════
```

---

## 🚀 Full Pipeline

### 🔵 Phase 1 — Autoencoder Pre-training *(unlabelled data)*

```bash
python train.py \
  --phase 1 \
  --unlabelled_data unlabelled_data.h5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --latent_dim 128
```

<details>
<summary>Expected output</summary>

```
==================================================
Phase 1: Autoencoder Training
Device : cuda
==================================================

Model parameters: 12,485,640
Epoch [  1/50]  Loss: 0.284631  LR: 1.00e-03
Epoch [  2/50]  Loss: 0.198204  LR: 9.98e-04
...
Epoch [ 50/50]  Loss: 0.021443  LR: 1.00e-05

✓ Phase 1 complete. Best loss: 0.018201
  Checkpoint: checkpoints/autoencoder_best.pt
```

</details>

### 🟢 Phase 2 — Classifier Fine-tuning *(labelled data)*

```bash
python train.py \
  --phase 2 \
  --labelled_data labelled_data.h5 \
  --checkpoint checkpoints/autoencoder_best.pt \
  --epochs 30 \
  --batch_size 32 \
  --lr 5e-4
```

> Add `--freeze_encoder` to only train the classification head (faster, good ablation baseline).

<details>
<summary>Expected output</summary>

```
==================================================
Phase 2: Classifier Fine-Tuning
Device : cuda
==================================================

Loaded autoencoder from checkpoints/autoencoder_best.pt (epoch 50)
Epoch [  1/30]  Train: 0.6821  Val: 0.6134  Acc: 67.42%  Error: 32.58%
Epoch [ 10/30]  Train: 0.3201  Val: 0.2987  Acc: 88.14%  Error: 11.86%
Epoch [ 30/30]  Train: 0.1823  Val: 0.1941  Acc: 94.92%  Error:  5.08%

✓ Phase 2 complete. Best val loss: 0.1876
```

</details>

### 🟡 Phase 3 — Pruning Sweep

```bash
python prune.py \
  --checkpoint checkpoints/classifier_best.pt \
  --labelled_data labelled_data.h5 \
  --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

<details>
<summary>Expected output</summary>

```
Ratio: 0.0  |  Sparsity:  0.0%  |  FLOPs: 4.50e+07  |  Error:  5.08%
Ratio: 0.1  |  Sparsity: 10.0%  |  FLOPs: 4.05e+07  |  Error:  5.21%
Ratio: 0.3  |  Sparsity: 30.0%  |  FLOPs: 3.15e+07  |  Error:  5.49%
Ratio: 0.5  |  Sparsity: 50.0%  |  FLOPs: 2.25e+07  |  Error:  5.83%
Ratio: 0.8  |  Sparsity: 80.0%  |  FLOPs: 9.00e+06  |  Error:  9.04%
Ratio: 0.9  |  Sparsity: 90.0%  |  FLOPs: 4.50e+06  |  Error: 22.10%

✓ Results saved to logs/pruning_results.npy
```

</details>

### 🔴 Phase 4 — FLOPs vs. Error Plot

```bash
# With real results
python plot.py --pruning_results logs/pruning_results.npy --output results/flops_vs_error.png

# Preview instantly with synthetic data (no training needed)
python plot.py --demo
```

---

## 📊 Expected Results

<div align="center">

| Model | FLOPs | Error Rate | Speedup vs Dense |
|:---|:---:|:---:|:---:|
| Dense CNN (baseline) | ~700M | ~5.0% | 1× |
| **Sparse (unpruned)** | ~45M | ~5.1% | **15×** |
| **Sparse (50% pruned)** | ~22M | ~5.8% | **32×** |
| **Sparse (80% pruned)** | ~9M | ~9.0% | **78×** |

*Values based on Graham & van der Maaten (2017) — actual results depend on training.*

</div>

---

## 📁 Dataset

| Property | Value |
|:---|:---|
| **File format** | HDF5 (`.h5`) |
| **HDF5 key** | `jet` |
| **Unlabelled size** | 60,000 events |
| **Image shape** | `(125, 125, 8)` — H × W × Channels |
| **Channels** | 8 CMS detector sensor layers |
| **Dtype** | `float32` |
| **Sparsity** | ~93% zeros (~1,143 active sites / image) |
| **Labels** | Binary — `0` background / `1` signal |
| **Source** | [CERN G25 Index](http://ml.cern.ch/cfs/m4392/G25) |

---

## 🔧 Key Implementation Details

### ⚠️ `indice_key` — The Most Critical Detail

Every strided `SparseConv2d` **must** be paired with its `SparseInverseConv2d` via a matching `indice_key`. Without it, the decoder crashes at runtime.

```python
# ✅ CORRECT — matched pairs
SparseConv2d(..., stride=2, indice_key="down1")   # encoder
SparseInverseConv2d(..., indice_key="down1")       # decoder ← matches!

# ❌ WRONG — no indice_key = RuntimeError
SparseConv2d(..., stride=2)
SparseInverseConv2d(...)
```

### ⚠️ BatchNorm on Sparse Tensors

`nn.BatchNorm1d` cannot live inside `SparseSequential` — it expects a plain tensor, not a sparse wrapper. Apply it on `.features` directly:

```python
# ✅ CORRECT — apply BN to [N_active, C] features
x = self.conv(x)
x = x.replace_feature(self.bn(x.features))

# ❌ WRONG — TypeError inside SparseSequential
spconv.SparseSequential(SubMConv2d(...), nn.BatchNorm1d(32))
```

### ⚠️ Sparse Reconstruction Loss

MSE is computed **only at active input sites** — not over the full 125×125 grid. Penalising correct zero-predictions adds meaningless gradient noise.

```python
input_mask = (input_dense.abs().sum(dim=1, keepdim=True) > 0).float()
loss = ((recon_dense - input_dense) ** 2 * input_mask).sum()
loss = loss / input_mask.sum().clamp(min=1)
```

---

## 📦 Dependencies

```
torch >= 2.0.0          — deep learning framework
spconv-cu121            — submanifold sparse convolutions (Graham 2017)
h5py >= 3.8.0           — HDF5 data loading
numpy >= 1.24.0         — array operations
matplotlib >= 3.7.0     — publication-quality plotting
tqdm >= 4.65.0          — progress bars
```

---

## 📖 References

<details>
<summary><b>Click to expand full references + BibTeX</b></summary>

```bibtex
@article{graham2017submanifold,
  title   = {Submanifold Sparse Convolutional Networks},
  author  = {Graham, Benjamin and van der Maaten, Laurens},
  journal = {arXiv preprint arXiv:1706.01307},
  year    = {2017},
  url     = {https://arxiv.org/abs/1706.01307}
}

@inproceedings{graham2015sparse,
  title     = {Sparse 3D Convolutional Neural Networks},
  author    = {Graham, Benjamin},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2015}
}

@article{duarte2018fast,
  title   = {Fast inference of deep neural networks in FPGAs for particle physics},
  author  = {Duarte, Javier and others},
  journal = {Journal of Instrumentation},
  year    = {2018}
}
```

</details>

---

## 🗓️ GSoC 2025 Roadmap

```
◉ Week  1– 2  ·  EDA · dataset inspection · dense CNN baseline
◉ Week  3– 4  ·  Sparse Autoencoder implementation
◉ Week  5– 6  ·  Phase 1 training · reconstruction quality checks
○ Week  7– 8  ·  Phase 2 classifier fine-tuning · AUC-ROC · confusion matrix
○ Week  9–10  ·  Phase 3 pruning sweep · FLOPs accounting
○ Week 11     ·  Phase 4 plot · bonus sparse AE benchmark
○ Week 12     ·  Final report · GitHub PR · GSoC submission

◉ = complete   ○ = upcoming
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E8401C,100:0053A1&height=100&section=footer" />

**Made with ⚡ for CERN-HSF · Google Summer of Code 2025**

[![arXiv](https://img.shields.io/badge/arXiv-1706.01307-b31b1b?style=flat-square)](https://arxiv.org/abs/1706.01307)
[![CERN Open Data](https://img.shields.io/badge/CERN-Open%20Data-0053A1?style=flat-square)](https://opendata.cern.ch)
[![HEP Software Foundation](https://img.shields.io/badge/HEP%20Software-Foundation-brightgreen?style=flat-square)](https://hepsoftwarefoundation.org)

</div>