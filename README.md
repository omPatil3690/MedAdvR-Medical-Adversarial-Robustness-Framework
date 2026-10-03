# 🏥 MedAdvR: Medical Adversarial Robustness Framework

> **A specialized, physiologically-constrained adversarial testing and hardening framework for deep learning in medical image segmentation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast%20sync-purple.svg)](https://github.com/astral-sh/uv)
[![Domain](https://img.shields.io/badge/Medical%20AI-Segmentation%20%26%20Robustness-success.svg)]()

---

## 📌 1. Overview & Motivation

Deep learning segmentation models (e.g., UNet, UNet++, Attention UNet) deployed in clinical healthcare settings often exhibit fragile decision boundaries. Small, imperceptible variations in scanner hardware, patient movement, tissue texture, or contrast calibration can cause catastrophic segmentation failures.

**MedAdvR** formulates medical model evaluation and defense as a **constrained min-max adversarial game**:
1. **The Attacker (Generator)**: Synthesizes subtle, domain-specific perturbations targeting tumor boundaries (edges), contrast (intensities), or tissue patterns (frequency spectra) to maximize segmentation failure.
2. **The Realism Constraint**: Enforces that perturbations never exceed clinical plausibility using structural similarity (SSIM) and barrier penalty functions anchored to the original clean scan.
3. **The Defender (Segmentation Model)**: Continuously defends against the generated perturbations via alternating adversarial replay cycles, systematically hardening decision boundaries.

---

## 🔄 2. Architecture & Workflow

```
                             ┌──────────────────────────────────────┐
                             │       Clean Medical Scan (x)         │
                             └──────────────┬───────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
           ┌──────────────────┐                         ┌─────────────────────┐
           │  Clean Baseline  │                         │ Adversarial Attacker│
           │  Pretraining     │                         │ (Generator UNet)    │
           └─────────┬────────┘                         └──────────┬──────────┘
                     │                                             │ Perturbation δ
                     │                                             ▼
                     │                                  ┌─────────────────────┐
                     │                                  │ Perturbed Image     │
                     │                                  │ x_adv = x + ε·tanh(δ)│
                     │                                  └──────────┬──────────┘
                     │                                             │
                     ▼                                             ▼
             ┌────────────────────────────────────────────────────────────┐
             │                 Adversarial Replay Buffer                  │
             │       Combined Dataset: Clean (x) + Adversarial (x_adv)    │
             └─────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                             ┌───────────────────────────┐
                             │    Hardened Defender      │
                             │ (UNet / UNet++ / AttnUNet)│
                             └───────────────────────────┘
```

---

## 📐 3. Mathematical Formulation

### 🔁 Global Objective
$$\max_{G} \mathcal{L}_{\text{attack}}(G, M) \quad \text{subject to} \quad R(x, G(x)) \ge \tau$$
$$\min_{M} \mathcal{L}_{\text{seg}}(M, x \cup G(x))$$

### 🔴 Attacker Objective (Generator)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \mathcal{L}_{\text{specialized}} + \lambda_{\text{real}} \cdot \mathcal{L}_{\text{realism}}$$

* **Attack Failure Loss**:
  $$\mathcal{L}_{\text{attack}} = -(1 - \text{IoU}(\hat{y}_{\text{adv}}, y))$$
  *(Negated so gradient descent maximizes segmentation error)*
* **Specialized Domain Losses**:
  * **Edge Attack**: $\mathcal{L}_{\text{edge}} = - \|\nabla(x_{\text{adv}}) - \nabla x\|_1$ (attacks structural contours)
  * **Intensity Attack**: $\mathcal{L}_{\text{intensity}} = - \|x_{\text{adv}} - x\|_1$ (attacks pixel distributions)
  * **Texture Attack**: $\mathcal{L}_{\text{texture}} = - \|\mathcal{F}(x_{\text{adv}}) - \mathcal{F}(x)\|_1$ (attacks 2D Fourier frequency domain)
* **Realism Constraint (Barrier Function)**:
  $$\mathcal{L}_{\text{realism}} = \text{ReLU}(\tau - R(x, x_{\text{adv}}))$$
  Anchored strictly to original image $x$ to prevent cumulative distortion drift.
* **Adaptive Multiplier ($\lambda$)**:
  $$\lambda \leftarrow \begin{cases} 1.1 \cdot \lambda & \text{if } \mathcal{L}_{\text{realism}} > 10^{-6} \text{ (violation penalty)} \\ 0.9 \cdot \lambda & \text{otherwise (satisfied)} \end{cases}$$

### 🛡️ Defender Objective (Segmentation Model)
$$\mathcal{L}_{\text{seg}} = 0.5 \cdot \text{BCEWithLogitsLoss}(\hat{y}, y) + 0.5 \cdot \mathcal{L}_{\text{IoU/Dice}}(\hat{y}, y)$$

---

## 🗂️ 4. Directory Structure

```
MedAdvR/
├── main.py                     # CLI entrypoint and orchestrator trigger
├── framework.py                # Min-max alternating training coordinator
├── pyproject.toml              # Project dependencies and UV environment config
├── dataset/
│   ├── loader.py               # Unified loader (2D PNG, DICOM, 3D NiFTI)
│   ├── augmentation.py         # Geometric & photometric augmentations
│   └── train_test_split.py     # Deterministic train/test split utilities
├── generator/
│   ├── model.py                # Generator factory
│   ├── unet.py                 # Residual perturbation generator (tanh bounded)
│   └── train.py                # Generator optimization with adaptive lambda
├── main_model/
│   ├── model.py                # Defender factory (UNet, UNet++, Attention UNet)
│   ├── unet.py                 # Standard UNet architecture
│   ├── unet_plusplus.py        # UNet++ with dense nested skip connections
│   ├── unet_attention.py       # Attention UNet with spatial Attention Gates
│   ├── train.py                # Clean pretraining and adversarial retraining
│   └── eval.py                 # Comprehensive metric evaluation (Dice, IoU, TP/FP/FN/TN)
├── utils/
│   ├── gen_losses.py           # Sobel edge, FFT texture, SSIM realism functions
│   ├── metric.py               # Batch-wise Dice and IoU metrics & loss
│   ├── train_helper.py         # DataLoader resolvers & helpers
│   └── save.py                 # Adversarial image & mask export utilities
├── visualisation/
│   └── train_vis.py            # Qualitative visualization tools
└── train_eval/                 # Saved evaluation artifacts and metrics
```

---

## ⚙️ 5. Setup & Installation

This project is managed using [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency resolution.

### 1. Clone and Install Dependencies
```bash
# Sync all dependencies via uv
uv sync
```

### 2. Supported Medical Data Formats
The data loader supports multiple medical imaging layouts:
* **2D Paired PNG / JPEG**: Image (`img.png`) and binary mask (`img_mask.png`).
* **DICOM**: Standard `.dcm` single or multi-frame files.
* **3D NiFTI (`.nii.gz`)**: Full 3D MRI/CT volumes sliced across depth.

---

## 🚀 6. Usage & Execution

### Basic Adversarial Training Run
```bash
uv run main.py \
  --dataset_path ./glioma/DATASET/Segmentation/Glioma \
  --dataset_type tumor \
  --model_type Unet \
  --gen_type edge \
  --device cuda \
  --batch_size 16 \
  --img_size 128 \
  --pretrain_epochs 10 \
  --cycles 5 \
  --gen_epochs 3 \
  --model_epochs 3 \
  --save_images
```

### Command-Line Arguments Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_path` | *Required* | Path to the medical dataset root |
| `--dataset_type` | `tumor` | Format type: `tumor`, `extracted`, `dicom`, `nifti` |
| `--model_type` | `Unet` | Segmentation model: `Unet`, `Unet++`, `Unet++Attention` |
| `--gen_type` | `edge` | Attack generator mode: `edge`, `intensity`, `texture` |
| `--device` | `cpu` | Hardware target: `cuda` or `cpu` |
| `--batch_size` | `16` | Batch size for training and evaluation |
| `--img_size` | `128` | Image resolution $(H, W)$ |
| `--pretrain_epochs` | `5` | Epochs to train model on clean data before attacks |
| `--cycles` | `5` | Number of alternating attacker-defender cycles |
| `--gen_epochs` | `3` | Generator optimization epochs per cycle |
| `--model_epochs` | `3` | Defender retraining epochs per cycle |
| `--save_images` | `False` | Flag to export adversarial samples to disk |
| `--save_dir` | `outputs` | Directory to save generated images and logs |
| `--max_buffer_size` | `5000` | Maximum capacity of adversarial replay buffer |

---

## 📊 7. Evaluation & Metrics

The framework evaluates segmentation quality across multiple clinical axes:
* **Dice Similarity Coefficient (DSC)**: Overlap fidelity for lesion boundaries.
* **Jaccard Index (IoU)**: Intersection over union area ratio.
* **Precision & Recall**: Delineates false alarm rate vs. missed pathology rate.
* **Pixel Statistics**: True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN).
* **Visual Evaluation**: Output prediction comparisons exported to `train_eval/eval_batch_*.png`.

---

## 🔮 8. Roadmap & Future Innovations

1. **Anatomical Deformations (STN & DVF)**: Integrate Spatial Transformer Networks to simulate non-rigid patient anatomical deformations.
2. **Scanner Physics Ingestion**: Add learnable Rician noise injectors and MRI $B_1$ magnetic field inhomogeneity (bias field) attacks.
3. **Advanced Clinical Metrics**: Add 95% Hausdorff Distance ($HD_{95}$) and Normalized Surface Dice (NSD).
4. **Epistemic Uncertainty Estimation**: Flag perturbed regions via Monte Carlo Dropout uncertainty heatmaps to notify radiologists.
5. **Interactive Web UI**: Real-time stress-testing playground for radiologists to probe model boundaries live.