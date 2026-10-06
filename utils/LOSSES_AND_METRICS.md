# 📐 MedAdvR: Comprehensive Mathematical & Intuitive Guide to Losses and Metrics

> **A detailed mathematical breakdown and intuitive reference for all loss functions, similarity metrics, and optimization objectives in the MedAdvR framework.**

---

## 🧭 1. Executive Mental Model

The adversarial testing framework operates on two distinct optimization fronts:
1. **The Segmentation Defender**: Minimizes segmentation error on both clean and adversarial scans.
2. **The Adversarial Generator (Attacker)**: Maximizes segmentation failure while satisfying strict clinical realism bounds.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   Clean Scan (x)                        │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │    Generator    │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        Adversarial Image
                                             (x_adv)
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
         ┌───────────────────┐                                       ┌───────────────────┐
         │ Specialized Loss  │                                       │   Realism Score   │
         │   (L_special)     │                                       │   R(x, x_adv)     │
         └─────────┬─────────┘                                       └─────────┬─────────┘
                   │                                                           │
                   ▼                                                           ▼
           Edge / Intensity /                                           Barrier Penalty
                Texture                                                  (L_realism)
```

---

## 🎯 2. Segmentation Metrics & Losses (`utils/metric.py`)

Medical image segmentation evaluates the spatial overlap between the model's prediction ($\hat{Y}$) and the ground truth mask ($Y$).

```
                 SEGMENTATION
                      │
            ┌─────────┴─────────┐
            │                   │
       Prediction          Ground Truth
           (A)                  (B)
            │                   │
            └─────────┬─────────┘
                      │
                   OVERLAP
                      │
             ┌────────┴────────┐
             │                 │
           Dice               IoU
             │                 │
     2 × intersection     intersection
     ────────────────     ────────────
       |A| + |B|          |A ∪ B|
             │                 │
             ▼                 ▼
        Dice Score         IoU Score
             │                 │
             ▼                 ▼
      1 - Dice Score      1 - IoU Score
             │                 │
             ▼                 ▼
         Dice Loss           IoU Loss
```

---

### A. Dice Similarity Coefficient (DSC / F1-Score)
Measures harmonic overlap between prediction $A$ and ground truth $B$:

$$\text{Dice}(A, B) = \frac{2 \cdot |A \cap B|}{|A| + |B| + \epsilon}$$

* **Intuition**: Gives double weight to the overlapping region relative to total pixel count.
* **Code Implementation (`dice_score`)**:
  ```python
  intersection = (preds * targets).sum(dim=(1, 2, 3))
  cardinality = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
  dice = (2.0 * intersection + eps) / (cardinality + eps)
  ```

### B. Soft Dice Loss
Continuous, differentiable loss using predicted sigmoid probabilities $p \in [0, 1]$:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p \cdot y) + \epsilon}{\sum p + \sum y + \epsilon}$$

---

### C. Jaccard Index / Intersection over Union (IoU)
Measures the ratio of overlapping area to total combined area:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B| + \epsilon}$$

* **Critical Distinction**: 
  $$\text{Union} = |A| + |B| - \text{Intersection}$$
  Subtracting the intersection prevents double-counting the overlapping pixels.
* **Mathematical Relation to Dice**:
  $$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}, \quad \text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
* **Code Implementation (`iou_score`)**:
  ```python
  intersection = (preds * targets).sum(dim=(1, 2, 3))
  union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
  iou = (intersection + eps) / (union + eps)
  ```

### D. Soft IoU / Jaccard Loss
$$\mathcal{L}_{\text{IoU}} = 1 - \text{IoU}$$

---

## ⚔️ 3. The Generator Attack Objective (`L_attack`)

In standard training, we minimize $\mathcal{L}_{\text{IoU}} = 1 - \text{IoU}$ to make segmentation better.  
For the **adversarial attacker**, we want to **maximize segmentation failure (minimize IoU)**:

$$\mathcal{L}_{\text{attack}} = -\mathcal{L}_{\text{IoU}} = -(1 - \text{IoU}(\hat{y}_{\text{adv}}, y)) = \text{IoU}(\hat{y}_{\text{adv}}, y) - 1$$

### Why the Negative Sign?
When gradient descent minimizes $\mathcal{L}_{\text{total}}$:
* If $\text{IoU} = 0.90 \implies \mathcal{L}_{\text{attack}} = -0.10$
* If $\text{IoU} = 0.05 \implies \mathcal{L}_{\text{attack}} = -0.95$ *(Much lower loss $\implies$ preferred by optimizer!)*

Minimizing this negative loss directly forces the generator to drive the model's IoU down to zero.

---

## 🎨 4. Specialized Perturbation Losses (`utils/gen_losses.py`)

To target specific physical/clinical failure modes, the generator uses specialized domain losses.

```
                           Specialized Losses
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
        Edge Loss            Intensity Loss          Texture Loss
       (Gradients)           (Pixel Values)         (Fourier FFT)
            │                      │                      │
     -|∇x_adv - ∇x|           -|x_adv - x|          -|F(x_adv) - F(x)|
```

---

### A. Edge Loss (`edge_loss`)
Attacks anatomical boundaries, tumor contours, and tissue transitions.

```python
def edge_loss(x_adv, x):
    grad_adv = torch.gradient(x_adv, dim=[2, 3])
    grad_x = torch.gradient(x, dim=[2, 3])
    diff_y = torch.abs(grad_adv[0] - grad_x[0])
    diff_x = torch.abs(grad_adv[1] - grad_x[1])
    return -torch.mean(diff_y + diff_x)
```

#### Mathematical Formulation:
$$\mathcal{L}_{\text{edge}} = - \frac{1}{HW} \sum_{h,w} \left( |\nabla_y x_{\text{adv}} - \nabla_y x| + |\nabla_x x_{\text{adv}} - \nabla_x x| \right)$$

* **How it works**: Computes finite spatial differences along height (dim 2) and width (dim 3).
* **Intuition**: Forces the generator to aggressively modify high-frequency boundary gradients where tumor borders are defined.

---

### B. Intensity Loss (`intensity_loss`)
Attacks contrast calibration, scanner gain, and voxel brightness.

```python
def intensity_loss(x_adv, x):
    return -torch.mean(torch.abs(x_adv - x))
```

#### Mathematical Formulation:
$$\mathcal{L}_{\text{intensity}} = - \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} |x_{\text{adv}} - x|$$

* **Intuition**: Direct $L_1$ pixel displacement. Drives contrast shifting across healthy tissue vs. lesion areas.

---

### C. Texture / Frequency Loss (`texture_loss`)
Attacks tissue grain, RF scanner noise, and frequency spectral characteristics using 2D Fast Fourier Transforms.

```python
def texture_loss(x_adv, x):
    f_x = torch.fft.fft2(x)
    f_x_adv = torch.fft.fft2(x_adv)
    return -torch.mean(torch.abs(f_x - f_x_adv))
```

#### Mathematical Formulation:
$$\mathcal{L}_{\text{texture}} = - \frac{1}{HW} \sum_{u,v} |\mathcal{F}(x_{\text{adv}})(u,v) - \mathcal{F}(x)(u,v)|$$

* **How it works**: Transforms spatial images into complex 2D frequency spectra. `torch.abs(...)` computes the Euclidean magnitude of the complex difference.
* **Intuition**: Forces the generator to inject frequency-domain anomalies without creating obvious localized spatial blobs.

---

### D. Loss vs. Score Distinction
* **`edge_loss`**: **Negative** (Used during training to maximize edge divergence via gradient descent).
* **`edge_diff_score`**: **Positive** (Used as an evaluation metric to measure edge change magnitude).

---

## 🛡️ 5. Structural Realism & Barrier Penalties

Without constraints, an adversarial generator would produce meaningless noise patterns. MedAdvR introduces a **Domain-Preserving Realism Score** $R(x, x_{\text{adv}})$.

### Key Design Principle:
> **Preserve whatever the generator is NOT actively attacking.**

---

### A. Structural Similarity Index (SSIM)
Evaluates luminance ($l$), contrast ($c$), and structural correlation ($s$):

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

* $\text{SSIM} \approx 1.0 \implies$ Near identical visual structure.
* Computed per image across the batch and averaged in `compute_ssim()`.

---

### B. Specialized Realism Scoring Functions (`realism_score`)

| Generator Mode | Formula for Realism Score $R(x, x_{\text{adv}})$ | Clinical Intuition |
| :--- | :--- | :--- |
| **Edge Attack** | $0.5 \cdot \text{SSIM} + 0.5 \cdot (1 - \text{IntensityDeviation})$ | Modify edges, but preserve global contrast and brightness |
| **Intensity Attack** | $0.5 \cdot \text{SSIM} + 0.5 \cdot (1 - \text{EdgeDiffScore})$ | Modify brightness, but preserve anatomical organ boundaries |
| **Texture Attack** | $0.5 \cdot \text{SSIM} + 0.5 \cdot (1 - \text{FrequencyDeviation})$ | Modify tissue grain, but prevent extreme spectral distortion |

---

### C. The Barrier Penalty Function (`realism_loss`)

To convert the constraint $R(x, x_{\text{adv}}) \ge \tau$ into a differentiable loss, we use a **one-sided ReLU barrier penalty**:

$$\mathcal{L}_{\text{realism}} = \text{ReLU}(\tau - R(x, x_{\text{adv}})) = \max(0, \tau - R)$$

```
                    τ - R (Deficit)
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
          R >= τ                     R < τ
      (Satisfied)                 (Violated)
             │                         │
             ▼                         ▼
          <= 0                        > 0
             │                         │
             ▼                         ▼
         Loss = 0             Loss = Penalty (τ - R)
```

* If $R = 0.95 \ge 0.90 \implies \mathcal{L}_{\text{realism}} = \max(0, -0.05) = \mathbf{0.0}$ (No penalty).
* If $R = 0.82 < 0.90 \implies \mathcal{L}_{\text{realism}} = \max(0, 0.08) = \mathbf{0.08}$ (Penalty activates).

---

### D. Adaptive Penalty Coefficient ($\lambda_{\text{real}}$)

Rather than fixing $\lambda$, the framework dynamically adapts constraint pressure:

$$\lambda_{\text{real}} \leftarrow \begin{cases} \min(\lambda \times 1.1, 10.0) & \text{if } \mathcal{L}_{\text{realism}} > 10^{-6} \text{ (Increase penalty on violation)} \\ \max(\lambda \times 0.9, 0.01) & \text{otherwise (Relax penalty when realistic)} \end{cases}$$

This drives training toward an exact equilibrium where $R(x, x_{\text{adv}}) \approx \tau = 0.90$.

---

## 🔒 6. Dual Perturbation Control Mechanism

MedAdvR uses **two complementary layers of defense** against unrealistic perturbations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. HARD ARCHITECTURAL BOUND (generator/unet.py)                            │
│    x_adv = x + 0.05 * tanh(out(x))                                          │
│    → Guarantees absolute pixel delta is strictly bounded: |δ| ≤ 0.05 (5%)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. SOFT MANIFOLD CONSTRAINT (utils/gen_losses.py)                           │
│    L_real = ReLU(0.90 - R(x, x_adv))                                        │
│    → Enforces global structural similarity and domain preservation          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 7. Summary Formula Matrix

| Component | Mathematical Expression | Optimization Direction |
| :--- | :--- | :--- |
| **Attack Objective** | $\mathcal{L}_{\text{attack}} = -(1 - \text{IoU}(\hat{y}_{\text{adv}}, y))$ | Minimizing drives IoU to $0$ (Total failure) |
| **Edge Objective** | $\mathcal{L}_{\text{edge}} = -\|\nabla x_{\text{adv}} - \nabla x\|_1$ | Minimizing maximizes edge distortion |
| **Intensity Objective**| $\mathcal{L}_{\text{intensity}} = -\|x_{\text{adv}} - x\|_1$ | Minimizing maximizes contrast shift |
| **Texture Objective** | $\mathcal{L}_{\text{texture}} = -\|\mathcal{F}(x_{\text{adv}}) - \mathcal{F}(x)\|_1$ | Minimizing maximizes frequency corruption |
| **Realism Penalty** | $\mathcal{L}_{\text{realism}} = \text{ReLU}(\tau - R(x, x_{\text{adv}}))$ | Minimizing forces $R(x, x_{\text{adv}}) \ge \tau$ |
| **Total Generator Loss**| $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \mathcal{L}_{\text{special}} + \lambda \mathcal{L}_{\text{realism}}$ | Competing multi-objective optimization |
| **Defender Loss** | $\mathcal{L}_{\text{seg}} = 0.5 \cdot \text{BCE} + 0.5 \cdot \mathcal{L}_{\text{Dice/IoU}}$ | Minimizing maximizes segmentation accuracy |
