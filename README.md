# MNIST Image Classification: MLP vs CNN
### A Methodological Comparison — SWE012 Deep Learning with Python | Team FCB

---

## Overview

This project is a **methodological deep dive** into two neural network architectures applied to the MNIST handwritten digit classification task. Rather than simply achieving high accuracy, the goal is to understand *why* one architecture outperforms the other — and to demonstrate that architectural design choices matter more than parameter count.

We train, compare, and analyze four models:
- **MLP Baseline** — fully connected network, no spatial awareness
- **MLP + Dropout** — regularized version to address overfitting
- **CNN Baseline** — convolutional network with spatial inductive bias
- **CNN + BatchNorm + Dropout** — regularized and stabilized CNN

---

## Key Findings

| Model | Test Accuracy | Test Loss | Overfit Gap |
|---|---|---|---|
| MLP (baseline) | 98.15% | 0.0908 | 1.74% ⚠ |
| MLP + Dropout | 98.22% | 0.0614 | 0.64% ✓ |
| CNN (baseline) | 99.18% | 0.0306 | 0.80% ✓ |
| CNN + BN + Dropout | 99.16% | 0.0352 | 0.51% ✓ |

**Main finding:** CNN outperforms MLP not because it has more parameters (421K vs 235K), but because its architecture matches the spatial structure of image data. This is called **inductive bias** — building the right assumption directly into the model design.

---

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (0–9), 28×28 pixels.

| Split | Size |
|---|---|
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |

Preprocessing: pixel values normalized from [0, 255] to [0.0, 1.0]. No augmentation applied — both models trained under identical conditions for a fair comparison.

---

## Architecture Design Rationale

### MLP

```
Input (28×28) → Flatten (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)
```

| Decision | Reason |
|---|---|
| Dense(256) | ~1/3 of input size (784) — standard first-layer sizing |
| Dense(128) | Funnel architecture — gradual abstraction |
| ReLU | No vanishing gradient; computationally efficient |
| No spatial encoding | MLP limitation — pixels treated as independent features |

**Limitation:** MLP has no notion of which pixels are neighbours. It must re-learn spatial relationships from scratch for every position.

### CNN

```
Input (28×28×1) → Conv2D(32, 3×3) → MaxPool(2×2) → Conv2D(64, 3×3) → MaxPool(2×2) → Dense(128) → Dense(10, Softmax)
```

| Decision | Reason |
|---|---|
| 3×3 kernel | Standard for image tasks — balances local connectivity with low parameter count |
| padding='same' | Preserves 28×28 spatial dimensions through first conv block |
| MaxPool(2×2) | Halves spatial dims; provides translation invariance |
| 32 → 64 filters | First block learns edges/corners; second learns higher-level features |
| Weight sharing | Same filter applied across all positions — learns a pattern once, reuses everywhere |

---

## Training Setup

All models trained under identical conditions to ensure fair comparison:

```
Optimizer : Adam (lr=0.001)
Loss      : sparse_categorical_crossentropy
Epochs    : 15
Batch size: 128
Seed      : 42
```

---

## Overfitting Analysis

Accuracy alone is misleading. The train-validation gap reveals whether a model is *learning* or *memorising*.

```
MLP baseline → train: 99.68%  val: 97.94%  gap: 1.74% ← overfitting
CNN baseline → train: 99.86%  val: 99.06%  gap: 0.80% ← healthy
```

CNN's smaller gap is a direct result of **weight sharing** — the model's effective parameter count is lower than its raw count suggests, providing built-in regularization.

---

## Regularization

### Dropout (rate=0.3)
Applied after Dense layers only. Randomly disables 30% of neurons each forward pass — the model cannot rely on any single neuron, forcing robust feature learning. Not applied before the Softmax output layer.

### BatchNormalization
Applied after each Conv layer (`Conv → BN → MaxPool` ordering). Normalizes mini-batch activations to zero mean and unit variance, reducing internal covariate shift and stabilizing gradient flow.

---

## Optimization Topics

### Optimizer Comparison

| Optimizer | Test Accuracy | Val Acc (Epoch 1) |
|---|---|---|
| SGD (lr=0.01) | 97.60% | 88.50% |
| SGD + Momentum (α=0.9) | 98.90% | 96.51% |
| RMSprop (lr=0.001) | 99.20% | 98.03% |
| **Adam (lr=0.001)** | **99.25%** | **98.18%** |

Adam was chosen for all experiments because it combines momentum (1st moment) with RMSprop-style adaptive rates (2nd moment), plus bias correction — giving the fastest and most reliable convergence with minimal tuning.

### Weight Initialization

| Initializer | First-layer std | Note |
|---|---|---|
| Zeros | 0.000 | Symmetry failure — network cannot learn |
| Xavier (Glorot) | 0.239 | Designed for tanh/sigmoid |
| He (Kaiming) | 0.275 | Correct for ReLU — compensates for signal loss |

He initialization is theoretically optimal for ReLU-based models. Keras uses Xavier by default; for ReLU networks He is strictly more appropriate.

### Learning Rate Selection

| Learning Rate | Val Accuracy | Behaviour |
|---|---|---|
| 0.00001 | 95.62% | Too slow — barely converges in 15 epochs |
| **0.001** | **99.12%** | **Optimal — Adam's default from Kingma & Ba (2014)** |
| 0.1 | 10.64% | Diverged — optimizer overshoots minima |

### Gradient Clipping

CNN gradient norms monitored over 5 epochs:

```
Mean:    0.605
Std:     0.249
Max:     2.353
95th %:  0.920
```

Gradients are well-behaved. `clipnorm=1.0` added to Adam as a defensive best practice — essential for RNNs and deeper architectures where gradient explosions are common.

### Learning Rate Scheduling

`ReduceLROnPlateau` applied to the best model:

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # halve lr on trigger
    patience=3,       # wait 3 epochs before reducing
    min_lr=1e-6
)
```

After each reduction, validation loss dips — the model takes smaller, more precise steps toward the optimum. No manual epoch tuning required; the scheduler responds directly to training dynamics.

---

## Error Analysis

### Most Confused Digit Pairs (MLP)

| Pair | Errors | Reason |
|---|---|---|
| 7 → 9 | 12 | Similar diagonal stroke |
| 4 → 9 | 11 | Closed lower loop similarity |
| 3 → 9 | 9 | Similar lower curve |
| 7 → 2 | 9 | Diagonal stroke confusion |
| 9 → 4 | 8 | Bidirectional confusion |

These confusions are structurally motivated — digits that share local shape features are harder to separate without spatial feature learning.

### Per-Class Accuracy

CNN outperforms MLP on 9 of 10 digit classes. The exception is **digit 9**, where CNN performs slightly worse (97.62% vs 98.41%). This reveals that single accuracy numbers can mislead — some errors are a property of the data itself, not a model failure. A handwritten 9 can genuinely resemble a 4.

---

## Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook (baseline + extensions)
├── README.md               # This file
```

---

## How to Run

**Kaggle (recommended):**
1. Upload `notebook.ipynb` to a Kaggle notebook
2. Enable GPU accelerator (optional but faster)
3. Run All cells in order

**Local:**
```bash
pip install tensorflow numpy matplotlib scikit-learn
jupyter notebook notebook.ipynb
```

---

## Requirements

```
tensorflow >= 2.x
numpy
matplotlib
scikit-learn
```

---
Kaggle link: https://www.kaggle.com/code/ipekercan/deep-learning?scriptVersionId=310203833
---

## References

- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.
- He, K. et al. (2015). Delving deep into rectifiers. *ICCV*.
- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press. Chapter 8.
- Srivastava, N. et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*.
