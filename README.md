# Self-Pruning Neural Network on CIFAR-10

A PyTorch implementation of a neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularization.

> **Case Study Submission** — Tredence Analytics, AI Engineering Internship 2025

---

## Approach

Instead of pruning a trained network post-hoc, this implementation embeds pruning directly into the training loop:

1. **Gated Weights:** Each weight in the network is paired with a learnable *gate score*. A sigmoid maps these scores to `[0, 1]`, and the gate multiplies its corresponding weight — effectively learning a soft mask over the entire network.

2. **L1 Sparsity Loss:** An L1 penalty on the gate values is added to the classification loss. Since L1 regularization drives values to *exactly* zero (unlike L2, which only drives them *near* zero), the network learns to shut off its own unnecessary connections.

3. **Sparsity Warmup:** λ is ramped linearly from 0 to its target value over the first few epochs, allowing the network to learn useful features before pruning pressure kicks in.

### Architecture

```
Input (3×32×32)
  → Conv Feature Extractor (Conv2d layers with BatchNorm + ReLU + MaxPool)
  → Flatten
  → PrunableLinear(2048, 512) → BatchNorm → ReLU → Dropout
  → PrunableLinear(512, 128)  → BatchNorm → ReLU → Dropout
  → PrunableLinear(128, 10)   → Output
```

Only the fully-connected classifier head uses `PrunableLinear` layers — the convolutional feature extractor remains standard, reflecting realistic pruning practice.

---

## Results

| λ (Lambda) | Test Accuracy (%) | Sparsity Level (%) | Notes                    |
|:----------:|:-----------------:|:-------------------:|:-------------------------|
| `1e-4`     | 90.34             | 9.76                | Low pruning pressure     |
| `5e-4`     | 90.52             | 45.97               | Moderate pruning         |
| `1e-3`     | **90.68**         | 69.76               | Best accuracy-sparsity balance |
| `5e-3`     | 90.53             | **99.58**           | Near-total pruning       |

> **Highlight:** Accuracy remains stable at ~90.3–90.7% even as sparsity increases from 10% to 99.6%, demonstrating that the self-pruning mechanism successfully identifies and removes redundant connections without degrading performance.

### Gate Value Distribution (Best Model)

![Gate Distribution](results/gate_distribution.png)

### Sparsity vs. Accuracy Tradeoff

![Tradeoff](results/sparsity_vs_accuracy.png)

### Layer-wise Sparsity Breakdown

![Layer Sparsity](results/layerwise_sparsity.png)

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
# Run all experiments (trains across multiple λ values)
python self_pruning_network.py
```

Training automatically downloads CIFAR-10 to a `data/` directory on first run.

### Output

- Console: per-epoch metrics (loss, accuracy, sparsity) and a final summary table
- `results/`: saved plots and metrics CSV

---

## Repository Structure

```
.
├── self_pruning_network.py    # Complete implementation (single script)
├── requirements.txt           # Pinned dependencies
├── results/                   # Generated plots and metrics
│   ├── gate_distribution.png
│   ├── sparsity_vs_accuracy.png
│   ├── layerwise_sparsity.png
│   ├── training_curves.png
│   └── experiment_results.csv
├── REPORT.md                  # Detailed analysis and findings
└── README.md                  # This file
```

---

## Author

**Rishabh Khale**
M.Tech CSE (AI & ML) — VIT Vellore
[LinkedIn](https://linkedin.com/in/rishabhkhale1998) · [GitHub](https://github.com/khalerishabh)

---

## License

This project is submitted as part of an internship case study evaluation. All code is original.