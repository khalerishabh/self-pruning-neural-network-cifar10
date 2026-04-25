"""
Self-Pruning Neural Network on CIFAR-10
========================================
A neural network that learns to prune itself during training using
learnable gate parameters and L1 sparsity regularization.

Author: Rishabh Khale
Case Study: Tredence Analytics — AI Engineering Internship 2025
"""

import os
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
RESULTS_DIR = "./results"
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 5                 # epochs over which λ ramps from 0 → target
PRUNE_THRESHOLD = 1e-2            # gates below this are considered pruned
LAMBDA_VALUES = [1e-4, 5e-4, 1e-3, 5e-3]

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Part 1 — PrunableLinear Layer
# ============================================================================

class PrunableLinear(nn.Module):
    """
    Linear layer with learnable per-weight gates for self-pruning.

    Each weight w_ij is paired with a gate score s_ij. During forward pass:
        gate = sigmoid(s_ij)
        pruned_weight = w_ij * gate
    The layer then computes: output = input @ pruned_weight.T + bias

    The L1 penalty on gate values (applied externally) drives unnecessary
    gates toward zero, effectively pruning the corresponding weights.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight.
        # Initialized to +2.0 so sigmoid(2.0) ≈ 0.88 — gates start mostly open,
        # giving the network a fair chance to learn before pruning pressure kicks in.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Fan-in based bias init (same as nn.Linear)
        fan_in = in_features
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (sigmoid of gate_scores), shape (out, in)."""
        return torch.sigmoid(self.gate_scores)

    def get_sparsity(self) -> float:
        """Percentage of gates below PRUNE_THRESHOLD."""
        gates = self.get_gates().detach()
        total = gates.numel()
        pruned = (gates < PRUNE_THRESHOLD).sum().item()
        return 100.0 * pruned / total

    def prune_and_freeze(self) -> None:
        """
        Hard-prune: zero out gates below threshold and detach from grad.
        Useful for inference — removes pruned weights from computation.
        """
        with torch.no_grad():
            mask = self.get_gates() >= PRUNE_THRESHOLD
            # Set gate_scores to large negative value where pruned
            self.gate_scores.data[~mask] = -20.0
            # Zero out corresponding weights
            self.weight.data[~mask] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights."""
        gates = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def __repr__(self) -> str:
        return (
            f"PrunableLinear(in={self.in_features}, out={self.out_features}, "
            f"sparsity={self.get_sparsity():.1f}%)"
        )


# ============================================================================
# Part 2 — Network Architecture
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    Hybrid CNN + PrunableLinear classifier for CIFAR-10.

    Architecture:
        Conv feature extractor (standard, not pruned)
        → AdaptiveAvgPool → Flatten
        → PrunableLinear(2048, 512) → BN → ReLU → Dropout
        → PrunableLinear(512, 128)  → BN → ReLU → Dropout
        → PrunableLinear(128, 10)   → Output
    """

    def __init__(self):
        super().__init__()

        # --- Convolutional feature extractor (standard, not pruned) ---
        self.features = nn.Sequential(
            # Block 1: 3 → 64, spatial: 32 → 16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 → 128, spatial: 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # After features: 128 channels × 8 × 8
        # AdaptiveAvgPool to fixed 4×4 spatial → 128 × 4 × 4 = 2048
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- PrunableLinear classifier head ---
        self.classifier = nn.Sequential(
            PrunableLinear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            PrunableLinear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten to (batch, 2048)
        x = self.classifier(x)
        return x

    def get_prunable_layers(self) -> list:
        """Return all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def get_all_gates(self) -> torch.Tensor:
        """Concatenated gate values from all PrunableLinear layers (1-D)."""
        gates = [layer.get_gates().view(-1) for layer in self.get_prunable_layers()]
        return torch.cat(gates)

    def get_total_sparsity(self) -> float:
        """Overall sparsity % across all PrunableLinear layers."""
        gates = self.get_all_gates().detach()
        total = gates.numel()
        pruned = (gates < PRUNE_THRESHOLD).sum().item()
        return 100.0 * pruned / total

    def get_layerwise_sparsity(self) -> dict:
        """Sparsity % per PrunableLinear layer, keyed by layer description."""
        result = {}
        for layer in self.get_prunable_layers():
            key = f"PrunableLinear({layer.in_features}→{layer.out_features})"
            result[key] = layer.get_sparsity()
        return result


# ============================================================================
# Part 3 — Sparsity Loss
# ============================================================================

def compute_sparsity_loss(model: SelfPruningNetwork) -> torch.Tensor:
    """
    Compute L1 norm of all gate values across PrunableLinear layers.

    Since gates are positive (post-sigmoid), L1 = sum of all gate values.
    Minimizing this sum pushes gates toward zero → pruning.
    """
    all_gates = model.get_all_gates()
    return all_gates.sum()


def get_scheduled_lambda(epoch: int, target_lambda: float) -> float:
    """
    Linearly ramp λ from 0 → target_lambda over WARMUP_EPOCHS.

    Allows the network to learn useful features before pruning pressure
    kicks in, leading to better final accuracy at the same sparsity level.
    """
    if epoch >= WARMUP_EPOCHS:
        return target_lambda
    return target_lambda * (epoch + 1) / WARMUP_EPOCHS


# ============================================================================
# Part 4 — Data Loading
# ============================================================================

def get_dataloaders() -> tuple:
    """
    Prepare CIFAR-10 train and test loaders with standard augmentation.

    Train: RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize
    Test:  Normalize only
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


# ============================================================================
# Part 5 — Training Loop
# ============================================================================

def train_one_epoch(
    model: SelfPruningNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    target_lambda: float,
) -> dict:
    """
    Train for one epoch with classification + scheduled sparsity loss.

    Returns dict with: train_loss, cls_loss, sparsity_loss, sparsity_pct
    """
    model.train()
    current_lambda = get_scheduled_lambda(epoch, target_lambda)

    total_loss = 0.0
    total_cls = 0.0
    total_sparse = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        cls_loss = F.cross_entropy(outputs, labels)
        sparsity_loss = compute_sparsity_loss(model)
        loss = cls_loss + current_lambda * sparsity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_sparse += sparsity_loss.item()
        n_batches += 1

    return {
        "train_loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "sparsity_loss": total_sparse / n_batches,
        "sparsity_pct": model.get_total_sparsity(),
        "effective_lambda": current_lambda,
    }


@torch.no_grad()
def evaluate(model: SelfPruningNetwork, loader: DataLoader) -> float:
    """Evaluate model on test set. Returns accuracy as percentage."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def run_experiment(target_lambda: float, train_loader, test_loader) -> dict:
    """
    Full training run for a single λ value.

    Returns dict with: lambda, test_accuracy, sparsity, gate_values,
                        layerwise_sparsity, model, history
    """
    set_seed()  # Reset seed per experiment for fair comparison

    model = SelfPruningNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []

    pbar = tqdm(range(EPOCHS), desc=f"λ={target_lambda:.0e}", ncols=100)
    for epoch in pbar:
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, epoch, target_lambda
        )
        test_acc = evaluate(model, test_loader)
        scheduler.step()

        epoch_data = {
            "epoch": epoch + 1,
            "test_accuracy": test_acc,
            **train_metrics,
        }
        history.append(epoch_data)

        pbar.set_postfix({
            "acc": f"{test_acc:.1f}%",
            "spar": f"{train_metrics['sparsity_pct']:.1f}%",
            "loss": f"{train_metrics['train_loss']:.4f}",
        })

    # Final evaluation
    final_acc = evaluate(model, test_loader)
    final_sparsity = model.get_total_sparsity()
    gate_values = model.get_all_gates().detach().cpu().numpy()
    layerwise = model.get_layerwise_sparsity()

    print(f"\n  → Final Accuracy: {final_acc:.2f}%  |  Sparsity: {final_sparsity:.2f}%")
    print(f"  → Layer-wise sparsity: {layerwise}")

    return {
        "lambda": target_lambda,
        "test_accuracy": final_acc,
        "sparsity": final_sparsity,
        "gate_values": gate_values,
        "layerwise_sparsity": layerwise,
        "model": model,
        "history": history,
    }


# ============================================================================
# Part 6 — Visualization
# ============================================================================

def plot_gate_distribution(gate_values: np.ndarray, lam: float) -> None:
    """
    Histogram of final gate values for a model.

    A successful result shows a bimodal distribution: large spike at 0
    (pruned connections) and a cluster near 1 (retained connections).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(gate_values, bins=100, edgecolor="black", alpha=0.75, color="#2196F3")

    # Mark the pruning threshold
    ax.axvline(x=PRUNE_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Prune threshold ({PRUNE_THRESHOLD})")

    pruned_pct = 100.0 * (gate_values < PRUNE_THRESHOLD).sum() / len(gate_values)

    ax.set_xlabel("Gate Value", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Gate Value Distribution  (λ = {lam:.0e},  Sparsity = {pruned_pct:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "gate_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sparsity_vs_accuracy(results: list) -> None:
    """Dual-axis plot: sparsity % and accuracy % vs λ."""
    lambdas = [r["lambda"] for r in results]
    accuracies = [r["test_accuracy"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = "#2196F3"
    color_spar = "#FF5722"

    ax1.set_xlabel("Lambda (λ)", fontsize=13)
    ax1.set_xscale("log")

    # Accuracy line
    ax1.set_ylabel("Test Accuracy (%)", fontsize=13, color=color_acc)
    line1 = ax1.plot(lambdas, accuracies, "o-", color=color_acc, linewidth=2,
                     markersize=8, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)

    # Sparsity line on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity (%)", fontsize=13, color=color_spar)
    line2 = ax2.plot(lambdas, sparsities, "s--", color=color_spar, linewidth=2,
                     markersize=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=color_spar)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11, loc="center left")

    ax1.set_title("Sparsity vs. Accuracy Tradeoff", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "sparsity_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_layerwise_sparsity(results: list) -> None:
    """Grouped bar chart of sparsity per PrunableLinear layer across λ values."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect layer names from first result
    layer_names = list(results[0]["layerwise_sparsity"].keys())
    n_layers = len(layer_names)
    n_lambdas = len(results)
    bar_width = 0.8 / n_lambdas
    x = np.arange(n_layers)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, r in enumerate(results):
        sparsities = [r["layerwise_sparsity"][name] for name in layer_names]
        offset = (i - n_lambdas / 2 + 0.5) * bar_width
        ax.bar(x + offset, sparsities, bar_width, label=f"λ={r['lambda']:.0e}",
               color=colors[i % len(colors)], edgecolor="black", alpha=0.85)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Sparsity (%)", fontsize=13)
    ax.set_title("Layer-wise Sparsity Across λ Values", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "layerwise_sparsity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(results: list) -> None:
    """Plot accuracy and sparsity curves over epochs for all λ values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, r in enumerate(results):
        epochs = [h["epoch"] for h in r["history"]]
        accs = [h["test_accuracy"] for h in r["history"]]
        spars = [h["sparsity_pct"] for h in r["history"]]
        color = colors[i % len(colors)]
        label = f"λ={r['lambda']:.0e}"

        ax1.plot(epochs, accs, color=color, linewidth=1.5, label=label)
        ax2.plot(epochs, spars, color=color, linewidth=1.5, label=label)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy Over Training", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Sparsity (%)", fontsize=12)
    ax2.set_title("Sparsity Over Training", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_results_csv(results: list) -> None:
    """Save experiment results to CSV."""
    path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Test_Accuracy_%", "Sparsity_%"])
        for r in results:
            writer.writerow([
                r["lambda"],
                f"{r['test_accuracy']:.2f}",
                f"{r['sparsity']:.2f}",
            ])
    print(f"  Saved: {path}")


def print_summary_table(results: list) -> None:
    """Print formatted results table to console."""
    table_data = []
    for r in results:
        table_data.append([
            f"{r['lambda']:.0e}",
            f"{r['test_accuracy']:.2f}%",
            f"{r['sparsity']:.2f}%",
        ])

    print(tabulate(
        table_data,
        headers=["Lambda (λ)", "Test Accuracy", "Sparsity Level"],
        tablefmt="grid",
        stralign="center",
    ))


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed()
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"Warmup Epochs: {WARMUP_EPOCHS} | Prune Threshold: {PRUNE_THRESHOLD}")
    print(f"Lambda values to test: {LAMBDA_VALUES}")
    print("=" * 60)

    # Load data once — shared across all experiments
    print("\nDownloading / loading CIFAR-10...")
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    all_results = []

    for lam in LAMBDA_VALUES:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: λ = {lam}")
        print(f"{'='*60}")
        result = run_experiment(lam, train_loader, test_loader)
        all_results.append(result)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print_summary_table(all_results)

    # ---- Generate All Plots ----
    print("\nGenerating plots...")

    # 1. Gate distribution for best model (highest sparsity with reasonable acc)
    viable = [r for r in all_results if r["test_accuracy"] > 60]
    if not viable:
        viable = all_results  # fallback if all models collapsed
    best = max(viable, key=lambda r: r["sparsity"])
    plot_gate_distribution(best["gate_values"], best["lambda"])

    # 2. Sparsity vs Accuracy tradeoff
    plot_sparsity_vs_accuracy(all_results)

    # 3. Layer-wise sparsity comparison
    plot_layerwise_sparsity(all_results)

    # 4. Training curves over epochs (bonus visualization)
    plot_training_curves(all_results)

    # 5. Save CSV
    save_results_csv(all_results)

    print(f"\n{'='*60}")
    print(f"  All results and plots saved to '{RESULTS_DIR}/'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()