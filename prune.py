"""
Phase 3: Pruning
=================
Systematically prunes the classifier model at increasing sparsity levels
and measures the trade-off between FLOPs and error rate.

Usage:
    python prune.py \
        --checkpoint checkpoints/classifier_best.pt \
        --labelled_data /path/to/labelled.npz \
        --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

This will:
  1. Load the best classifier checkpoint
  2. For each pruning ratio, apply L1-unstructured pruning to all Linear
     and Conv layers
  3. Evaluate on the validation set
  4. Record (FLOPs, error_rate) for each ratio
  5. Save results to logs/pruning_results.npy
"""

import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import spconv.pytorch as spconv

from model import SparseJetAutoencoder, SparseJetClassifier
from data  import get_labelled_loaders
from train import estimate_flops


# ─────────────────────────────────────────────
#  Apply pruning to a model
# ─────────────────────────────────────────────

def apply_pruning(model, pruning_ratio: float):
    """
    Apply L1 unstructured pruning to all Linear layers and
    the weight matrices of all spconv layers.

    L1 unstructured pruning: zeroes out the weights with the
    smallest absolute values. The 'amount' parameter is the
    fraction of weights to zero out.

    Args:
        model         : SparseJetClassifier
        pruning_ratio : float in [0, 1), fraction of weights to prune
    """
    if pruning_ratio == 0.0:
        return model  # no-op

    # Collect all pruneable layers
    layers_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers_to_prune.append((module, "weight"))
        elif isinstance(module, (spconv.SubMConv2d,
                                  spconv.SparseConv2d,
                                  spconv.SparseInverseConv2d)):
            layers_to_prune.append((module, "weight"))

    # Apply global L1 unstructured pruning
    prune.global_unstructured(
        layers_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )

    # Make pruning permanent (remove the mask buffers, zero weights stay zero)
    for module, param_name in layers_to_prune:
        prune.remove(module, param_name)

    return model


# ─────────────────────────────────────────────
#  Count actual non-zero parameters (after pruning)
# ─────────────────────────────────────────────

def count_nonzero_params(model):
    total    = 0
    nonzero  = 0
    for p in model.parameters():
        total   += p.numel()
        nonzero += p.nonzero().shape[0]
    return nonzero, total


# ─────────────────────────────────────────────
#  Evaluate model
# ─────────────────────────────────────────────

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x_sparse, y in val_loader:
            x_sparse = spconv.SparseConvTensor(
                features=x_sparse.features.to(device),
                indices=x_sparse.indices.to(device),
                spatial_shape=x_sparse.spatial_shape,
                batch_size=x_sparse.batch_size
            )
            y      = y.to(device)
            logits = model(x_sparse)
            preds  = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.numel()
    error_rate = 1.0 - (correct / total)
    return error_rate


# ─────────────────────────────────────────────
#  Main pruning sweep
# ─────────────────────────────────────────────

def run_pruning_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Phase 3: Pruning Sweep")
    print(f"Device  : {device}")
    print(f"Ratios  : {args.ratios}")
    print(f"{'='*50}\n")

    os.makedirs("logs", exist_ok=True)

    # ── Load base model ──
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    autoencoder = SparseJetAutoencoder(in_channels=8, latent_dim=args.latent_dim)

    # Reconstruct classifier from checkpoint
    base_model = SparseJetClassifier(autoencoder, freeze_encoder=False)
    base_model.load_state_dict(ckpt["model_state"])
    print(f"Loaded classifier checkpoint: {args.checkpoint}")

    # ── Data ──
    _, val_loader = get_labelled_loaders(
        args.labelled_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ── Baseline FLOPs (unpruned) ──
    base_flops = estimate_flops(base_model)
    print(f"Baseline FLOPs: {base_flops:,.0f}")

    results = []  # list of (flops, error_rate, ratio, n_nonzero)

    for ratio in args.ratios:
        # Deep copy so we always start from the same pretrained weights
        model = copy.deepcopy(base_model).to(device)

        # Apply pruning
        model = apply_pruning(model, ratio)

        # Count surviving weights
        nonzero, total_params = count_nonzero_params(model)
        actual_sparsity = 1.0 - nonzero / total_params

        # Effective FLOPs after pruning
        # (pruned weights are zero, so multiply-adds with them cost ~0)
        effective_flops = base_flops * (nonzero / total_params)

        # Evaluate
        error_rate = evaluate(model, val_loader, device)

        results.append({
            "ratio":           ratio,
            "effective_flops": effective_flops,
            "error_rate":      error_rate,
            "actual_sparsity": actual_sparsity,
            "nonzero_params":  nonzero,
            "total_params":    total_params
        })

        print(f"Ratio: {ratio:.1f}  |  "
              f"Sparsity: {actual_sparsity*100:.1f}%  |  "
              f"FLOPs: {effective_flops:.2e}  |  "
              f"Error: {error_rate*100:.2f}%")

    # Save
    np.save("logs/pruning_results.npy", results)
    print(f"\n✓ Pruning sweep complete. Results saved to logs/pruning_results.npy")

    return results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str,   required=True)
    parser.add_argument("--labelled_data", type=str,   required=True)
    parser.add_argument("--ratios",        type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--latent_dim",    type=int,   default=128)
    parser.add_argument("--num_workers",   type=int,   default=4)
    args = parser.parse_args()

    run_pruning_sweep(args)