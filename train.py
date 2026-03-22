"""
Training Scripts
=================
Phase 1: Autoencoder pre-training on unlabelled data
Phase 2: Classifier fine-tuning on labelled data

Usage:
    # Phase 1
    python train.py --phase 1 \
        --unlabelled_data /path/to/unlabelled.npz \
        --epochs 50 --batch_size 32 --lr 1e-3

    # Phase 2
    python train.py --phase 2 \
        --labelled_data /path/to/labelled.npz \
        --checkpoint checkpoints/autoencoder_best.pt \
        --epochs 30 --batch_size 32 --lr 5e-4
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import SparseJetAutoencoder, SparseJetClassifier
from data  import get_unlabelled_loader, get_labelled_loaders


# ─────────────────────────────────────────────
#  Loss: Sparse Reconstruction Loss
# ─────────────────────────────────────────────

def sparse_reconstruction_loss(recon_sparse, input_sparse):
    """
    MSE loss computed only at the active input sites.

    Why not a simple MSE on dense tensors?
      The decoder's active sites may not exactly match the input's.
      We compute loss at the intersection — the coordinates that exist
      in both input and output.

    For simplicity (and because active sites rarely shift dramatically
    after a good bottleneck), we compare reconstructed features at
    every active output site against the nearest input feature.

    A cleaner approach used here: convert both to dense and take MSE.
    This is valid because the decoder is forced to reconstruct the
    full 125×125 structure anyway.
    """
    # .dense() → [B, C, H, W]
    recon_dense = recon_sparse.dense()
    input_dense = input_sparse.dense()

    # Compute MSE only at locations active in the INPUT
    # (we don't penalise for predicting zero where input is zero)
    input_mask = (input_dense.abs().sum(dim=1, keepdim=True) > 0).float()  # [B,1,H,W]

    mse = ((recon_dense - input_dense) ** 2 * input_mask).sum()
    n_active = input_mask.sum().clamp(min=1)
    return mse / n_active


# ─────────────────────────────────────────────
#  FLOP counter (approximate)
# ─────────────────────────────────────────────

def estimate_flops(model, n_active_sites=1143, spatial_size=125):
    """
    Rough FLOP estimate for a sparse model.

    For SubMConv2d / SparseConv2d, FLOPs ≈ 2 * kernel^2 * in_ch * out_ch * n_active
    (factor 2 for multiply-add)

    This is an approximation — actual spconv FLOPs depend on the rule book,
    which changes with each batch. We use the average active site count.
    """
    total = 0
    for name, module in model.named_modules():
        import spconv.pytorch as spconv
        if isinstance(module, (spconv.SubMConv2d, spconv.SparseConv2d,
                                spconv.SparseInverseConv2d)):
            k  = module.kernel_size if isinstance(module.kernel_size, int) \
                 else module.kernel_size[0]
            ic = module.in_channels
            oc = module.out_channels
            # SubMConv preserves active sites; SparseConv may change them
            flops = 2 * k * k * ic * oc * n_active_sites
            total += flops
        elif isinstance(module, nn.Linear):
            total += 2 * module.in_features * module.out_features
    return total


# ─────────────────────────────────────────────
#  Phase 1: Autoencoder Training
# ─────────────────────────────────────────────

def train_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Phase 1: Autoencoder Training")
    print(f"Device : {device}")
    print(f"{'='*50}\n")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    # Data
    loader = get_unlabelled_loader(
        args.unlabelled_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = SparseJetAutoencoder(
        in_channels=8,
        latent_dim=args.latent_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimiser + scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {"train_loss": []}
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            # batch is already a SparseConvTensor (from collate_unlabelled)
            batch = batch.to(device) if hasattr(batch, 'to') else batch

            # Move features and indices to device manually
            import spconv.pytorch as spconv
            batch = spconv.SparseConvTensor(
                features=batch.features.to(device),
                indices=batch.indices.to(device),
                spatial_shape=batch.spatial_shape,
                batch_size=batch.batch_size
            )

            optimizer.zero_grad()
            recon = model(batch)
            loss  = sparse_reconstruction_loss(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  Loss: {avg_loss:.6f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "loss":        best_loss,
                "args":        vars(args)
            }, "checkpoints/autoencoder_best.pt")

        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       f"checkpoints/autoencoder_epoch{epoch}.pt")

    # Save loss history
    np.save("logs/autoencoder_loss.npy", np.array(history["train_loss"]))
    print(f"\n✓ Phase 1 complete. Best loss: {best_loss:.6f}")
    print(f"  Checkpoint: checkpoints/autoencoder_best.pt")


# ─────────────────────────────────────────────
#  Phase 2: Classifier Fine-Tuning
# ─────────────────────────────────────────────

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Phase 2: Classifier Fine-Tuning")
    print(f"Device : {device}")
    print(f"{'='*50}\n")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    # Load pretrained autoencoder
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Run Phase 1 first: python train.py --phase 1 ..."
        )

    autoencoder = SparseJetAutoencoder(in_channels=8, latent_dim=args.latent_dim)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    autoencoder.load_state_dict(ckpt["model_state"])
    print(f"Loaded autoencoder from {args.checkpoint} (epoch {ckpt['epoch']})")

    # Build classifier
    model = SparseJetClassifier(
        autoencoder,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Data
    train_loader, val_loader = get_labelled_loaders(
        args.labelled_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Loss & optimiser
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history   = {"train_loss": [], "val_loss": [], "val_acc": [], "val_error": []}
    best_val  = float("inf")

    import spconv.pytorch as spconv

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for x_sparse, y in train_loader:
            x_sparse = spconv.SparseConvTensor(
                features=x_sparse.features.to(device),
                indices=x_sparse.indices.to(device),
                spatial_shape=x_sparse.spatial_shape,
                batch_size=x_sparse.batch_size
            )
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_sparse)
            loss   = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        # ── Validate ──
        model.eval()
        val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for x_sparse, y in val_loader:
                x_sparse = spconv.SparseConvTensor(
                    features=x_sparse.features.to(device),
                    indices=x_sparse.indices.to(device),
                    spatial_shape=x_sparse.spatial_shape,
                    batch_size=x_sparse.batch_size
                )
                y = y.to(device)
                logits = model(x_sparse)
                val_loss += loss_fn(logits, y).item()
                preds     = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.numel()

        scheduler.step()

        avg_train = train_loss / n_batches
        avg_val   = val_loss   / len(val_loader)
        val_acc   = correct / total
        val_error = 1.0 - val_acc

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)
        history["val_error"].append(val_error)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
              f"Acc: {val_acc*100:.2f}%  Error: {val_error*100:.2f}%")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_error":   val_error,
                "args":        vars(args)
            }, "checkpoints/classifier_best.pt")

    np.save("logs/classifier_history.npy", history)
    print(f"\n✓ Phase 2 complete. Best val loss: {best_val:.6f}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CERN Sparse Jet Training")
    parser.add_argument("--phase",           type=int,   required=True,
                        choices=[1, 2],      help="1=autoencoder, 2=classifier")
    parser.add_argument("--unlabelled_data", type=str,   default=None,
                        help="Path to unlabelled .h5 file")
    parser.add_argument("--labelled_data",   type=str,   default=None,
                        help="Path to labelled .h5 file")
    parser.add_argument("--checkpoint",      type=str,
                        default="checkpoints/autoencoder_best.pt")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--latent_dim",      type=int,   default=128)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--freeze_encoder",  action="store_true",
                        help="Freeze encoder weights during Phase 2")
    args = parser.parse_args()

    if args.phase == 1:
        if args.unlabelled_data is None:
            parser.error("--unlabelled_data required for phase 1")
        train_autoencoder(args)
    else:
        if args.labelled_data is None:
            parser.error("--labelled_data required for phase 2")
        train_classifier(args)