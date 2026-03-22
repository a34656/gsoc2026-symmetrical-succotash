"""
Data Loading for CMS Jet Dataset — HDF5 Format
================================================
Handles both the unlabelled dataset (Phase 1 autoencoder training)
and the labelled dataset (Phase 2 fine-tuning).

Expected file format (CERN G25 index):
  Files are .h5 (HDF5), not .npz.

  Common HDF5 key structures (auto-detected):
    Option A : f["X"]              shape (N, 125, 125, 8)   — images only
    Option B : f["X"], f["y"]      X=(N,125,125,8), y=(N,)  — labelled
    Option C : f["data"]           shape (N, 125, 125, 8)
    Option D : f["jetImage"]       shape (N, 125, 125, 8)
    Option E : f["X"], f["label"]  alternative label key

  If your file has a different structure, run the inspector first:
    python data.py --inspect your_file.h5

Usage:
    from data import get_unlabelled_loader, get_labelled_loaders
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import spconv.pytorch as spconv

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required: pip install h5py"
    )


# ─────────────────────────────────────────────
#  HDF5 Inspector (run this first on your file)
# ─────────────────────────────────────────────

def inspect_h5(path: str):
    """
    Print the full structure of an HDF5 file.
    Run this to find the correct key names for your dataset.

    Usage:
        python data.py --inspect /path/to/your_file.h5
    """
    print(f"\n{'='*55}")
    print(f"HDF5 File Inspector: {path}")
    print(f"{'='*55}")

    with h5py.File(path, "r") as f:
        def _show(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[Dataset] /{name}")
                print(f"{indent}          shape : {obj.shape}")
                print(f"{indent}          dtype : {obj.dtype}")
                if obj.size > 0:
                    # obj.flat does not exist in h5py — read a flat slice instead
                    flat_data = obj[()].flatten()
                    sample = flat_data[:min(5, len(flat_data))].tolist()
                    print(f"{indent}          sample: {sample}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}[Group]   /{name}")

        f.visititems(_show)
        print(f"\nTop-level keys: {list(f.keys())}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
#  Auto-detect the image and label keys
# ─────────────────────────────────────────────

# Priority-ordered candidate keys for images
_IMAGE_KEY_CANDIDATES = [
    "jet", "X", "data", "jetImage", "images", "jet_images",
    "x", "input", "features", "CaloJet"
]

# Priority-ordered candidate keys for labels
_LABEL_KEY_CANDIDATES = [
    "y", "label", "labels", "target", "targets",
    "Y", "pid", "class", "truth"
]


def _find_key(h5file, candidates):
    """Return the first candidate key that exists in the HDF5 file."""
    for key in candidates:
        if key in h5file:
            return key
    # Fallback: return first dataset key found
    for key in h5file.keys():
        if isinstance(h5file[key], h5py.Dataset):
            return key
    return None


def load_h5(path: str, require_labels: bool = False):
    """
    Load image array (and optionally labels) from an HDF5 file.

    Returns:
        X : np.ndarray shape (N, 125, 125, 8)  float32
        y : np.ndarray shape (N,)               float32  OR  None
    """
    with h5py.File(path, "r") as f:
        # ── Find image key ──
        img_key = _find_key(f, _IMAGE_KEY_CANDIDATES)
        if img_key is None:
            raise KeyError(
                f"Cannot find image data in {path}.\n"
                f"Available keys: {list(f.keys())}\n"
                f"Run: python data.py --inspect {path}"
            )

        X = f[img_key][:]   # load everything into RAM

        # ── Ensure shape is (N, H, W, C) ──
        # Some files store as (N, C, H, W) — detect and fix
        if X.ndim == 4:
            n, a, b, c = X.shape
            if a < b and a < c:
                # Likely (N, C, H, W) since C << H, W
                print(f"[data] Detected channel-first format (N,C,H,W) → "
                      f"transposing to (N,H,W,C)")
                X = np.transpose(X, (0, 2, 3, 1))
        elif X.ndim == 3:
            # (N, H*W, C) or (N, H, W) — handle flat or single-channel
            if X.shape[-1] == 8:
                # (N, 125*125, 8) → reshape
                N = X.shape[0]
                X = X.reshape(N, 125, 125, 8)
            else:
                X = X[..., np.newaxis]   # add channel dim

        X = X.astype(np.float32)

        # ── Find label key ──
        y = None
        lbl_key = _find_key(f, _LABEL_KEY_CANDIDATES)
        if lbl_key is not None and lbl_key != img_key:
            y = f[lbl_key][:].astype(np.float32).ravel()
        elif require_labels:
            raise KeyError(
                f"Cannot find labels in {path}.\n"
                f"Available keys: {list(f.keys())}\n"
                f"Run: python data.py --inspect {path}"
            )

    print(f"[data] Loaded '{img_key}' from {os.path.basename(path)}: "
          f"shape={X.shape}, dtype={X.dtype}")
    if y is not None:
        print(f"[data] Labels '{lbl_key}': shape={y.shape}, "
              f"unique={np.unique(y)}")
    return X, y


# ─────────────────────────────────────────────
#  Conversion: dense image → SparseConvTensor
# ─────────────────────────────────────────────

def dense_to_sparse_tensor(images: torch.Tensor,
                            device=None) -> spconv.SparseConvTensor:
    """
    Convert a dense batch [B, H, W, C] to a SparseConvTensor.
    Zero pixels are treated as inactive sites.
    """
    if device is not None:
        images = images.to(device)

    B, H, W, C = images.shape

    # Active mask: at least one channel is non-zero
    mask    = (images.abs().sum(dim=-1) > 0)           # [B, H, W]
    indices = mask.nonzero(as_tuple=False)              # [N_active, 3]

    if indices.shape[0] == 0:
        indices  = torch.zeros((1, 3), dtype=torch.int32, device=images.device)
        features = torch.zeros((1, C), dtype=torch.float32, device=images.device)
    else:
        b_idx    = indices[:, 0]
        h_idx    = indices[:, 1]
        w_idx    = indices[:, 2]
        features = images[b_idx, h_idx, w_idx, :]      # [N_active, C]

    return spconv.SparseConvTensor(
        features=features.float(),
        indices=indices.int(),
        spatial_shape=[H, W],
        batch_size=B
    )


# ─────────────────────────────────────────────
#  Dataset: Unlabelled (Phase 1)
# ─────────────────────────────────────────────

class JetUnlabelledDataset(Dataset):
    """
    Loads the unlabelled CMS jet dataset for autoencoder pre-training.

    Args:
        h5_path    : path to .h5 file
        normalize  : per-channel normalization by max absolute value
        max_samples: optionally cap the dataset size (e.g. for quick tests)
    """

    def __init__(self, h5_path: str, normalize: bool = True,
                 max_samples: int = None):
        X, _ = load_h5(h5_path, require_labels=False)

        if max_samples is not None:
            X = X[:max_samples]

        if normalize:
            for c in range(X.shape[-1]):
                max_val = np.abs(X[..., c]).max()
                if max_val > 0:
                    X[..., c] /= max_val

        self.X = torch.from_numpy(X)   # [N, H, W, C]

        sparsity = (self.X == 0).float().mean().item()
        n_active = int((self.X.abs().sum(-1) > 0).float().mean().item()
                       * X.shape[1] * X.shape[2])
        print(f"[Unlabelled] {len(self.X)} samples  |  "
              f"sparsity {sparsity*100:.1f}%  |  "
              f"~{n_active} active sites/image")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]   # [H, W, C]


# ─────────────────────────────────────────────
#  Dataset: Labelled (Phase 2)
# ─────────────────────────────────────────────

class JetLabelledDataset(Dataset):
    """
    Loads the labelled CMS jet dataset for classifier fine-tuning.

    Args:
        h5_path    : path to .h5 file containing images AND labels
        normalize  : per-channel normalization
        max_samples: optionally cap the dataset size
    """

    def __init__(self, h5_path: str, normalize: bool = True,
                 max_samples: int = None):
        X, y = load_h5(h5_path, require_labels=True)

        if max_samples is not None:
            X = X[:max_samples]
            y = y[:max_samples]

        if normalize:
            for c in range(X.shape[-1]):
                max_val = np.abs(X[..., c]).max()
                if max_val > 0:
                    X[..., c] /= max_val

        self.X = torch.from_numpy(X)                   # [N, H, W, C]
        self.y = torch.from_numpy(y).unsqueeze(1)      # [N, 1]

        pos = int(y.sum()); neg = len(y) - pos
        print(f"[Labelled] {len(self.X)} samples  |  "
              f"class 0: {neg}  class 1: {pos}  "
              f"({pos/len(y)*100:.1f}% positive)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # ([H,W,C], [1])


# ─────────────────────────────────────────────
#  Collate functions
# ─────────────────────────────────────────────

def collate_unlabelled(batch):
    images = torch.stack(batch, dim=0)   # [B, H, W, C]
    return dense_to_sparse_tensor(images)


def collate_labelled(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return dense_to_sparse_tensor(images), labels


# ─────────────────────────────────────────────
#  DataLoader factories
# ─────────────────────────────────────────────

def get_unlabelled_loader(h5_path, batch_size=32, shuffle=True,
                          num_workers=4, max_samples=None):
    dataset = JetUnlabelledDataset(h5_path, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_unlabelled,
        pin_memory=True
    )


def get_labelled_loaders(h5_path, batch_size=32, val_split=0.2,
                         num_workers=4, max_samples=None):
    """Returns (train_loader, val_loader) with 80/20 split."""
    dataset = JetLabelledDataset(h5_path, max_samples=max_samples)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_labelled, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_labelled, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
#  CLI — run this before anything else
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=str, metavar="FILE",
                        help="Print the full structure of an HDF5 file")
    args = parser.parse_args()

    if args.inspect:
        inspect_h5(args.inspect)
    else:
        parser.print_help()