"""
Sparse Autoencoder for CERN CMS Jet Data
=========================================
Architecture based on:
  Graham & van der Maaten (2017) — Submanifold Sparse Convolutional Networks
  arXiv:1706.01307

Convolution types used:
  SubMConv2d  = VSC (Valid Sparse Conv) — preserves sparsity, stride=1 only
  SparseConv2d = SC  (Sparse Conv)      — dilates/downsamples, used with stride=2
  SparseInverseConv2d                   — inverse of SparseConv2d, needs indice_key

Data:
  Input shape : (batch, 125, 125, 8)  — 8-channel unrolled CMS detector grid
  Active sites: ~1143 / 15625 pixels  — ~7% occupancy
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv


# ─────────────────────────────────────────────
#  Helper: apply BatchNorm safely on sparse tensors
# ─────────────────────────────────────────────

class SparseBatchNorm(nn.Module):
    """
    BatchNorm for sparse tensors.
    spconv sparse tensors store data as .features (shape: [N_active, C]).
    nn.BatchNorm1d works on [N, C], so we apply it to .features directly.
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # x is a SparseConvTensor; x.features has shape [N_active, C]
        x = x.replace_feature(self.bn(x.features))
        return x


# ─────────────────────────────────────────────
#  Encoder Block (reusable)
# ─────────────────────────────────────────────

class SparseEncoderBlock(nn.Module):
    """
    One encoder stage:
      SubMConv2d (stride=1, preserves sparsity) → BN → ReLU
      SparseConv2d (stride=2, downsamples)      → BN → ReLU
    """
    def __init__(self, in_ch, out_ch, indice_key):
        super().__init__()
        self.vsc = spconv.SubMConv2d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = SparseBatchNorm(out_ch)
        self.relu1 = nn.ReLU()

        self.sc = spconv.SparseConv2d(
            out_ch, out_ch, kernel_size=3, stride=2, padding=1,
            bias=False, indice_key=indice_key          # ← must match decoder
        )
        self.bn2 = SparseBatchNorm(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.vsc(x)
        x = x.replace_feature(self.relu1(self.bn1(x).features))
        x = self.sc(x)
        x = x.replace_feature(self.relu2(self.bn2(x).features))
        return x


# ─────────────────────────────────────────────
#  Decoder Block (reusable)
# ─────────────────────────────────────────────

class SparseDecoderBlock(nn.Module):
    """
    One decoder stage:
      SparseInverseConv2d (upsamples, reverses the stride=2 SC) → BN → ReLU
      SubMConv2d (stride=1, refines features)                   → BN → ReLU
    """
    def __init__(self, in_ch, out_ch, indice_key):
        super().__init__()
        # indice_key MUST match the corresponding SparseConv2d in the encoder
        self.inv = spconv.SparseInverseConv2d(
            in_ch, out_ch, kernel_size=3,
            bias=False, indice_key=indice_key
        )
        self.bn1 = SparseBatchNorm(out_ch)
        self.relu1 = nn.ReLU()

        self.vsc = spconv.SubMConv2d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = SparseBatchNorm(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.inv(x)
        x = x.replace_feature(self.relu1(self.bn1(x).features))
        x = self.vsc(x)
        x = x.replace_feature(self.relu2(self.bn2(x).features))
        return x


# ─────────────────────────────────────────────
#  Full Sparse Autoencoder
# ─────────────────────────────────────────────

class SparseJetAutoencoder(nn.Module):
    """
    Sparse Autoencoder for CMS jet images.

    Encoder:
        125×125×8  →  63×63×32  →  32×32×64  →  flatten  →  latent(128)

    Decoder:
        latent(128)  →  reshape  →  32×32×64  →  63×63×32  →  125×125×8

    The encoder (without the final linear layers) is detached for
    Phase 2 fine-tuning on labelled data.

    Args:
        in_channels  : number of input channels (8 for CMS data)
        latent_dim   : size of the bottleneck latent vector (default 128)
        spatial_size : input grid size (default 125 for 125×125)
    """

    def __init__(self, in_channels=8, latent_dim=128, spatial_size=125):
        super().__init__()

        self.in_channels  = in_channels
        self.latent_dim   = latent_dim
        self.spatial_size = spatial_size

        # ── Encoder sparse blocks ──────────────────────────────────────────
        # Stage 1: 125×125 → 63×63,  8 ch → 32 ch
        self.enc1 = SparseEncoderBlock(in_ch=in_channels, out_ch=32,
                                       indice_key="down1")
        # Stage 2: 63×63  → 32×32,  32 ch → 64 ch
        self.enc2 = SparseEncoderBlock(in_ch=32, out_ch=64,
                                       indice_key="down2")

        # After enc2 the spatial size is ceil(125/4) = 32  (125→63→32)
        self._bottleneck_spatial = 32
        self._bottleneck_ch      = 64
        flat_dim = self._bottleneck_ch * self._bottleneck_spatial ** 2  # 64*32*32

        # ── Dense bottleneck ───────────────────────────────────────────────
        self.fc_encode = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU()
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU()
        )

        # ── Decoder sparse blocks ──────────────────────────────────────────
        # Stage 1: 32×32 → 63×63,  64 ch → 32 ch  (reverses down2)
        self.dec1 = SparseDecoderBlock(in_ch=64, out_ch=32,
                                       indice_key="down2")
        # Stage 2: 63×63 → 125×125, 32 ch → in_channels  (reverses down1)
        self.dec2 = SparseDecoderBlock(in_ch=32, out_ch=in_channels,
                                       indice_key="down1")

        # Final SubMConv to reconstruct exactly in_channels
        self.final_conv = spconv.SubMConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=True
        )

    # ── Forward ──────────────────────────────────────────────────────────

    def encode(self, x_sparse):
        """
        Run the sparse encoder, return latent vector.
        x_sparse : spconv SparseConvTensor [batch, 125, 125, in_channels]
        returns  : latent  [batch, latent_dim]
        """
        x = self.enc1(x_sparse)           # → sparse [batch, 63, 63, 32]
        x = self.enc2(x)                  # → sparse [batch, 32, 32, 64]

        # Convert sparse → dense and flatten
        # .dense() returns [batch, C, H, W]
        x_dense = x.dense()               # [B, 64, 32, 32]
        B = x_dense.shape[0]
        x_flat  = x_dense.view(B, -1)     # [B, 64*32*32]

        latent  = self.fc_encode(x_flat)  # [B, latent_dim]
        return latent

    def decode(self, latent, batch_size, device):
        """
        Run the sparse decoder, return reconstructed sparse tensor.
        latent     : [batch, latent_dim]
        batch_size : int
        device     : torch.device
        returns    : SparseConvTensor [batch, 125, 125, in_channels]
        """
        flat_dim = self._bottleneck_ch * self._bottleneck_spatial ** 2

        x = self.fc_decode(latent)                        # [B, flat_dim]
        x = x.view(batch_size,
                   self._bottleneck_ch,
                   self._bottleneck_spatial,
                   self._bottleneck_spatial)              # [B, 64, 32, 32]

        # Convert dense → sparse for the inverse convolutions
        x_sparse = self._dense_to_sparse(x, device)

        x = self.dec1(x_sparse)    # → sparse [batch, 63, 63, 32]
        x = self.dec2(x)           # → sparse [batch, 125, 125, in_channels]
        x = self.final_conv(x)     # refine, keep same shape
        return x

    def forward(self, x_sparse):
        """
        Full autoencoder forward pass.
        Returns reconstructed SparseConvTensor (same spatial shape as input).
        """
        B      = x_sparse.batch_size
        device = x_sparse.features.device
        latent = self.encode(x_sparse)
        recon  = self.decode(latent, B, device)
        return recon

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _dense_to_sparse(x_dense, device):
        """
        Convert a dense tensor [B, C, H, W] back into a SparseConvTensor.
        Only non-zero spatial locations become active sites.
        """
        B, C, H, W = x_dense.shape
        # Find non-zero locations: mask over spatial dims
        mask = (x_dense.abs().sum(dim=1) > 0)          # [B, H, W]
        indices = mask.nonzero(as_tuple=False)          # [N, 3] = (b, h, w)

        if indices.shape[0] == 0:
            # Edge case: all zeros — create a dummy active site
            indices = torch.zeros((1, 3), dtype=torch.int32, device=device)

        # features: gather channel vectors at active locations
        b_idx = indices[:, 0]
        h_idx = indices[:, 1]
        w_idx = indices[:, 2]
        features = x_dense[b_idx, :, h_idx, w_idx]     # [N, C]

        # spconv expects indices as [N, 3] = (batch, x, y) in int32
        sp_indices = indices.int()

        sparse = spconv.SparseConvTensor(
            features=features,
            indices=sp_indices,
            spatial_shape=[H, W],
            batch_size=B
        )
        return sparse


# ─────────────────────────────────────────────
#  Phase 2: Classifier (encoder head)
# ─────────────────────────────────────────────

class SparseJetClassifier(nn.Module):
    """
    Fine-tuning wrapper for Phase 2.
    Takes the pretrained encoder from SparseJetAutoencoder and
    attaches a binary classification head.

    Args:
        autoencoder : pretrained SparseJetAutoencoder instance
        freeze_encoder : if True, encoder weights are frozen (feature extraction)
                         if False, encoder weights are fine-tuned (full fine-tune)
    """

    def __init__(self, autoencoder: SparseJetAutoencoder, freeze_encoder=False):
        super().__init__()

        # Borrow the encoder components from the autoencoder
        self.enc1      = autoencoder.enc1
        self.enc2      = autoencoder.enc2
        self.fc_encode = autoencoder.fc_encode

        if freeze_encoder:
            for p in self.enc1.parameters():   p.requires_grad = False
            for p in self.enc2.parameters():   p.requires_grad = False
            for p in self.fc_encode.parameters(): p.requires_grad = False

        latent_dim = autoencoder.latent_dim

        # Classification head: latent → binary logit
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)         # raw logit; use BCEWithLogitsLoss
        )

    def forward(self, x_sparse):
        """
        x_sparse : SparseConvTensor
        returns  : logits [batch, 1]  (raw, before sigmoid)
        """
        latent = self._encode(x_sparse)
        logits = self.classifier(latent)   # [B, 1]
        return logits

    def _encode(self, x_sparse):
        x = self.enc1(x_sparse)
        x = self.enc2(x)
        x_dense = x.dense()
        B = x_dense.shape[0]
        x_flat = x_dense.view(B, -1)
        latent = self.fc_encode(x_flat)
        return latent


# ─────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Simulate one batch of sparse CMS jet data
    B, H, W, C = 2, 125, 125, 8
    N_active    = 1143          # ~7% occupancy per image

    # Random active coordinates for each batch item
    coords_list = []
    for b in range(B):
        xy = np.random.choice(H * W, N_active, replace=False)
        rows = xy // W
        cols = xy % W
        batch_col = np.full(N_active, b)
        coords_list.append(np.stack([batch_col, rows, cols], axis=1))

    coords   = np.concatenate(coords_list, axis=0)             # [B*N, 3]
    features = np.random.randn(B * N_active, C).astype(np.float32)

    sp_indices = torch.from_numpy(coords).int().to(device)
    sp_feats   = torch.from_numpy(features).to(device)

    x_sparse = spconv.SparseConvTensor(
        features=sp_feats,
        indices=sp_indices,
        spatial_shape=[H, W],
        batch_size=B
    )

    # ── Test Autoencoder ──
    model = SparseJetAutoencoder(in_channels=C, latent_dim=128).to(device)
    print(f"\nAutoencoder parameters: {sum(p.numel() for p in model.parameters()):,}")

    recon = model(x_sparse)
    print(f"Input  active sites : {sp_feats.shape[0]}")
    print(f"Output active sites : {recon.features.shape[0]}")
    print(f"Output feature shape: {recon.features.shape}")

    # ── Test Classifier ──
    clf = SparseJetClassifier(model, freeze_encoder=False).to(device)
    print(f"\nClassifier parameters: {sum(p.numel() for p in clf.parameters()):,}")

    labels  = torch.zeros(B, 1).to(device)
    logits  = clf(x_sparse)
    print(f"Logits shape: {logits.shape}")

    loss_fn = nn.BCEWithLogitsLoss()
    loss    = loss_fn(logits, labels)
    print(f"Classification loss: {loss.item():.4f}")

    print("\n✓ All checks passed.")