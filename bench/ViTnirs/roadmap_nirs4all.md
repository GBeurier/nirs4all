# ViT-NIRS Universal Encoder — nirs4all-Integrated Roadmap

> Adapting the Universal Spectral Embedding project to leverage nirs4all components
> for synthetic data generation, augmentation, preprocessing, and data management.

---

## Overview

This roadmap transforms the generic ViT-NIRS implementation plan into a nirs4all-integrated project. We maximize reuse of existing operators while implementing custom training loops outside the nirs4all pipeline system (which is optimized for sklearn-style workflows, not complex self-supervised pretraining).

### Real Data Location

**X-bank:** `/home/delete/NIRS DB/x_bank/`
- One CSV file per X collection (dataset)
- ~45 real datasets with varying sizes (60–15,000 samples) and wavelength grids (250–2,100 points)

### Integration Strategy

| Layer | nirs4all Usage | Custom Implementation |
|-------|----------------|----------------------|
| **Data Model** | `SpectroDataset` for all data | — |
| **Synthetic Data** | `nirs4all.generate()` + `SyntheticDatasetBuilder` | — |
| **Augmentation** | Existing operators (noise, scatter, baseline drift) | MAE masking strategy |
| **Preprocessing** | Baseline operators (`ASLSBaseline`, `ArPLS`) | Caching workflow |
| **Splitting** | `KennardStoneSplitter`, `SPXYSplitter` | LODO wrapper |
| **Training Loop** | — | PyTorch custom loop |
| **Model Architecture** | — | Spectral Perceiver (ViT) |

---

## 1. Training Strategy: Targets, Sampling & Synthetic Data

### 1.1 Do You Need Targets to Train the Encoder?

**Short answer: No.** For a **generic, transferable embedding**, the safest default is:

1. **Self-supervised pretraining** (MAE + invariance losses) on *all spectra* (labeled and unlabeled)
2. Use targets **only for downstream training** (ridge/PLS/TabPFN/MLP) or light fine-tuning

This avoids the encoder "baking in" dataset/task-specific shortcuts.

#### When Targets Can Help (Optional, Later)

Targets can improve embeddings **if done carefully**:

| Approach | When to Use | Benefit | Risk |
|----------|-------------|---------|------|
| **A) Multi-task supervised fine-tuning** | After self-supervised pretraining, with many labeled tasks | Aligns embedding with predictive directions | Can reduce cross-dataset generality if dominated by few datasets |
| **B) Semi-supervised auxiliary objective** | Keep MAE as main loss, add supervised head with low weight (0.05–0.2) | Slight task alignment without collapsing generality | Minimal if weight is low |
| **C) Supervised-only from scratch** | **Avoid unless** you have very large labeled coverage across instruments | — | With many small datasets (N=60…), tends to overfit and learn dataset ID |

#### Practical Rule

For **universal embedding**, follow this sequence:
1. Self-supervised pretraining (no targets)
2. Evaluate downstream performance
3. Optionally fine-tune with targets on selected tasks

### 1.2 How to Subsample Datasets During Training

You have extreme imbalance (60 → 15,000 samples). **Prevent big datasets from dominating.**

#### Recommended Default: Uniform-by-Dataset Sampling

Each training step:
1. Sample `dataset_id` **uniformly** from the 40–45 real datasets
2. Sample one example **uniformly** inside that dataset
3. Optionally replace with synthetic with probability `p_synth`

This ensures every dataset contributes equally to the representation.

#### Stabilizers (Strongly Recommended)

**A) Cap per-dataset steps per epoch (virtual epochs)**

Define an "epoch" as a fixed number of steps, not "one pass through all data":
```python
steps_per_epoch = K * num_datasets  # e.g., K=200
# With 45 datasets → 9,000 steps/epoch
```
This makes training consistent despite size imbalance.

**B) Length-aware batching (optional, for speed)**

Variable-length spectra cause padding/packing overhead. Use bucketing:
- Group samples into bins by length (e.g., 250–500, 500–900, …)
- Sample batch from one bin at a time

#### Extra Strategies

| Situation | Strategy |
|-----------|----------|
| **Long spectra dominate compute** | Randomly **subsample points** or **crop wavelength windows** during pretraining. Always provide true `λ` values. Teaches encoder to work with partial coverage. |
| **Some datasets are noisy/low quality** | Use per-dataset weights in sampling (start uniform, then downweight pathological sets). Or use curriculum: start with cleaner/larger datasets, then broaden. |
| **LODO evaluation** | Do **not** include the held-out dataset in encoder pretraining for strict transfer test. For general foundation model, pretrain on everything but clarify that LODO evaluates downstream only. |

### 1.3 What Makes Synthetic Spectra Useful?

Synthetic helps when it provides **correct invariances** and **realistic diversity** without creating easy shortcuts.

#### Minimum Requirements

**A) Match the "nuisance factors" seen in reality**

Synthetic must cover:
- Baseline drift shapes (smooth, variable amplitude)
- Scatter (multiplicative + offset), sometimes wavelength-dependent
- Noise profiles (including wavelength-dependent noise)
- Slight wavelength shifts / resolution effects (if realistic)
- Missing bands / partial coverage (optional but valuable)

**B) Match instrument heterogeneity**

At least one of:
- Generate spectra on **multiple wavelength grids** (range/step vary)
- Simulate re-sampling artifacts + different resolutions
- Produce metadata ("instrument", "mode") consistent with generated signals

**C) Avoid synthetic-specific artifacts**

The model must NOT learn "this looks synthetic". Ensure:
- No fixed template baseline
- No constant noise scale across all samples
- No deterministic peak shapes always identical
- No unnatural quantization or clipping patterns

#### Utility Checks (Run Before Spending 200 Epochs)

1. **Linear separability of synthetic vs real**
   - Train a small classifier on embeddings to predict `is_synthetic`
   - If near-perfect early, synthetic is too distinguishable → reduce artifacts / add domain randomization

2. **Downstream lift on small real datasets**
   - Pretrain encoder with and without synthetic
   - Compare LODO / cross-instrument generalization
   - Synthetic is useful only if it improves these, not just MAE loss

3. **Paired-view invariance (highest-value use of synthetic)**
   - Generate *paired observations* of same underlying sample:
     - Same latent composition → two instruments / two baseline/scatter settings
   - Use for contrastive/invariance training

#### What Synthetic Does NOT Need To Be

- Does not need perfect chemical accuracy to be useful for representation learning
- Must be **structurally plausible** and cover **real nuisance variability**

### 1.4 Practical Defaults (Recommended Starting Point)

```yaml
# Training strategy defaults
training_strategy:
  use_targets_in_pretraining: false  # Self-supervised only
  sampling: "uniform_by_dataset"
  steps_per_epoch: 9000              # K=200 × 45 datasets
  p_synth: 0.3                       # 30% synthetic mix

  # Synthetic focus areas
  synthetic_priorities:
    - instrument_grids              # Variable wavelength ranges/steps
    - baseline_variation            # Drift diversity
    - scatter_variation             # MSC-like effects
    - noise_profiles                # Including wavelength-dependent
    - paired_views                  # For contrastive learning
```

---

## 2. Project Structure (nirs4all-Aligned)

```
bench/ViTnirs/
├── data/
│   ├── loaders.py              # SpectroDataset-based loaders
│   ├── collate.py              # Variable-length collation
│   ├── multi_dataset.py        # Multi-dataset wrapper + uniform sampling
│   └── cache/
│       └── asls/               # Precomputed baselines (per dataset)
├── augmentation/
│   ├── __init__.py
│   ├── nirs4all_adapters.py    # Wrappers for nirs4all augmenters
│   ├── mae_masking.py          # Block-based MAE masking (NEW)
│   └── compose.py              # Composition utilities
├── preprocessing/
│   ├── asls_cache.py           # AsLS precomputation script
│   └── normalizers.py          # Per-spectrum normalization
├── models/
│   ├── encoder/
│   │   ├── spectral_perceiver.py
│   │   ├── positional_encoding.py
│   │   └── metadata_embedding.py
│   ├── decoder/
│   │   └── mae_decoder.py
│   └── heads/
│       ├── regression.py
│       └── classification.py
├── losses/
│   ├── mae.py
│   ├── baseline_consistency.py
│   └── contrastive.py          # Optional
├── training/
│   ├── pretrain.py
│   ├── finetune.py
│   └── configs/
├── evaluation/
│   ├── downstream.py
│   ├── lodo.py
│   └── ablations.py
├── scripts/
│   ├── precompute_asls.py
│   ├── run_pretraining.py
│   └── run_evaluation.py
├── onboarding.md               # Original spec (reference)
├── roadmap_nirs4all.md         # This file
└── configs/
    ├── default.yaml
    └── ablation_*.yaml
```

---

## 3. Data Layer: SpectroDataset Integration

### 3.1 Real Data Loading

**Location:** `/home/delete/NIRS DB/x_bank/` (one CSV per dataset)

Each real dataset is loaded as a `SpectroDataset`:

```python
from nirs4all.data import SpectroDataset
from pathlib import Path

XBANK_PATH = Path("/home/delete/NIRS DB/x_bank")

def load_xbank_datasets() -> list[SpectroDataset]:
    """Load all X-bank datasets for pretraining."""
    datasets = []
    for csv_file in XBANK_PATH.glob("*.csv"):
        dataset = SpectroDataset.from_csv(csv_file)
        datasets.append(dataset)
    return datasets

# Access components (for pretraining, we only need X and wavelengths)
dataset = datasets[0]
x = dataset.x()                      # Spectral intensities (N × L)
wavelengths = dataset.headers()      # Wavelengths (float)
signal_type = dataset.signal_type()  # 'absorbance', 'reflectance', etc.

# Targets are ONLY used for downstream evaluation, not pretraining
y = dataset.y()  # Only for Phase 0 baselines and downstream heads

# Metadata access (for metadata embedding, optional)
instrument = dataset.metadata_column("instrument")
manufacturer = dataset.metadata_column("manufacturer")
```

**SpectroDataset provides:**
- Variable wavelength grids via `headers()` / `float_headers()`
- Multi-source support for heterogeneous instrument data
- Metadata storage for instrument/manufacturer/mode
- Sample indexing with augmentation tracking

### 3.2 Precomputed AsLS Baseline

Use nirs4all's baseline operators but cache results:

```python
# preprocessing/asls_cache.py
from nirs4all.operators.transforms import ASLSBaseline
import numpy as np
from pathlib import Path

def precompute_baselines(dataset: SpectroDataset, cache_dir: Path, lam=1e6, p=0.01):
    """Precompute and cache AsLS baselines for a dataset."""
    cache_path = cache_dir / f"{dataset.name}_asls.npy"

    if cache_path.exists():
        return np.load(cache_path)

    asls = ASLSBaseline(lam=lam, p=p)
    X = dataset.x()

    # AsLS typically removes baseline; we want the baseline itself
    X_corrected = asls.fit_transform(X)
    baselines = X - X_corrected  # Extract baseline component

    np.save(cache_path, baselines)
    return baselines
```

**CLI Script:**
```bash
python scripts/precompute_asls.py --datasets data/real/ --output data/cache/asls/
```

### 3.3 Synthetic Data via nirs4all.generate

Leverage the comprehensive synthetic generator. **For pretraining, we generate X (spectra) only — targets are NOT needed.**

```python
import nirs4all
from nirs4all.synthesis import SyntheticDatasetBuilder

# Quick generation (no targets needed for self-supervised pretraining)
synth_dataset = nirs4all.generate(
    n_samples=10000,
    complexity="realistic",  # Uses physical Beer-Lambert model
    wavelength_range=(850, 2500),
    as_dataset=True,  # Returns SpectroDataset
)

# Fine-grained control with builder (focus on nuisance diversity, not targets)
synth_dataset = (
    SyntheticDatasetBuilder(n_samples=10000)
    .with_features(
        wavelength_range=(850, 2500),
        wavelength_step=2.0,
        noise_level=(0.005, 0.02),       # Variable noise (not constant!)
        baseline_variation=(0.05, 0.2),   # Variable baseline drift
        scattering_variation=(0.02, 0.1), # Variable scatter
    )
    # NO .with_targets() for pretraining — self-supervised!
    .with_metadata(
        include_instrument=True,
        include_measurement_mode=True,
    )
    .with_batch_effects(
        n_batches=15,  # Simulate different instruments
        batch_variation=0.15,
    )
    .build()
)
```

**Multi-instrument simulation (critical for generalization):**
```python
from nirs4all.synthesis import InstrumentSimulator, MeasurementModeSimulator

# Simulate heterogeneous instruments
instrument_sim = InstrumentSimulator(
    resolution_range=(2.0, 8.0),  # Variable resolution
    noise_profile="wavelength_dependent",
    wavelength_ranges=[
        (850, 1700),   # Portable
        (1000, 2500),  # Lab
        (850, 2500),   # Full range
    ],
)

mode_sim = MeasurementModeSimulator(
    modes=["reflectance", "transmittance", "absorbance"],
)
```

**Paired-view generation (for contrastive invariance):**
```python
# Generate same underlying composition with different instrument effects
def generate_paired_views(builder, n_pairs=5000):
    """Generate paired spectra for contrastive learning."""
    # Same latent composition, different nuisance factors
    view1 = builder.with_batch_effects(batch_variation=0.1).build()
    view2 = builder.with_batch_effects(batch_variation=0.15).build()
    return view1, view2
```

---

## 4. Augmentation Layer: Reusing nirs4all Operators

### 4.1 Mapping: Original Spec → nirs4all Operators

| Original Augmentation | nirs4all Operator | File |
|-----------------------|-------------------|------|
| Baseline drift (smooth additive) | `PolynomialBaselineDrift`, `LinearBaselineDrift` | `operators/augmentation/spectral.py` |
| Scatter (multiplicative + offset) | `MultiplicativeNoise`, `ScatterSimulationMSC` | `operators/augmentation/spectral.py` |
| Noise | `GaussianAdditiveNoise` | `operators/augmentation/spectral.py` |
| Band dropout | `BandMasking`, `ChannelDropout` | `operators/augmentation/spectral.py` |
| Wavelength shift | `WavelengthShift` | `operators/augmentation/spectral.py` |
| Wavelength stretch | `WavelengthStretch` | `operators/augmentation/spectral.py` |
| Local warp | `LocalWavelengthWarp`, `SmoothMagnitudeWarp` | `operators/augmentation/spectral.py` |

### 4.2 Augmentation Adapter

```python
# augmentation/nirs4all_adapters.py
from nirs4all.operators.augmentation import (
    GaussianAdditiveNoise,
    MultiplicativeNoise,
    PolynomialBaselineDrift,
    LinearBaselineDrift,
    BandMasking,
    WavelengthShift,
    ScatterSimulationMSC,
)
import torch
import numpy as np

class NIRSAugmentationPipeline:
    """Wraps nirs4all augmenters for PyTorch training."""

    def __init__(self, p_baseline=0.5, p_scatter=0.5, p_noise=0.8, p_band_mask=0.2):
        self.augmenters = [
            (p_baseline, PolynomialBaselineDrift(degree=3, amplitude=0.05)),
            (p_scatter, MultiplicativeNoise(scale_range=(0.95, 1.05), offset_range=(-0.02, 0.02))),
            (p_noise, GaussianAdditiveNoise(sigma=0.01)),
            (p_band_mask, BandMasking(n_bands=2, band_width=20)),
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentations with specified probabilities."""
        for prob, aug in self.augmenters:
            if np.random.random() < prob:
                x = aug.fit_transform(x.reshape(1, -1)).squeeze()
        return x

    def to_torch(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.__call__(x)).float()
```

### 4.3 MAE Block Masking (NEW - Not in nirs4all)

This is **new functionality** specific to ViT pretraining:

```python
# augmentation/mae_masking.py
import torch
import numpy as np

class MAEBlockMasker:
    """Block-based masking for Masked Autoencoder pretraining.

    Creates contiguous masked regions in wavelength space,
    more realistic than IID masking for spectroscopy.
    """

    def __init__(self, mask_ratio: float = 0.7, min_block_size: int = 5, max_block_size: int = 50):
        self.mask_ratio = mask_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def __call__(self, seq_len: int) -> torch.Tensor:
        """Generate block mask for sequence of given length.

        Returns:
            mask: Tensor[seq_len] with 1=visible, 0=masked
        """
        mask = torch.ones(seq_len, dtype=torch.bool)
        n_mask = int(seq_len * self.mask_ratio)
        masked_count = 0

        while masked_count < n_mask:
            # Random block size
            block_size = np.random.randint(self.min_block_size, self.max_block_size + 1)
            block_size = min(block_size, n_mask - masked_count)

            # Random start position
            start = np.random.randint(0, seq_len - block_size + 1)

            # Mask the block
            mask[start:start + block_size] = False
            masked_count += block_size

        return mask

    def apply(self, x: torch.Tensor, mask_token_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to spectrum.

        Args:
            x: Spectrum tensor [seq_len] or [batch, seq_len]
            mask_token_value: Value to use for masked positions

        Returns:
            x_masked: Spectrum with masked positions replaced
            mask: Boolean mask (1=visible, 0=masked)
        """
        if x.dim() == 1:
            mask = self(len(x))
            x_masked = x.clone()
            x_masked[~mask] = mask_token_value
            return x_masked, mask
        else:
            # Batch processing
            masks = torch.stack([self(x.shape[1]) for _ in range(x.shape[0])])
            x_masked = x.clone()
            x_masked[~masks] = mask_token_value
            return x_masked, masks
```

**Note:** Consider adding `MAEBlockMasker` to nirs4all's augmentation module if it proves useful beyond this project.

---

## 5. Data Splitting: nirs4all Splitters for Downstream

### 5.1 Intra-Dataset Splits

```python
from nirs4all.operators.splitters import KennardStoneSplitter, SPXYSplitter
from sklearn.model_selection import ShuffleSplit

# Representative train/test split using Kennard-Stone
ks_splitter = KennardStoneSplitter(n_train=0.8)

# SPXY for regression tasks (considers both X and y)
spxy_splitter = SPXYSplitter(test_size=0.2)

# Standard cross-validation
cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2)
```

### 5.2 Leave-One-Dataset-Out (LODO) Wrapper (NEW)

```python
# evaluation/lodo.py
from nirs4all.data import SpectroDataset
from typing import Iterator, Tuple, List

class LODOSplitter:
    """Leave-One-Dataset-Out cross-validation for multi-dataset scenarios."""

    def __init__(self, datasets: List[SpectroDataset]):
        self.datasets = datasets
        self.dataset_names = [d.name for d in datasets]

    def split(self) -> Iterator[Tuple[List[SpectroDataset], SpectroDataset]]:
        """Yield (train_datasets, test_dataset) for each fold."""
        for i, test_dataset in enumerate(self.datasets):
            train_datasets = [d for j, d in enumerate(self.datasets) if j != i]
            yield train_datasets, test_dataset

    def __len__(self) -> int:
        return len(self.datasets)
```

---

## 6. Model Architecture (Custom - Outside nirs4all)

### 6.1 Positional Encoding on Wavelengths

```python
# models/encoder/positional_encoding.py
import torch
import torch.nn as nn
import math

class WavelengthPositionalEncoding(nn.Module):
    """Positional encoding using real wavelength values.

    Uses log(λ) for better coverage across typical NIR ranges.
    """

    def __init__(self, d_model: int, use_log: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_log = use_log

        # Learnable scaling parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelengths: [batch, seq_len] or [seq_len] - actual wavelength values

        Returns:
            pe: [batch, seq_len, d_model] positional encodings
        """
        if wavelengths.dim() == 1:
            wavelengths = wavelengths.unsqueeze(0)

        # Optionally use log scale
        if self.use_log:
            wavelengths = torch.log(wavelengths + 1e-6)

        # Normalize
        wavelengths = (wavelengths - wavelengths.mean(dim=-1, keepdim=True)) / (
            wavelengths.std(dim=-1, keepdim=True) + 1e-6
        )
        wavelengths = wavelengths * self.scale + self.bias

        # Sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=wavelengths.device)
            * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(*wavelengths.shape, self.d_model, device=wavelengths.device)
        pe[..., 0::2] = torch.sin(wavelengths.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(wavelengths.unsqueeze(-1) * div_term)

        return pe
```

### 6.2 Spectral Perceiver Architecture

```python
# models/encoder/spectral_perceiver.py
import torch
import torch.nn as nn
from .positional_encoding import WavelengthPositionalEncoding

class SpectralPerceiver(nn.Module):
    """Perceiver-style encoder for variable-length spectra.

    Uses cross-attention from fixed latents to variable input tokens,
    followed by self-attention among latents.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_latents: int = 64,
        n_heads: int = 4,
        n_cross_layers: int = 2,
        n_self_layers: int = 3,
        d_meta: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = WavelengthPositionalEncoding(d_model)

        # Metadata embedding (optional)
        self.meta_proj = nn.Linear(d_meta, d_model) if d_meta > 0 else None

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))

        # Cross-attention layers (latents attend to input)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_cross_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_cross_layers)
        ])

        # Self-attention layers (among latents)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
            for _ in range(n_self_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        meta: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len] spectral intensities
            wavelengths: [batch, seq_len] wavelength values
            meta: [batch, d_meta] optional metadata embedding
            mask: [batch, seq_len] optional mask (1=visible, 0=masked)

        Returns:
            z_latents: [batch, n_latents, d_model] latent representations
            z_embed: [batch, d_model] pooled embedding
        """
        batch_size = x.shape[0]

        # Project input to d_model
        tokens = self.input_proj(x.unsqueeze(-1))  # [B, L, D]

        # Add positional encoding
        tokens = tokens + self.pos_enc(wavelengths)

        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)

        # Inject metadata into latents if provided
        if meta is not None and self.meta_proj is not None:
            meta_emb = self.meta_proj(meta).unsqueeze(1)  # [B, 1, D]
            latents = latents + meta_emb

        # Cross-attention: latents attend to tokens
        for cross_attn, norm in zip(self.cross_attn_layers, self.cross_norms):
            attn_out, _ = cross_attn(latents, tokens, tokens, key_padding_mask=~mask if mask is not None else None)
            latents = norm(latents + attn_out)

        # Self-attention among latents
        for self_attn in self.self_attn_layers:
            latents = self_attn(latents)

        latents = self.final_norm(latents)

        # Pool to single embedding (mean over latents)
        z_embed = latents.mean(dim=1)

        return latents, z_embed
```

---

## 7. Training Infrastructure

### 7.1 Multi-Dataset DataLoader (Uniform-by-Dataset)

```python
# data/multi_dataset.py
from nirs4all.data import SpectroDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiSpectralDataset(Dataset):
    """Multi-dataset wrapper with uniform-by-dataset sampling.

    Implements:
    - Uniform-by-dataset sampling (prevents large datasets from dominating)
    - Virtual epochs (fixed steps_per_epoch regardless of dataset sizes)
    - Synthetic mixing with probability p_synth
    """

    def __init__(
        self,
        real_datasets: list[SpectroDataset],
        synth_dataset: SpectroDataset | None = None,
        p_synth: float = 0.3,
        steps_per_epoch: int | None = None,  # Fixed epoch length (virtual epoch)
        augmentation = None,
        baseline_cache: dict[str, np.ndarray] | None = None,
    ):
        self.real_datasets = real_datasets
        self.synth_dataset = synth_dataset
        self.p_synth = p_synth
        self.augmentation = augmentation
        self.baseline_cache = baseline_cache or {}

        # Virtual epoch: K steps per dataset (default K=200)
        K = 200
        if steps_per_epoch is None:
            self.steps_per_epoch = K * len(real_datasets)  # e.g., 200 × 45 = 9000
        else:
            self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        # Virtual epoch length (NOT total samples)
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # Uniform-by-dataset sampling
        if self.synth_dataset and np.random.random() < self.p_synth:
            dataset = self.synth_dataset
            sample_idx = np.random.randint(dataset.num_samples)
        else:
            # Pick dataset uniformly
            dataset = np.random.choice(self.real_datasets)
            # Pick sample uniformly within dataset
            sample_idx = np.random.randint(dataset.num_samples)

        # Extract sample
        x = dataset.x()[sample_idx]
        wavelengths = np.array(dataset.float_headers())

        # Get baseline if cached
        baseline = None
        if dataset.name in self.baseline_cache:
            baseline = self.baseline_cache[dataset.name][sample_idx]

        # Get metadata
        meta = self._extract_metadata(dataset, sample_idx)

        # Apply augmentation
        if self.augmentation:
            x = self.augmentation(x)

        return {
            'x': torch.from_numpy(x).float(),
            'wavelengths': torch.from_numpy(wavelengths).float(),
            'baseline': torch.from_numpy(baseline).float() if baseline is not None else None,
            'meta': meta,
            'dataset_id': dataset.name,
        }

    def _extract_metadata(self, dataset, idx):
        """Extract metadata dict for embedding."""
        meta = {}
        for col in ['instrument', 'manufacturer', 'mode']:
            try:
                meta[col] = dataset.metadata_column(col)[idx]
            except:
                meta[col] = "UNKNOWN"
        return meta


def variable_length_collate(batch):
    """Collate function for variable-length spectra (no padding)."""
    return {
        'x': [b['x'] for b in batch],
        'wavelengths': [b['wavelengths'] for b in batch],
        'baseline': [b['baseline'] for b in batch],
        'meta': [b['meta'] for b in batch],
        'dataset_id': [b['dataset_id'] for b in batch],
    }
```

### 7.2 Training Loop (Self-Supervised)

```python
# training/pretrain.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_epoch(model, decoder, dataloader, optimizer, masker, config, device):
    model.train()
    decoder.train()

    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        losses = []
        for i in range(len(batch['x'])):
            x = batch['x'][i].to(device)
            wavelengths = batch['wavelengths'][i].to(device)
            baseline = batch['baseline'][i]

            # 1. Per-spectrum normalization
            x_norm = (x - x.mean()) / (x.std() + 1e-6)

            # 2. MAE masking
            x_masked, mask = masker.apply(x_norm)

            # 3. Forward encoder
            z_latents, z_embed = model(
                x_masked.unsqueeze(0),
                wavelengths.unsqueeze(0),
                mask=mask.unsqueeze(0)
            )

            # 4. Decode masked positions
            x_hat = decoder(z_latents, wavelengths.unsqueeze(0), mask.unsqueeze(0))

            # 5. MAE loss (only on masked positions)
            L_mae = ((x_hat.squeeze() - x_norm)[~mask] ** 2).mean()
            losses.append(L_mae)

            # 6. Baseline-consistency loss (if baseline available)
            if baseline is not None:
                baseline = baseline.to(device)
                x_corrected = x_norm - (baseline - baseline.mean()) / (baseline.std() + 1e-6)
                _, z_embed_corrected = model(
                    x_corrected.unsqueeze(0),
                    wavelengths.unsqueeze(0),
                )
                L_base = ((z_embed - z_embed_corrected) ** 2).mean()
                losses.append(config.lambda_baseline * L_base)

        # Backprop
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

---

## 8. Components to Add to nirs4all (Future)

During ViT development, the following operators may warrant inclusion in nirs4all:

### 8.1 Potential New Operators

| Operator | Module | Description |
|----------|--------|-------------|
| `MAEBlockMasker` | `operators/augmentation/` | Block-based masking for self-supervised pretraining |
| `BaselineCacher` | `operators/transforms/` | Precompute and cache baseline corrections |
| `UniformDatasetSampler` | `operators/splitters/` | Uniform-by-dataset sampling for multi-dataset training |
| `LODOSplitter` | `operators/splitters/` | Leave-One-Dataset-Out cross-validation |

### 8.2 Potential SpectroDataset Extensions

- **Baseline caching**: Add `set_cached_baseline()` / `cached_baseline()` methods
- **Pre-tokenized storage**: Store pre-computed (λ, x) tokens for efficient loading

---

## 9. Implementation Phases (Aligned with Original Spec)

### Phase 0: Baselines (Using nirs4all Pipelines)
```python
import nirs4all
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import ASLSBaseline, SNV

# Classical baseline
result = nirs4all.run(
    pipeline=[ASLSBaseline(), SNV(), PLSRegression(n_components=10)],
    dataset="path/to/dataset",
)
print(f"PLS RMSE: {result.best_rmse:.4f}")
```

### Phase 1: Minimal Encoder Feasibility
- Implement `SpectralPerceiver` + `MAEDecoder`
- Train MAE-only (no augmentation, no meta)
- Compare embedding + Ridge vs PCA + Ridge

### Phase 2: Ablation Study
- A) Classical: nirs4all pipeline (ASLSBaseline + SNV + PLS)
- B) Encoder MAE only
- C) Encoder MAE + nirs4all augmentations
- D) Encoder MAE + augmentations + baseline-consistency loss

### Phase 3: LODO Evaluation
- Use `LODOSplitter` with all 45 datasets
- Compare methods on held-out datasets

### Phase 4: Synthetic Data Impact
- Vary `p_synth` in `MultiSpectralDataset`
- Compare real-only vs real+synth pretraining

### Phase 5: Metadata Impact
- Enable/disable metadata embedding
- Test with metadata dropout

### Phase 6: Final Downstream Evaluation
- Embedding + Ridge/PLS/TabPFN vs heavy pipelines

---

## 10. Configuration System

```yaml
# configs/default.yaml
data:
  real_datasets_dir: "/home/delete/NIRS DB/x_bank"  # X-bank location
  synth_samples: 10000
  synth_complexity: "realistic"

# Training strategy (see Section 1)
training_strategy:
  use_targets_in_pretraining: false  # Self-supervised only
  sampling: "uniform_by_dataset"
  steps_per_epoch: 9000              # K=200 × 45 datasets
  p_synth: 0.3                       # 30% synthetic mix

augmentation:
  p_baseline_drift: 0.5
  p_scatter: 0.5
  p_noise: 0.8
  p_band_mask: 0.2

masking:
  mask_ratio: 0.7
  min_block_size: 5
  max_block_size: 50

model:
  d_model: 128
  n_latents: 64
  n_heads: 4
  n_cross_layers: 2
  n_self_layers: 3
  d_meta: 32

training:
  epochs: 200
  batch_size: 64  # Adjust per GPU (3090: 64, 4090: 96-128, A100: 128-256)
  lr: 1e-4
  weight_decay: 0.05
  lambda_baseline: 0.1
  warmup_epochs: 10

losses:
  mae_weight: 1.0
  baseline_consistency_weight: 0.1
  contrastive_weight: 0.0  # Optional, enable for paired-view training

evaluation:
  downstream_methods: ["ridge", "pls", "tabpfn"]
```

---

## 11. Deliverables Checklist

- [ ] **Data Layer**
  - [ ] X-bank loader (`/home/delete/NIRS DB/x_bank/`)
  - [ ] SpectroDataset loaders for real datasets
  - [ ] AsLS baseline precomputation script
  - [ ] Synthetic data generation configs (no targets)
  - [ ] Multi-dataset wrapper with uniform-by-dataset sampling
  - [ ] Virtual epoch implementation (steps_per_epoch)
  - [ ] Variable-length collation

- [ ] **Augmentation**
  - [ ] nirs4all adapter wrappers
  - [ ] MAE block masking implementation
  - [ ] Paired-view generation for contrastive learning
  - [ ] Composition pipeline

- [ ] **Model**
  - [ ] WavelengthPositionalEncoding
  - [ ] SpectralPerceiver encoder
  - [ ] MAE decoder
  - [ ] Metadata embedding (optional)

- [ ] **Training**
  - [ ] Self-supervised pretraining script (no targets)
  - [ ] Baseline-consistency loss
  - [ ] Contrastive loss (optional, for paired views)
  - [ ] Mixed precision support
  - [ ] Checkpoint saving

- [ ] **Synthetic Data Validation**
  - [ ] Linear separability check (synthetic vs real)
  - [ ] Downstream lift comparison (with/without synthetic)
  - [ ] Paired-view invariance test

- [ ] **Evaluation**
  - [ ] LODO evaluation script
  - [ ] Downstream task evaluation (targets used HERE only)
  - [ ] Ablation framework
  - [ ] Robustness testing (inject drift/scatter)

- [ ] **Documentation**
  - [ ] Setup instructions
  - [ ] Configuration reference
  - [ ] Results reporting (metrics JSON + plots)

---

## 12. Key Differences from Original Spec

| Aspect | Original Spec | nirs4all-Integrated |
|--------|---------------|---------------------|
| Data location | Custom `data/real/` | `/home/delete/NIRS DB/x_bank/` |
| Data format | Custom numpy files | `SpectroDataset` objects |
| Synthetic generation | Custom `generator.py` | `nirs4all.generate()` (no targets) |
| Augmentations | From scratch | Reuse `operators/augmentation/` |
| Baseline correction | Custom AsLS | `ASLSBaseline` + caching |
| Splitting | Custom | nirs4all splitters + LODO wrapper |
| Baselines | Manual | nirs4all pipelines |
| Training loop | From scratch | Custom (outside nirs4all pipeline) |
| Sampling | Implicit | Explicit uniform-by-dataset + virtual epochs |
| Target usage | Ambiguous | Clear: self-supervised pretraining, targets for downstream only |

### Summary: When Targets Are Used

| Phase | Uses Targets? | Description |
|-------|---------------|-------------|
| **Pretraining** | ❌ No | Self-supervised (MAE + invariance losses) |
| **Phase 0 Baselines** | ✅ Yes | Classical methods (PLS, Ridge) for reference |
| **Downstream Evaluation** | ✅ Yes | Train lightweight heads on embeddings |
| **Optional Fine-tuning** | ✅ Yes | After pretraining, with small LR |

This integration provides a solid foundation of data handling and preprocessing while allowing full flexibility for custom ViT training.
