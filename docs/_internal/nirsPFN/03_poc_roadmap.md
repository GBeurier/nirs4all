# NIRS Spectral Encoder for TabPFN: PoC Roadmap

## Implementation Strategy for nirs4all

This document provides a detailed technical roadmap for implementing a **NIRS Spectral Encoder** that improves upon the current SOTA pipeline (`ASLSBaseline → PCA(0.99) → TabPFN`).

---

## ⚠️ Revised Approach: Spectral Encoder, Not Full PFN

**Key insight from current SOTA**: TabPFN already works excellently for NIRS when fed proper features. The bottleneck is feature extraction, not in-context learning.

**Current pipeline**:
```
Raw Spectra → ASLSBaseline → PCA(0.99) → TabPFN → Predictions
                   ↓              ↓
              [Heuristic]    [Loses structure]
```

**Proposed pipeline**:
```
Raw Spectra → NIRSEncoder (learned) → TabPFN → Predictions
                    ↓
         [Preserves band structure]
         [Learns baseline invariance]
         [Trained on synthetic data]
```

---

## 1. Proof-of-Concept Objectives

### 1.1 What We're Proving

| Hypothesis | How We Validate |
|------------|-----------------|
| A learned encoder beats PCA | Compare NIRSEncoder→TabPFN vs PCA→TabPFN |
| Synthetic training transfers to real | Train encoder on synthetic, evaluate on real |
| Band structure matters | Ablation: with/without positional encoding |
| Baseline invariance can be learned | Compare with/without ASLSBaseline preprocessing |

### 1.2 Success Criteria for PoC

| Metric | Baseline (ASLSBaseline+PCA+TabPFN) | Target | Stretch |
|--------|-------------------------------------|--------|---------|
| RMSE improvement | 0% (reference) | **>5%** | >15% |
| R² improvement | 0% (reference) | **>0.02** | >0.05 |
| Few-shot (n=25) | Current perf | **+10%** | +20% |
| Inference time | Current | **≤1.5×** | ≤1× |
| Training time | - | **<8 GPU-hours** | <2 hours |

### 1.3 Out of Scope for PoC

- Full NIRS-PFN architecture (use existing TabPFN)
- Multi-task simultaneous training
- Instrument-specific adaptation
- Uncertainty quantification

---

## 2. Technical Implementation Plan

### 2.1 Repository Structure

```
nirs4all/
├── nirs4all/
│   ├── data/
│   │   └── synthetic/
│   │       └── encoder_training.py  # NEW: Encoder training data generator
│   │
│   └── operators/
│       └── transforms/
│           └── nirs_encoder.py  # NEW: Learned spectral encoder
│
├── bench/
│   └── nirsPFN/
│       ├── 01_synthetic_generator_analysis.md
│       ├── 02_nirspfn_development_plan.md
│       ├── 03_poc_roadmap.md                   # This doc
│       │
│       ├── experiments/
│       │   ├── exp01_current_baseline.py       # Document current SOTA
│       │   ├── exp02_encoder_architectures.py  # Test encoder variants
│       │   ├── exp03_synthetic_training.py     # Train on synthetic
│       │   ├── exp04_transfer_evaluation.py    # Evaluate on real data
│       │   └── exp05_ablations.py              # Component ablations
│       │
│       └── training/
│           ├── train_encoder.py
│           └── checkpoints/
```

### 2.2 Development Sprints (4 weeks)

---

## Sprint 0: Baseline Documentation (Days 1-2)

**Goal**: Document current SOTA performance precisely.

```python
# bench/nirsPFN/experiments/exp01_current_baseline.py
"""
Document exact performance of current SOTA: ASLSBaseline → PCA(0.99) → TabPFN
"""

from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import ASLSBaseline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from tabpfn import TabPFNRegressor

DATASETS = {
    "corn": "path/to/corn",
    "tecator": "path/to/tecator",
    "tablets": "path/to/tablets",
}

def run_baseline_documentation():
    """Document the exact current SOTA performance."""

    for name, path in DATASETS.items():
        pipeline = [
            {"split": "repeated_kfold", "n_splits": 5, "n_repeats": 3},
            {"_or_": [
                # Reference: PLS
                {"model": PLSRegression(n_components=10), "name": "PLS-10"},

                # Current SOTA: ASLSBaseline + PCA + TabPFN
                [
                    ASLSBaseline(),
                    PCA(n_components=0.99),
                    {"model": TabPFNRegressor(), "name": "SOTA"}
                ],
            ]}
        ]

        runner = PipelineRunner(verbose=1)
        runner.run(PipelineConfigs(pipeline, "baseline"), DatasetConfigs(path))
```

**Deliverables**:
- [ ] Exact RMSE/R² for ASLSBaseline+PCA+TabPFN on all benchmark datasets
- [ ] Few-shot performance curve (n=10, 25, 50, 100)
- [ ] Baseline document with reproducible numbers

---

## Sprint 1: NIRS Encoder Architecture (Days 3-7)

**Goal**: Implement learnable spectral encoder.

### 1.1 Design Principles

The encoder must:
1. **Reduce dimensionality**: ~1000-2000 wavelengths → ~100-300 features
2. **Preserve band structure**: Keep peaks, shoulders, band ratios
3. **Learn baseline invariance**: Replace heuristic ASLSBaseline
4. **Be TabPFN-compatible**: Output features TabPFN can consume

### 1.2 Encoder Architectures

```python
# nirs4all/operators/transforms/nirs_encoder.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NIRSEncoderV1(nn.Module):
    """Option A: 1D-CNN Encoder - Simple, fast, proven."""

    def __init__(self, input_dim: int = 1000, output_dim: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class NIRSEncoderV2(nn.Module):
    """Option B: Patch Transformer - Better for band relationships."""

    def __init__(
        self, input_dim: int = 1000, output_dim: int = 128,
        patch_size: int = 20, embed_dim: int = 64, num_layers: int = 2,
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = input_dim // patch_size

        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Create patches
        x = x.unfold(1, self.patch_size, self.patch_size)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.size(1), :]

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        return self.output(x[:, 0, :])


class NIRSEncoderTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper for NIRS encoder."""

    def __init__(
        self, encoder_path: str = None, encoder_type: str = "v2",
        output_dim: int = 128, device: str = "cuda",
    ):
        self.encoder_path = encoder_path
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.device = device
        self._encoder = None

    @classmethod
    def pretrained(cls, device: str = "cuda"):
        """Load pretrained encoder."""
        return cls(
            encoder_path="nirs4all/models/pretrained/nirs_encoder_v2.pt",
            encoder_type="v2", device=device
        )

    def fit(self, X, y=None):
        if self._encoder is None:
            EncoderClass = NIRSEncoderV2 if self.encoder_type == "v2" else NIRSEncoderV1
            self._encoder = EncoderClass(X.shape[1], self.output_dim).to(self.device)
            if self.encoder_path:
                self._encoder.load_state_dict(torch.load(self.encoder_path))
            self._encoder.eval()
        return self

    def transform(self, X):
        with torch.no_grad():
            X_t = torch.from_numpy(np.array(X)).float().to(self.device)
            return self._encoder(X_t).cpu().numpy()
```

**Deliverables**:
- [ ] CNN encoder (V1) and Transformer encoder (V2)
- [ ] sklearn-compatible `NIRSEncoderTransformer` wrapper
- [ ] Unit tests and benchmarks

---

## Sprint 2: Encoder Training (Days 8-14)

**Goal**: Train encoder on synthetic NIRS data with contrastive learning.

### 2.1 Training Objective

We train with **contrastive learning** - no labels needed:
- Positive pairs: same spectrum with different augmentations
- Negative pairs: different spectra

### 2.2 Training Data Generator

```python
# nirs4all/synthesis/encoder_training.py

import numpy as np
from .builder import SyntheticDatasetBuilder


class EncoderTrainingDataGenerator:
    """Generate paired spectra for contrastive learning."""

    def __init__(self, wavelength_range=(900, 1700), batch_size=256, random_state=None):
        self.wavelength_range = wavelength_range
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)

    def _augment(self, spectrum):
        """Apply random augmentations."""
        aug = spectrum.copy()

        # Baseline shift
        if self.rng.random() < 0.5:
            x = np.linspace(0, 1, len(spectrum))
            aug += np.polyval(self.rng.uniform(-0.1, 0.1, 3), x)

        # Multiplicative scatter
        if self.rng.random() < 0.5:
            aug *= 1 + self.rng.uniform(-0.1, 0.1)

        # Noise
        aug += self.rng.normal(0, 0.005, len(spectrum))

        return aug

    def generate_batch(self):
        builder = SyntheticDatasetBuilder(
            n_samples=self.batch_size,
            random_state=int(self.rng.integers(0, 2**31))
        ).with_features(
            wavelength_range=self.wavelength_range,
            complexity="realistic",
        ).with_targets(range=(0, 100))

        X, y = builder.build_arrays()

        return {
            "anchor": np.array([self._augment(s) for s in X]),
            "positive": np.array([self._augment(s) for s in X]),
        }

    def __iter__(self):
        while True:
            yield self.generate_batch()
```

### 2.3 Training Script

```python
# bench/nirsPFN/training/train_encoder.py

import torch
import torch.nn.functional as F
from torch.optim import AdamW


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        similarity = torch.mm(anchor, positive.t()) / self.temperature
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(similarity, labels)


def train_encoder(num_epochs=100, batch_size=256, device="cuda"):
    from nirs4all.synthesis.encoder_training import EncoderTrainingDataGenerator
    from nirs4all.operators.transforms.nirs_encoder import NIRSEncoderV2

    data_gen = EncoderTrainingDataGenerator(batch_size=batch_size)
    sample = next(iter(data_gen))

    encoder = NIRSEncoderV2(input_dim=sample["anchor"].shape[1]).to(device)
    optimizer = AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = InfoNCELoss()

    for epoch in range(num_epochs):
        encoder.train()
        epoch_loss = 0

        for i, batch in enumerate(data_gen):
            if i >= 100:  # batches per epoch
                break

            anchor = torch.from_numpy(batch["anchor"]).float().to(device)
            positive = torch.from_numpy(batch["positive"]).float().to(device)

            optimizer.zero_grad()
            loss = criterion(encoder(anchor), encoder(positive))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/100:.4f}")

    torch.save(encoder.state_dict(), "encoder_v2_final.pt")
    return encoder
```

**Deliverables**:
- [ ] Contrastive training pipeline
- [ ] Trained encoder checkpoints
- [ ] Training curves and validation

---

## Sprint 3: Evaluation (Days 15-21)

**Goal**: Compare learned encoder vs current SOTA.

```python
# bench/nirsPFN/experiments/exp04_transfer_evaluation.py

def run_evaluation():
    """Compare learned encoder vs ASLSBaseline+PCA."""

    for name, path in DATASETS.items():
        pipeline = [
            {"split": "repeated_kfold", "n_splits": 5, "n_repeats": 3},
            {"_or_": [
                # Current SOTA
                [ASLSBaseline(), PCA(n_components=0.99),
                 {"model": TabPFNRegressor(), "name": "SOTA"}],

                # NEW: Learned encoder
                [NIRSEncoderTransformer(encoder_path="encoder_v2_final.pt"),
                 {"model": TabPFNRegressor(), "name": "NIRSEncoder"}],
            ]}
        ]

        runner = PipelineRunner(verbose=1)
        runner.run(PipelineConfigs(pipeline, "eval"), DatasetConfigs(path))
```

**Deliverables**:
- [ ] Comparison table (RMSE, R² per dataset)
- [ ] Few-shot evaluation
- [ ] Ablation studies
- [ ] Go/no-go decision

---

## Sprint 4: Integration (Days 22-28)

**Goal**: Package for nirs4all.

**Deliverables**:
- [ ] `NIRSEncoderTransformer.pretrained()` convenience method
- [ ] Pretrained weights packaged
- [ ] User documentation
- [ ] Example notebook

---

## 3. Timeline Summary

| Week | Sprint | Key Deliverables |
|------|--------|------------------|
| 1 | 0-1 | Baseline docs, encoder implementation |
| 2 | 2 | Trained encoder |
| 3 | 3 | Evaluation, ablations, go/no-go |
| 4 | 4 | Integration, documentation |

---

## 4. Go/No-Go Criteria

### Week 2 Checkpoint

| Metric | Go | No-Go |
|--------|-----|-------|
| RMSE vs SOTA | ≤ 1.05× | > 1.2× |
| R² vs SOTA | ≥ -0.02 | < -0.05 |

### Week 4 Final

| Metric | Ship It | Iterate | Abandon |
|--------|---------|---------|---------|
| Avg RMSE improvement | >5% | 0-5% | <0% |
| Datasets improved | ≥3/4 | 2/4 | ≤1/4 |

---

## 5. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Encoder overfits to synthetic | Aggressive augmentation, early stopping |
| Distribution shift | Diverse synthetic priors |
| TabPFN doesn't like encoder features | Match PCA output statistics |

---

## 6. Next Steps After PoC

**If success**: Optimize encoder, expand prior, publish findings

**If failure**: Analyze bottleneck, try hybrid (encoder + PCA), consider full NIRS-PFN

---

*NIRS Spectral Encoder PoC Roadmap v2.0*
*Revised strategy based on ASLSBaseline+PCA+TabPFN SOTA*
*January 2026*
