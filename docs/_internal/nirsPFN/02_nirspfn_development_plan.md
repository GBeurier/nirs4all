# NIRS Spectral Encoder: Development Plan

## Improving TabPFN Performance for Near-Infrared Spectroscopy

---

## Executive Summary

### The Key Insight

**Current SOTA**: `ASLSBaseline → PCA(0.99) → TabPFN` significantly outperforms traditional methods.

This tells us:
1. ✅ **TabPFN's in-context learning works** for NIRS data
2. ❌ **PCA is the bottleneck** - it discards spectral structure
3. ❌ **ASLSBaseline is heuristic** - a learned approach could be better

### The Strategy

Instead of building a full NIRS-PFN from scratch (high risk, 6+ months), we will:

1. **Train a spectral encoder** on synthetic data (4 weeks)
2. **Replace PCA** with the learned encoder
3. **Keep TabPFN** as the prediction backbone

```
OLD:  Spectra → ASLSBaseline → PCA(0.99) → TabPFN → Predictions
                    ↓              ↓
               [Heuristic]   [Loses structure]

NEW:  Spectra → NIRSEncoder (learned) → TabPFN → Predictions
                        ↓
              [Preserves bands]
              [Baseline invariant]
              [Trained on synthetic]
```

### Expected Outcomes

| Metric | Target | Rationale |
|--------|--------|-----------|
| RMSE improvement | >5% | Better features → better predictions |
| Few-shot (n=25) | >10% | Structure preservation helps small samples |
| Inference overhead | <50% | Encoder is lightweight |
| Development time | 4 weeks | Focused scope |

---

## 1. Why Not Build a Full NIRS-PFN?

### 1.1 Risk-Reward Analysis

| Approach | Risk | Time | Compute | Potential Gain |
|----------|------|------|---------|----------------|
| **Full NIRS-PFN** | 🔴 High | 6-7 months | ~500 GPU-hrs | Uncertain |
| **Fine-tune TabPFN** | 🟡 Medium | 2-3 months | ~100 GPU-hrs | Moderate |
| **Spectral Encoder** | 🟢 Low | 4 weeks | ~8 GPU-hrs | High |

### 1.2 Why Encoder-First is Better

1. **Proven foundation**: TabPFN already works - don't reinvent it
2. **Lower risk**: If encoder fails, we've lost 4 weeks, not 6 months
3. **Composable**: Encoder can be used with any downstream model
4. **Interpretable**: We can analyze what the encoder learns
5. **Fast iteration**: Train in hours, not weeks

### 1.3 When to Consider Full NIRS-PFN

Revisit full NIRS-PFN only if:
- Encoder + TabPFN plateaus
- TabPFN itself becomes the bottleneck
- We need domain-specific uncertainty quantification
- Cross-instrument transfer fails completely

---

## 2. Technical Approach

### 2.1 What the Encoder Must Do

| Requirement | Why | How |
|-------------|-----|-----|
| Reduce dimensions | TabPFN needs ~100-300 features | Compression layers |
| Preserve band structure | Peaks, shoulders, ratios matter | Positional encoding |
| Learn baseline invariance | Replace ASLSBaseline | Contrastive training |
| Generalize across instruments | Real-world variation | Synthetic diversity |
| Be fast | Production use | Lightweight architecture |

### 2.2 Architecture Options

#### Option A: 1D-CNN (Simple, Fast)

```
Spectrum (1000 dims)
    ↓
Conv1D(32) → BN → ReLU → MaxPool
    ↓
Conv1D(64) → BN → ReLU → MaxPool
    ↓
Conv1D(128) → BN → ReLU → MaxPool
    ↓
Flatten → FC(256) → FC(128)
    ↓
Features (128 dims)
```

**Pros**: Fast, proven for spectral data, easy to train
**Cons**: Limited receptive field, may miss long-range patterns

#### Option B: Patch Transformer (Recommended)

```
Spectrum (1000 dims)
    ↓
Split into patches (50 × 20 dims)
    ↓
Patch embedding + Positional encoding
    ↓
Transformer Encoder (2-4 layers)
    ↓
CLS token → FC(128)
    ↓
Features (128 dims)
```

**Pros**: Captures band relationships, wavelength-aware, scalable
**Cons**: Slightly more complex, needs positional encoding

#### Option C: Hybrid CNN + Attention

```
Spectrum (1000 dims)
    ↓
Local Conv layers (extract peaks, shoulders)
    ↓
Downsample
    ↓
Self-attention (global band relationships)
    ↓
Pool + FC(128)
    ↓
Features (128 dims)
```

**Pros**: Best of both worlds
**Cons**: More hyperparameters

### 2.3 Recommended: Option B (Patch Transformer)

**Rationale**:
- Explicit wavelength position encoding (bands have meaning)
- Self-attention captures peak-to-peak relationships
- Similar to successful ViT approach in vision
- Scales to variable wavelength counts

### 2.4 Training Approach: Contrastive Learning

**Key insight**: We don't need labels to train the encoder!

#### Training Objective

```python
# Positive pairs: same spectrum, different augmentations
anchor = augment(spectrum)      # e.g., add baseline, noise
positive = augment(spectrum)    # different augmentation

# Encoder should map them to similar embeddings
z_anchor = encoder(anchor)
z_positive = encoder(positive)

# InfoNCE loss: maximize similarity of positive pairs
loss = -log(exp(sim(z_anchor, z_positive)) / sum(exp(sim(z_anchor, z_neg))))
```

#### Augmentations (Synthetic Invariances)

The encoder learns to be invariant to:

| Augmentation | What it teaches |
|--------------|-----------------|
| Baseline shift (polynomial) | Baseline invariance |
| Multiplicative scatter | SNV-like invariance |
| Additive noise | Noise robustness |
| Wavelength shift | Calibration drift tolerance |
| Peak broadening | Instrument variation |

This is more powerful than ASLSBaseline because the invariances are **learned from data**, not hand-coded.

---

## 3. Synthetic Data Strategy

### 3.1 Current Generator Capabilities

The nirs4all synthetic generator already provides:

| Feature | Status | Sufficient for Encoder? |
|---------|--------|------------------------|
| Beer-Lambert physics | ✅ Implemented | ✅ Yes |
| 31 predefined components | ✅ Implemented | ⚠️ Needs expansion |
| Baseline effects | ✅ Implemented | ✅ Yes |
| Scatter effects | ✅ Implemented | ✅ Yes |
| Noise models | ✅ Implemented | ✅ Yes |
| Batch effects | ✅ Implemented | ✅ Yes |
| Non-linear targets | ✅ Implemented | ✅ Yes (for optional supervised loss) |

### 3.2 Minimal Extensions Needed

For encoder training, we need **spectral shape diversity**, not task diversity:

| Extension | Priority | Effort | Impact |
|-----------|----------|--------|--------|
| Random component generation | High | 1-2 days | More spectral shapes |
| Wavelength range variation | High | 1 day | Generalization |
| Peak width variation | Medium | 1 day | Instrument diversity |
| Temperature effects | Low | 2 days | Optional |

### 3.3 Training Data Generation

```python
# Simplified: generate diverse spectra for contrastive training
def generate_training_batch(batch_size=256):
    spectra = []
    for _ in range(batch_size):
        # Random configuration each time
        config = {
            "n_components": random.randint(3, 10),
            "wavelength_range": random_range(),
            "noise_level": random.uniform(0.001, 0.02),
            "baseline_degree": random.randint(0, 3),
        }
        spectrum = generate_spectrum(config)
        spectra.append(spectrum)
    return np.array(spectra)
```

---

## 4. Development Phases

### Phase 1: Baseline Documentation (Days 1-2)

**Goal**: Establish precise baseline metrics.

**Deliverables**:
- [ ] Exact RMSE/R² for ASLSBaseline+PCA+TabPFN on all benchmarks
- [ ] Ablation: ASLSBaseline contribution vs PCA contribution
- [ ] Few-shot curves (n=10, 25, 50, 100)
- [ ] Inference time measurements

**Go/No-Go**: Baseline documented → Proceed

---

### Phase 2: Encoder Implementation (Days 3-7)

**Goal**: Implement encoder architectures.

**Deliverables**:
- [ ] NIRSEncoderV1 (CNN)
- [ ] NIRSEncoderV2 (Transformer)
- [ ] NIRSEncoderV3 (Hybrid)
- [ ] sklearn-compatible wrapper
- [ ] Unit tests
- [ ] Forward pass benchmarks

**Architecture Specs**:

```python
# Recommended configuration
NIRSEncoderV2:
    input_dim: variable (typically 400-2000)
    output_dim: 128
    patch_size: 20
    embed_dim: 64
    num_heads: 4
    num_layers: 2
    parameters: ~200K
```

**Go/No-Go**: All architectures working → Proceed

---

### Phase 3: Training Infrastructure (Days 8-10)

**Goal**: Set up contrastive training pipeline.

**Deliverables**:
- [ ] EncoderTrainingDataGenerator
- [ ] Augmentation functions
- [ ] InfoNCE loss implementation
- [ ] Training script with logging (W&B)
- [ ] Checkpoint saving

**Training Config**:

```python
TRAINING_CONFIG = {
    "batch_size": 256,
    "batches_per_epoch": 100,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "temperature": 0.07,  # InfoNCE temperature
    "gpu": "single A100 or V100",
    "estimated_time": "2-4 hours",
}
```

**Go/No-Go**: Training converges on synthetic → Proceed

---

### Phase 4: Training and Validation (Days 11-14)

**Goal**: Train encoder and validate on held-out synthetic data.

**Experiments**:
1. Train NIRSEncoderV2 with contrastive loss
2. Compare architectures (V1 vs V2 vs V3)
3. Ablate augmentation strategies
4. Validate on held-out synthetic datasets

**Validation Metrics**:
- Contrastive loss convergence
- t-SNE visualization of embeddings
- Linear probe accuracy on synthetic data

**Intermediate Go/No-Go (Day 12)**:

| Metric | Go | Iterate | Abandon |
|--------|-----|---------|---------|
| Loss converging | ✅ < 2.0 | 2.0-3.0 | > 3.0 |
| Embeddings cluster | ✅ Visible | Weak | None |

---

### Phase 5: Real Data Evaluation (Days 15-21)

**Goal**: Evaluate encoder on real benchmark datasets.

**Experiments**:
1. Replace PCA with trained encoder
2. Compare: NIRSEncoder+TabPFN vs ASLSBaseline+PCA+TabPFN
3. Ablation: with/without ASLSBaseline before encoder
4. Few-shot evaluation
5. Cross-dataset generalization

**Benchmark Datasets**:

| Dataset | Domain | Samples | Wavelengths | Task |
|---------|--------|---------|-------------|------|
| Corn | Agriculture | 80 | 700 | Regression |
| Tecator | Food | 215 | 100 | Regression |
| Tablets | Pharma | 654 | 650 | Regression |
| Wheat | Agriculture | 155 | 1050 | Regression |

**Success Metrics**:

| Metric | Target | Stretch |
|--------|--------|---------|
| Avg RMSE improvement | >5% | >15% |
| Datasets improved | ≥3/4 | 4/4 |
| Few-shot (n=25) improvement | >10% | >20% |

**Go/No-Go**:

| Result | Decision |
|--------|----------|
| >5% improvement on ≥3 datasets | ✅ Ship it |
| 0-5% improvement | 🔄 Iterate (more training, different arch) |
| Worse than baseline | ❌ Analyze failure, consider alternatives |

---

### Phase 6: Integration and Documentation (Days 22-28)

**Goal**: Package for nirs4all.

**Deliverables**:
- [ ] `NIRSEncoderTransformer` operator in nirs4all
- [ ] `NIRSEncoderTransformer.pretrained()` method
- [ ] Pretrained weights packaged
- [ ] User documentation
- [ ] Example notebook
- [ ] Performance comparison table

**Integration**:

```python
# Final user-facing API
from nirs4all.operators.transforms import NIRSEncoderTransformer
from tabpfn import TabPFNRegressor

pipeline = [
    ShuffleSplit(n_splits=5),
    NIRSEncoderTransformer.pretrained(),  # Drop-in replacement for ASLSBaseline+PCA
    {"model": TabPFNRegressor(), "name": "NIRSEncoder+TabPFN"}
]
```

---

## 5. Risk Analysis

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Encoder doesn't transfer to real data | Medium | High | Diverse synthetic training, domain randomization |
| Overfits to synthetic patterns | Medium | Medium | Strong augmentation, early stopping |
| TabPFN doesn't like encoder features | Low | Medium | Match PCA output statistics, normalize |
| Training instable | Low | Low | Use proven architectures, gradient clipping |

### 5.2 Failure Analysis

If the encoder approach fails, we learn:

1. **What failed?**
   - Encoder capacity?
   - Synthetic-real gap?
   - Training objective?

2. **Next steps:**
   - Try supervised encoder training (needs labels)
   - Hybrid: encoder features + PCA features
   - Fine-tune TabPFN directly
   - Full NIRS-PFN (last resort)

---

## 6. Resource Requirements

### 6.1 Compute

| Phase | GPU-Hours | GPU Type | Cost |
|-------|-----------|----------|------|
| Development | 2 | V100/A100 | ~$4 |
| Training | 4-8 | A100 | ~$8-16 |
| Evaluation | 2 | V100 | ~$4 |
| **Total** | ~8-12 | | **~$20** |

### 6.2 Time

| Phase | Days | Cumulative |
|-------|------|------------|
| Baseline | 2 | Day 2 |
| Implementation | 5 | Day 7 |
| Training infra | 3 | Day 10 |
| Training | 4 | Day 14 |
| Evaluation | 7 | Day 21 |
| Integration | 7 | Day 28 |
| **Total** | **28 days** | **4 weeks** |

### 6.3 Personnel

- 1 ML engineer: 4 weeks (part-time OK)
- Domain expert: occasional consultation

---

## 7. Success Criteria Summary

### 7.1 PoC Success (Week 3)

| Criterion | Threshold |
|-----------|-----------|
| RMSE vs baseline | ≤ 1.0× (not worse) |
| At least one dataset improved | Yes |
| Training time | < 8 hours |

### 7.2 Full Success (Week 4)

| Criterion | Threshold |
|-----------|-----------|
| Avg RMSE improvement | > 5% |
| Datasets improved | ≥ 3/4 |
| Few-shot improvement | > 10% |
| Inference overhead | < 50% |

### 7.3 Stretch Goals

| Goal | Threshold |
|------|-----------|
| RMSE improvement | > 15% |
| No ASLSBaseline needed | Encoder handles baselines |
| Cross-instrument transfer | Works without fine-tuning |

---

## 8. Future Directions (Post-PoC)

If the encoder succeeds:

### 8.1 Short-term (1-2 months)

- Optimize encoder for production (ONNX, quantization)
- Train on expanded synthetic prior
- Add more benchmark datasets
- Publish blog post / technical report

### 8.2 Medium-term (3-6 months)

- Fine-tuning capability for domain-specific adaptation
- Multi-source support (NIR + markers)
- Uncertainty-aware encoder
- Integration with calibration transfer

### 8.3 Long-term (6+ months)

- Consider full NIRS-PFN if encoder hits ceiling
- Pre-trained weights for different spectral regions
- Cross-domain transfer (Raman, MIR, UV-Vis)

---

## 9. Comparison: Encoder vs Full PFN

| Aspect | Spectral Encoder | Full NIRS-PFN |
|--------|------------------|---------------|
| **Development time** | 4 weeks | 6-7 months |
| **Compute cost** | ~$20 | ~$1000 |
| **Risk** | Low | High |
| **TabPFN dependency** | Yes | No |
| **Flexibility** | Works with any model | Standalone |
| **Interpretability** | Moderate | Lower |
| **Maintenance** | Low | High |
| **Potential ceiling** | Medium | Higher |

**Conclusion**: Start with encoder. It's the pragmatic choice.

---

## 10. Decision Framework

```
Week 2: Encoder trained
         │
         ├─ Loss converging? ─── No ──→ Debug training
         │       │
         │      Yes
         │       ↓
Week 3: Evaluate on 1 real dataset
         │
         ├─ Competitive with baseline? ─── No ──→ Analyze gap
         │       │                                    │
         │      Yes                                   ├─ Synthetic gap? → Expand prior
         │       ↓                                    ├─ Architecture? → Try other version
         │                                            └─ Fundamental? → Abandon or pivot
Week 4: Full evaluation
         │
         ├─ >5% improvement? ─── No ──→ Iterate one more week
         │       │                           │
         │      Yes                          └─ Still no? → Ship as alternative, not default
         │       ↓
         │
       Ship it!
         │
         ├─ Integrate into nirs4all
         ├─ Publish pretrained weights
         └─ Document findings
```

---

*NIRS Spectral Encoder Development Plan v2.0*
*Revised strategy: Encoder-first, pragmatic approach*
*January 2026*
