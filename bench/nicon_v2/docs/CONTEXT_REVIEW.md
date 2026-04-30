# nicon_v2 — Context Review

A focused synthesis of (i) the existing `nicon`/`decon` implementations, (ii) the TabPFN-paper bench in `bench/nirs_synthetic_pfn/`, (iii) the AOM-PLS / AOM-Ridge bench in `bench/AOM_v0/`, and (iv) the academic literature compiled in `source_materials/literature_review/`. This is the document the agent reads first when starting a new iteration.

## 1. The existing CNNs

* **`nicon`.** Shallow 3-conv-layer 1-D CNN. Aggressive stride-based downsampling (5, 3, 3) with broad kernels (15, 21, 5). Mixed activations (SELU → ReLU → ELU) and BatchNorm in convs 2-3. Dense head `Dense(16, sigmoid) → Dense(1, sigmoid)` (regression) or `Dense(num_classes, softmax)` (classification).
* **`decon`.** Six-block depthwise-separable stack with channel multiplication (×2 per block: 1 → 64), three MaxPool1D(2) stages, final SeparableConv + Conv1D + MaxPool(5,3) + Dense(128) → Dense(32) → Dense(1, sigmoid).
* **`nicon_VG`.** VGG-style variant with same-padding 3-kernel convs and larger dense head (Dense(1024)·2). Most expensive of the three.
* **Frameworks.** TF/Keras, PyTorch, JAX with `@framework(...)` decorators. JAX implementation also defines a TransformerBlock used by `transformer_VG`.

The chronic failure modes (sigmoid output saturation, mixed-activation pipeline, large-kernel stride-based downsampling, lack of concat-derivatives or augmentation, no UQ) are documented in `WEAKNESS_ANALYSIS.md`.

## 2. The TabPFN-paper bench (bench/nirs_synthetic_pfn/)

* Status: **NO-GO** for nirs4all integration as of last run (`reports/integration_gate_status.md`). Realism gate fails for 23/71 raw rows; transfer validation blocked.
* Datasets used: 77 real datasets (61 regression, 16 classification) from `bench/tabpfn_paper/data/`. Names include ALPINE, AMYLOSE, BEEFMARBLING, BEER, BERRY (brix/ph/ta), BISCUIT (Fat/Sucrose), COLZA, CORN (Oil/Starch), DIESEL, DarkResp, ECOSIS_LeafTraits, FUSARIUM, GRAPEVINE, IncombustibleMaterial, LUCAS, MALARIA, MANURE21, MILK, PEACH, PHOSPHORUS, TABLET, WOOD, etc.
* Evaluation models in the *transfer* stage: only Ridge, PLS, PCA+Ridge.
* No CNN, Transformer, nicon, decon are run in this bench at all — that is the gap nicon_v2 fills.
* Key reports:
  * `reports/integration_gate_status.md`
  * `reports/real_synthetic_scorecards.md`
  * `reports/transfer_validation.md`
  * `reports/adversarial_auc.md`

We do **not** depend on the (currently failing) realism gate for the nicon_v2 benchmark; we run nicon_v2 on the *real* datasets directly.

## 3. The AOM-PLS / AOM-Ridge bench (bench/AOM_v0/)

* **AOM-PLS** (`bench/AOM_v0/aompls/`): comprehensive `LinearSpectralOperator` framework with operator banks (compact 9, default 77, extended 82, family-pruned 15), NIPALS/SIMPLS engines, selection policies (none / global / per_component / soft / superblock / active_superblock), torch backend.
* **AOM-Ridge** (`bench/AOM_v0/Ridge/aomridge/`): Dual-Ridge (RKHS) framework with strict-linear kernels per operator, MKL aggregation, fold-local CV, SPXYFold + RepeatedSPXYFold, branch-selecting outer wrapper. **84 % wins vs paper Ridge** on the curated 39-dataset cohort.
* Documentation skeleton (mirrored in nicon_v2):
  * `IMPLEMENTATION_PLAN.md` (phases)
  * `IMPLEMENTATION_LOG.md` (append-only)
  * `BENCHMARK_PROTOCOL.md`
  * `MATH_SPEC.md`
  * `codex_review_prompts/{math,code,test,publication}_review.md`
  * `publication/{manuscript,figures,tables,scripts,journal,arxiv,supplement}/`
* The Codex review loop is documented in `bench/AOM_v0/Ridge/docs/CODEX_REVIEW_WORKFLOW.md` with backlog files (`CODEX_BACKLOG_*.md`) per round.

## 4. Literature synthesis (`source_materials/literature_review/`)

40 references, 7 themes, 4 gap-analysis entries.

* **Gap G1.** Three CNN failure modes: small-n collapse (Padarian's 10 000-sample heuristic), instrument transfer brittleness (Mishra & Passos 2021 J. Chemom. 35:e3367), preprocessing entanglement (Mishra & Passos 2022 TRAC).
* **Gap G2.** TabPFN advantage at n = 50-1000 with ≤ 500 features — exactly the NIRS regime; no published TabPFN-v2 NIR benchmark.
* **Gap G3.** SOTA recipe = concat-derivatives + EMSC augmentation + 3-5 conv blocks + GAP + AdamW + spatial dropout.
* **Gap G4.** Eight open research questions; nicon_v2 explicitly targets:
  * RQ1: TabPFN-v2 vs DeepSpectra/CNN on standard NIR.
  * RQ2: Differentiable preprocessing stack (Helin learnable EMSC + learnable SG).
  * RQ3: AOM-style stacking with a CNN expert.
  * RQ4: C-Mixup for NIRS.

## 5. The benchmark target

The strongest existing baseline (per `bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv`) is the AOM-Ridge `branch_global` variant. Beating it on ≥ 50 % of the curated 39-dataset cohort with paired Wilcoxon p < 0.05 is the stopping criterion (cf. `Prompt.md`, `BENCHMARK_PROTOCOL.md`).

## 6. Reproducibility hardware

WSL2 / Linux 6.6 / Python 3.13 / 12 physical cores / 24 logical cores / 62.7 GB RAM. **NVIDIA RTX 4090 (CUDA 8.9, 24 GB VRAM, 20 GB free)** — sufficient for all planned phases including 5-member deep ensembles trained in parallel.

## 7. Path conventions

* outer workspace root = `/home/delete/nirs4all/`
* nirs4all package root (where `bench/`, `nirs4all/`, `pyproject.toml` live) = `/home/delete/nirs4all/nirs4all/`
* AOM-PLS bench = `/home/delete/nirs4all/nirs4all/bench/AOM_v0/`
* AOM-Ridge bench = `/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/`
* **nicon_v2 bench (this project) = `/home/delete/nirs4all/nirs4all/bench/nicon_v2/`** (next to AOM_v0, after the 2026-04-30 move).

`nicon_v2.datasets.NIRS4ALL_PKG_ROOT = parents[3]` of `nicon_v2/datasets.py` resolves to the nirs4all package root. The cohort manifest CSV and the AOM-Ridge curated results CSV are derived from `NIRS4ALL_PKG_ROOT`, so any future relocation is a single-file edit.

Recommended PYTHONPATH for benchmarks and tests (cwd = nirs4all package root):

```
PYTHONPATH=bench/nicon_v2:bench/AOM_v0:bench/AOM_v0/Ridge
```

(All three directories must be on the path because `aomridge` imports from `aompls`.)
