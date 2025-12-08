# Benchmark and Research Tools

This directory contains research and benchmarking tools for NIRS data analysis studies.

## Directory Structure

```
bench/
│
├── synthetic/                  # Synthetic NIRS spectra generation
│   ├── generator.py           # SyntheticNIRSGenerator class
│   ├── comparator.py          # Compare synthetic vs real data
│   ├── visualizer.py          # Visualization tools
│   ├── S1_synthetic_generation.py  # Example script
│   └── charts/                # Generated visualization outputs
│
├── tabpfn/                    # TabPFN integration for NIRS
│   ├── spectral_latent_features.py  # Feature extraction for TabPFN
│   ├── tabpfn_run.py               # TabPFN exploration script
│   ├── tuned_tabpfn.py            # Hyperparameter tuning
│   └── search_space.py            # Search space definitions
│
├── models/                    # Custom neural network architectures
│   ├── custom_nicon.py        # NICON variants
│   ├── custom_NN.py           # Various 1D CNN/MLP models
│   └── full_nicon.py          # Full NICON benchmark
│
├── _datasets/                 # Benchmark datasets (not tracked in git)
├── batch_regression.py        # Batch regression analysis
├── bench_ipls_jax.py         # iPLS NumPy vs JAX benchmark
├── nitro_regression.py       # Nitro dataset regression
├── old_pls.py                # Legacy PLS implementations
├── data_prep.ipynb           # Data preparation notebook
└── predictions.ipynb         # Predictions analysis notebook
```

## Usage

### Splitter Selection

Compare train/test splitting strategies:

```bash
cd bench/splitter_selection
python run_splitter_selection.py --data_dir path/to/data --output_dir results/
```

### Synthetic Data Generation

Generate realistic synthetic NIRS spectra:

```bash
cd bench
python -c "from synthetic import SyntheticNIRSGenerator; g = SyntheticNIRSGenerator(); X, y = g.generate(n_samples=100)"
```

### TabPFN Integration

Use TabPFN with NIRS data:

```bash
cd bench/tabpfn
python tabpfn_run.py --data_dir path/to/data
```

## Key Features

### Splitter Selection
- 16+ splitting strategies (Random, Kennard-Stone, SPXY, Duplex, etc.)
- Multi-model ensemble evaluation
- Bootstrap confidence intervals
- Representativeness metrics
- See [Methodology_split.md](splitter_selection/Methodology_split.md) for details

### Synthetic Generation
- Realistic NIRS spectra with chemical components
- Batch effects and noise modeling
- Useful for data augmentation and testing
