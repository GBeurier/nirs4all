# Benchmark and Research Tools

This directory contains research and benchmarking tools for NIRS data analysis studies.

## Directory Structure

```
bench/
├── preprocessing_selection/    # Preprocessing technique selection framework
│   ├── run_pp_selection.py    # Main runner script
│   ├── selector.py            # PreprocessingSelector orchestrator
│   ├── metrics.py             # Unsupervised and supervised metrics
│   ├── proxy_models.py        # Fast proxy models (Ridge, KNN)
│   ├── combinations.py        # Combination analysis (MI, Grassmann)
│   ├── systematic_selection.py # Systematic selection pipeline
│   ├── example_*.py           # Example scripts
│   ├── systematic/            # Modular systematic selection components
│   └── README_pp_selection.md # Detailed documentation
│
├── splitter_selection/         # Train/test splitting strategy comparison
│   ├── run_splitter_selection*.py  # Runner scripts (base, enhanced, classification)
│   ├── splitter_strategies.py      # 16+ splitting algorithms
│   ├── splitter_evaluation*.py     # Evaluation modules
│   ├── splitter_visualization*.py  # Visualization modules
│   ├── unsupervised_splitters.py   # Chemometrics splitters (Puchwein, Duplex, etc.)
│   ├── Methodology_split.md        # Detailed methodology documentation
│   └── NB1_*.ipynb                 # Jupyter notebook for analysis
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

### Preprocessing Selection

Find optimal preprocessing techniques before training:

```bash
cd bench/preprocessing_selection
python run_pp_selection.py --depth 3 --plots
```

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

### Preprocessing Selection
- 4-stage evaluation (unsupervised → diversity → proxy models → augmentation)
- 16 preprocessing techniques evaluated
- Reduces search space by 10-20× while maintaining performance
- See [README_pp_selection.md](preprocessing_selection/README_pp_selection.md) for details

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
