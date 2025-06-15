#!/usr/bin/env python3

# Test simple du chargement
from pathlib import Path
from nirs4all.dataset import SpectroDataset

print("Testing SpectroDataset loading...")

ds = SpectroDataset.load(Path("data/coffee_spectra"))

print(f"Dataset loaded")
print(f"Index: {ds.index}")
print(f"Targets: {ds.targets}")
print(f"Store blocks: {len(ds.store._blocks)}")

if ds.index:
    print(f"Index shape: {ds.index.df.shape}")
    print(f"Index columns: {ds.index.df.columns}")

if ds.targets:
    print(f"Targets shape: {ds.targets.df.shape}")
    print(f"Targets columns: {ds.targets.df.columns}")
