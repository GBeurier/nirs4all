import numpy as np
import polars as pl
from pathlib import Path
from .SpectraDataset import SpectraDataset
from .core.blocks import Block
from .index.frame import IndexFrame
from .targets import TargetTable
from .processing import TransformationPath


def create_sample_dataset() -> SpectraDataset:
    """Crée un dataset d'exemple pour les tests."""
    ds = SpectraDataset()

    # Créer des données factices
    n_samples = 100
    n_features_raman = 256
    n_features_nirs = 144

    # Données Raman (source 0)
    raman_data = np.random.randn(n_samples, n_features_raman).astype(np.float32)
    raman_block = Block(raman_data, source_id=0, processing_id=0)
    ds.store.add_block(raman_block, 0, 0, n_samples)

    # Données NIRS (source 1)
    nirs_data = np.random.randn(n_samples, n_features_nirs).astype(np.float32)
    nirs_block = Block(nirs_data, source_id=1, processing_id=0)
    ds.store.add_block(nirs_block, 1, 0, n_samples)

    # Index DataFrame
    index_data = {
        "row": list(range(n_samples)),
        "sample": list(range(n_samples)),
        "origin": ["original"] * n_samples,
        "partition": ["train"] * 80 + ["test"] * 20,
        "group": np.random.randint(0, 8, n_samples).tolist(),
        "branch": [0] * n_samples,
        "processing": [0] * n_samples
    }
    ds.index = IndexFrame(pl.DataFrame(index_data))

    # Targets
    targets_data = {
        "row": list(range(n_samples)),
        "species": ["coffea-" + str(i % 3) for i in range(n_samples)],
        "species_idx": np.random.randint(0, 3, n_samples).tolist()
    }
    ds.targets = TargetTable(pl.DataFrame(targets_data))

    # Enregistrer le processing "raw"
    ds.processing.get_or_create(
        TransformationPath(),
        n_features_raman,
        np.float32,
        {"name": "raw"}
    )

    return ds


# Créer le répertoire de test et le dataset
def setup_test_data():
    test_dir = Path("../../../data/coffee_spectra")
    test_dir.mkdir(parents=True, exist_ok=True)

    ds = create_sample_dataset()
    ds.save(test_dir, overwrite=True)
    print(f"Sample dataset created at {test_dir}")


if __name__ == "__main__":
    setup_test_data()
