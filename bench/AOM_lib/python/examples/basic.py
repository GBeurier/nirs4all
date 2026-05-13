"""Minimal Python example: fit + predict on synthetic spectra."""

from __future__ import annotations

import numpy as np

from aompls import AOMPLSCompact, tune


def main() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 256
    bumps = 0.5 + 0.1 * (np.arange(n) % 5)
    t = (np.arange(p) - p / 2) / 32.0
    X = bumps[:, None] * np.exp(-(t**2)) + 0.02 * rng.normal(size=(n, p))
    y = bumps.astype(float)

    m = AOMPLSCompact(max_components=10, n_folds=5, preproc="snv").fit(X, y)
    print(f"Selected: {m.selected_operator_name_} (idx {m.selected_operator_index_}), k={m.n_components_}")
    print(f"Training RMSE: {np.sqrt(np.mean((m.predict(X) - y) ** 2)):.4f}")

    # Tiny HPO grid (downscaled for the demo)
    res = tune(
        X, y,
        max_components_grid=(5, 10),
        preproc_grid=("none", "snv"),
        outer_folds=3,
        n_folds=5,
    )
    print(f"Best HPO config: {res.best_params}, RMSE={res.best_score:.4f}")


if __name__ == "__main__":
    main()
