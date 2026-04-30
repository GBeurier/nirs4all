"""Stacking smoke tests on a synthetic linear+nonlinear problem."""

from __future__ import annotations

import numpy as np

from nicon_v2.models.stacking import StackedRegressor, StackingConfig


def _toy_problem(n: int = 100, p: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.3, size=p)
    X = rng.normal(size=(n, p))
    y = X @ w + 0.2 * np.sin(X[:, 0] * 3) + rng.normal(scale=0.1, size=n)
    X_test = rng.normal(size=(n // 2, p))
    y_test = X_test @ w + 0.2 * np.sin(X_test[:, 0] * 3) + rng.normal(scale=0.1, size=n // 2)
    return X, y, X_test, y_test


def test_stacking_ridge_pls_runs_and_predicts():
    X, y, X_test, y_test = _toy_problem(n=80, p=20)
    cfg = StackingConfig(base_learners=("ridge", "pls"), n_folds=3, seed=0)
    model = StackedRegressor(cfg=cfg).fit(X, y)
    pred = model.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
    assert np.isfinite(model.selected_meta_alpha_)
    assert model.selected_meta_alpha_ in cfg.meta_alphas


def test_stacking_beats_naive_mean():
    X, y, X_test, y_test = _toy_problem(n=80, p=20)
    cfg = StackingConfig(base_learners=("ridge", "pls"), n_folds=3, seed=0)
    model = StackedRegressor(cfg=cfg).fit(X, y)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < rmse_naive


def test_stacking_hyperparams_serialisable():
    cfg = StackingConfig(base_learners=("ridge", "pls"), n_folds=3, seed=0)
    model = StackedRegressor(cfg=cfg).fit(*_toy_problem(n=60, p=15)[:2])
    hp = model.hyperparams
    assert hp["model"] == "Stacked"
    assert hp["base_learners"] == ["ridge", "pls"]
    assert hp["selected_meta_alpha"] == model.selected_meta_alpha_
    assert hp["alpha_grid_max"] >= 1e3  # Codex F3 fix


# Codex round 3 finding #4 — fold-isolation spy test.

class _SpyEstimator:
    """Records the indices it was fitted on for downstream isolation assertions."""

    def __init__(self, recorder: dict, key: str):
        self._recorder = recorder
        self._key = key
        self._mean: float = 0.0
        self._fitted_idx: set[int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SpyEstimator":
        idx = set(int(i) for i in y)  # caller encodes the ids in the y vector
        self._mean = float(np.mean(y))
        self._fitted_idx = idx
        # Record (key, len) so the test can verify per-fold cardinality.
        self._recorder.setdefault(self._key, []).append(idx)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean, dtype=float)


def test_stacking_does_not_leak_validation_fold():
    """Each base estimator's fit set in fold k must NOT contain that fold's validation indices."""
    import numpy as np
    rng = np.random.default_rng(0)
    n, p = 30, 8
    X = rng.normal(size=(n, p))
    # Encode the row id in `y`. The fold validation set's row ids must never appear in the
    # spy estimator's fit-time recorder.
    y = np.arange(n, dtype=float)

    recorder: dict = {}

    # Monkey-patch the base-estimator factory just for this test.
    from nicon_v2.models import stacking as st

    real_factory = st._build_base_estimator
    st._build_base_estimator = lambda name, seed, cfg: _SpyEstimator(recorder, key=name)
    try:
        cfg = StackingConfig(base_learners=("ridge", "pls"), n_folds=3, seed=0)
        # Seed-aware splitter inside StackedRegressor.fit
        kf = st.KFold(n_splits=3, shuffle=True, random_state=0)
        fold_ids = list(kf.split(X))
        StackedRegressor(cfg=cfg).fit(X, y)

        # For each fold (tr, va) and each base learner, the recorded OOF fit set
        # (fits 0, 1, 2) must NOT contain that fold's validation indices. The 4th
        # recorded fit is the post-OOF refit on the full training set — allowed.
        for learner_name, fits in recorder.items():
            assert len(fits) == 4, f"{learner_name} should have 3 OOF fits + 1 full refit; got {len(fits)}"
            for k, (tr, va) in enumerate(fold_ids):
                assert fits[k] == set(int(i) for i in tr), (
                    f"fold {k} {learner_name}: spy recorded {fits[k]} expected {set(tr)}"
                )
                assert not (fits[k] & set(int(i) for i in va)), (
                    f"fold {k} {learner_name}: validation index leaked into fit set"
                )
            # 4th fit = full training set.
            assert fits[3] == set(range(n)), f"final refit should see all {n} rows"
    finally:
        st._build_base_estimator = real_factory


def test_stacking_alpha_grid_extends_to_1e4():
    cfg = StackingConfig()
    assert max(cfg.meta_alphas) >= 1e4
