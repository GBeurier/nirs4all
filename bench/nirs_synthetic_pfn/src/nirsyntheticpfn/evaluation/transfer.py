"""Phase B3 transfer-validation smoke metrics."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE
from nirsyntheticpfn.evaluation.realism import RealDataset, align_to_real_grid, load_real_spectra

TransferStatus = Literal["completed", "blocked", "unsupported"]

B2_REALISM_RISK = "B2_realism_failed"
REAL_ONLY_CLAIM_STATUS = "not_applicable_real_only"
SYNTHETIC_DIAGNOSTIC_STATUS = "diagnostic_only"
SYNTHETIC_DIAGNOSTIC_BLOCKED_BY_B2_STATUS = "diagnostic_only_blocked_by_B2_realism_failed"


@dataclass(frozen=True)
class RegressionCohort:
    """Loaded real regression cohort for bounded repeated splits."""

    dataset: RealDataset
    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.dataset.database_name}/{self.dataset.dataset}"


@dataclass(frozen=True)
class TransferRow:
    """One B3 transfer-validation CSV/report row."""

    status: TransferStatus
    source: str
    task: str
    dataset: str
    split_index: int | None
    route: str
    model: str
    n_train: int
    n_test: int
    n_features: int
    n_synthetic: int
    synthetic_preset: str
    rmse: float | None
    mae: float | None
    r2: float | None
    phase_a_gate_override: str
    b1_downstream_training_status: str
    b2_realism_risk: str
    synthetic_transfer_claim_status: str
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


def load_regression_cohort(dataset: RealDataset, *, root: Path) -> RegressionCohort:
    """Load train+test spectra and labels from one local regression row."""
    if dataset.task != "regression":
        raise ValueError(f"B3 smoke only supports regression rows, got {dataset.task!r}")
    X, wavelengths = load_real_spectra(dataset, root=root)
    y_train = _load_target_vector(root / dataset.ytrain_path)
    y_test = _load_target_vector(root / dataset.ytest_path)
    y = np.concatenate([y_train, y_test])
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"feature/target row mismatch for {dataset.key}: {X.shape[0]} != {y.shape[0]}")
    if X.shape[0] < 6:
        raise ValueError(f"too few rows for repeated regression splits: {X.shape[0]}")
    return RegressionCohort(dataset=dataset, X=X, y=y, wavelengths=wavelengths)


def downsample_regression_cohort(
    cohort: RegressionCohort,
    *,
    max_samples: int,
    random_state: int,
) -> RegressionCohort:
    """Deterministically cap real rows for bounded B3 runtime."""
    if max_samples <= 0 or cohort.X.shape[0] <= max_samples:
        return cohort
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(cohort.X.shape[0], size=max_samples, replace=False))
    return RegressionCohort(
        dataset=cohort.dataset,
        X=cohort.X[indices],
        y=cohort.y[indices],
        wavelengths=cohort.wavelengths,
    )


def evaluate_regression_transfer_smoke(
    *,
    cohort: RegressionCohort,
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
    synthetic_preset: str,
    n_splits: int,
    test_fraction: float,
    random_state: int,
    b1_downstream_training_status: str,
    b2_realism_failed: bool,
) -> list[TransferRow]:
    """Run real-only baselines and synthetic PCA diagnostics on repeated real splits."""
    _require_sklearn()
    rows: list[TransferRow] = []
    synthetic_risk = B2_REALISM_RISK if b2_realism_failed else "B2_realism_not_failed"
    synthetic_diagnostic_status = (
        SYNTHETIC_DIAGNOSTIC_BLOCKED_BY_B2_STATUS
        if b2_realism_failed
        else SYNTHETIC_DIAGNOSTIC_STATUS
    )
    tstr_blocked_reason = (
        "blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, "
        "not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims."
        if b2_realism_failed
        else (
            "blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, "
            "not calibrated to the selected real analyte labels."
        )
    )
    for split_index in range(n_splits):
        train_idx, test_idx = _split_indices(
            n_rows=cohort.X.shape[0],
            test_fraction=test_fraction,
            random_state=random_state + split_index,
        )
        X_train = cohort.X[train_idx]
        X_test = cohort.X[test_idx]
        y_train = cohort.y[train_idx]
        y_test = cohort.y[test_idx]
        rows.extend([
            _fit_predict_row(
                cohort=cohort,
                split_index=split_index,
                route="real_only",
                model="ridge",
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                synthetic_preset="none",
                n_synthetic=0,
                b1_downstream_training_status=b1_downstream_training_status,
                b2_realism_risk=synthetic_risk,
                synthetic_transfer_claim_status=REAL_ONLY_CLAIM_STATUS,
            ),
            _fit_predict_row(
                cohort=cohort,
                split_index=split_index,
                route="real_only",
                model="pls",
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                synthetic_preset="none",
                n_synthetic=0,
                b1_downstream_training_status=b1_downstream_training_status,
                b2_realism_risk=synthetic_risk,
                synthetic_transfer_claim_status=REAL_ONLY_CLAIM_STATUS,
            ),
            _fit_predict_row(
                cohort=cohort,
                split_index=split_index,
                route="real_only",
                model="pca_ridge",
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                synthetic_preset="none",
                n_synthetic=0,
                b1_downstream_training_status=b1_downstream_training_status,
                b2_realism_risk=synthetic_risk,
                synthetic_transfer_claim_status=REAL_ONLY_CLAIM_STATUS,
            ),
        ])
        rows.append(_synthetic_pca_row(
            cohort=cohort,
            split_index=split_index,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            synthetic_X=synthetic_X,
            synthetic_wavelengths=synthetic_wavelengths,
            synthetic_preset=synthetic_preset,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=synthetic_risk,
            synthetic_transfer_claim_status=synthetic_diagnostic_status,
        ))
        rows.append(blocked_tstr_row(
            cohort=cohort,
            split_index=split_index,
            synthetic_preset=synthetic_preset,
            n_synthetic=int(synthetic_X.shape[0]),
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=synthetic_risk,
            blocked_reason=tstr_blocked_reason,
        ))
    return rows


def blocked_tstr_row(
    *,
    cohort: RegressionCohort,
    split_index: int | None,
    synthetic_preset: str,
    n_synthetic: int,
    b1_downstream_training_status: str,
    b2_realism_risk: str,
    blocked_reason: str,
) -> TransferRow:
    """Represent supervised TSTR as blocked when labels/domains are not aligned."""
    return TransferRow(
        status="blocked",
        source=cohort.dataset.source,
        task=cohort.dataset.task,
        dataset=cohort.name,
        split_index=split_index,
        route="TSTR",
        model="synthetic_only_supervised",
        n_train=0,
        n_test=0,
        n_features=int(cohort.X.shape[1]),
        n_synthetic=n_synthetic,
        synthetic_preset=synthetic_preset,
        rmse=None,
        mae=None,
        r2=None,
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        b1_downstream_training_status=b1_downstream_training_status,
        b2_realism_risk=b2_realism_risk,
        synthetic_transfer_claim_status="blocked",
        blocked_reason=blocked_reason,
    )


def write_transfer_csv(rows: list[TransferRow], path: Path) -> None:
    """Write B3 transfer-validation rows as a flat CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = [row.to_dict() for row in rows]
    fieldnames = list(TransferRow(
        status="blocked",
        source="none",
        task="regression",
        dataset="none",
        split_index=None,
        route="none",
        model="none",
        n_train=0,
        n_test=0,
        n_features=0,
        n_synthetic=0,
        synthetic_preset="none",
        rmse=None,
        mae=None,
        r2=None,
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        b1_downstream_training_status="blocked",
        b2_realism_risk=B2_REALISM_RISK,
        synthetic_transfer_claim_status="blocked",
        blocked_reason="no_rows",
    ).to_dict())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def _fit_predict_row(
    *,
    cohort: RegressionCohort,
    split_index: int,
    route: str,
    model: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    synthetic_preset: str,
    n_synthetic: int,
    b1_downstream_training_status: str,
    b2_realism_risk: str,
    synthetic_transfer_claim_status: str,
) -> TransferRow:
    try:
        estimator = _make_estimator(model, X_train)
        estimator.fit(X_train, y_train)
        y_pred = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
        rmse, mae, r2 = _regression_metrics(y_test, y_pred)
        return _completed_row(
            cohort=cohort,
            split_index=split_index,
            route=route,
            model=model,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            n_features=int(X_train.shape[1]),
            n_synthetic=n_synthetic,
            synthetic_preset=synthetic_preset,
            rmse=rmse,
            mae=mae,
            r2=r2,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=b2_realism_risk,
            synthetic_transfer_claim_status=synthetic_transfer_claim_status,
        )
    except Exception as exc:
        return _blocked_metric_row(
            cohort=cohort,
            split_index=split_index,
            route=route,
            model=model,
            n_features=int(X_train.shape[1]),
            n_synthetic=n_synthetic,
            synthetic_preset=synthetic_preset,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=b2_realism_risk,
            synthetic_transfer_claim_status=synthetic_transfer_claim_status,
            blocked_reason=f"{exc.__class__.__name__}: {exc}",
        )


def _synthetic_pca_row(
    *,
    cohort: RegressionCohort,
    split_index: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
    synthetic_preset: str,
    b1_downstream_training_status: str,
    b2_realism_risk: str,
    synthetic_transfer_claim_status: str,
) -> TransferRow:
    if b1_downstream_training_status != "allowed":
        return _blocked_metric_row(
            cohort=cohort,
            split_index=split_index,
            route="RTSR_diagnostic",
            model="synthetic_pca_ridge",
            n_features=int(cohort.X.shape[1]),
            n_synthetic=int(synthetic_X.shape[0]),
            synthetic_preset=synthetic_preset,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=b2_realism_risk,
            synthetic_transfer_claim_status="blocked",
            blocked_reason="blocked_by_B1_prior_predictive_status",
        )
    try:
        _, synthetic_aligned, target_wavelengths = align_to_real_grid(
            X_train,
            cohort.wavelengths,
            synthetic_X,
            synthetic_wavelengths,
        )
        real_mask = np.isin(cohort.wavelengths, target_wavelengths)
        X_train_aligned = X_train[:, real_mask]
        X_test_aligned = X_test[:, real_mask]
        y_pred = _fit_synthetic_pca_ridge(
            synthetic_X=synthetic_aligned,
            X_train=X_train_aligned,
            y_train=y_train,
            X_test=X_test_aligned,
            random_state=split_index,
        )
        rmse, mae, r2 = _regression_metrics(y_test, y_pred)
        return _completed_row(
            cohort=cohort,
            split_index=split_index,
            route="RTSR_diagnostic",
            model="synthetic_pca_ridge",
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            n_features=int(X_train_aligned.shape[1]),
            n_synthetic=int(synthetic_aligned.shape[0]),
            synthetic_preset=synthetic_preset,
            rmse=rmse,
            mae=mae,
            r2=r2,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=b2_realism_risk,
            synthetic_transfer_claim_status=synthetic_transfer_claim_status,
        )
    except Exception as exc:
        return _blocked_metric_row(
            cohort=cohort,
            split_index=split_index,
            route="RTSR_diagnostic",
            model="synthetic_pca_ridge",
            n_features=int(cohort.X.shape[1]),
            n_synthetic=int(synthetic_X.shape[0]),
            synthetic_preset=synthetic_preset,
            b1_downstream_training_status=b1_downstream_training_status,
            b2_realism_risk=b2_realism_risk,
            synthetic_transfer_claim_status=synthetic_transfer_claim_status,
            blocked_reason=f"{exc.__class__.__name__}: {exc}",
        )


def _fit_synthetic_pca_ridge(
    *,
    synthetic_X: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n_components = _pca_components(synthetic_X, X_train)
    scaler = StandardScaler().fit(synthetic_X)
    pca = PCA(n_components=n_components, random_state=random_state).fit(scaler.transform(synthetic_X))
    X_train_latent = pca.transform(scaler.transform(X_train))
    X_test_latent = pca.transform(scaler.transform(X_test))
    model = Ridge(alpha=1.0)
    model.fit(X_train_latent, y_train)
    return np.asarray(model.predict(X_test_latent), dtype=float).reshape(-1)


def _make_estimator(model: str, X_train: np.ndarray) -> Any:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model == "pls":
        n_components = max(1, min(8, X_train.shape[0] - 1, X_train.shape[1]))
        return make_pipeline(StandardScaler(), PLSRegression(n_components=n_components, scale=False))
    if model == "pca_ridge":
        n_components = max(1, min(12, X_train.shape[0] - 1, X_train.shape[1]))
        return make_pipeline(StandardScaler(), PCA(n_components=n_components), Ridge(alpha=1.0))
    raise ValueError(f"unsupported model {model!r}")


def _completed_row(
    *,
    cohort: RegressionCohort,
    split_index: int,
    route: str,
    model: str,
    n_train: int,
    n_test: int,
    n_features: int,
    n_synthetic: int,
    synthetic_preset: str,
    rmse: float,
    mae: float,
    r2: float,
    b1_downstream_training_status: str,
    b2_realism_risk: str,
    synthetic_transfer_claim_status: str,
) -> TransferRow:
    return TransferRow(
        status="completed",
        source=cohort.dataset.source,
        task=cohort.dataset.task,
        dataset=cohort.name,
        split_index=split_index,
        route=route,
        model=model,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_synthetic=n_synthetic,
        synthetic_preset=synthetic_preset,
        rmse=rmse,
        mae=mae,
        r2=r2,
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        b1_downstream_training_status=b1_downstream_training_status,
        b2_realism_risk=b2_realism_risk,
        synthetic_transfer_claim_status=synthetic_transfer_claim_status,
        blocked_reason="",
    )


def _blocked_metric_row(
    *,
    cohort: RegressionCohort,
    split_index: int,
    route: str,
    model: str,
    n_features: int,
    n_synthetic: int,
    synthetic_preset: str,
    b1_downstream_training_status: str,
    b2_realism_risk: str,
    synthetic_transfer_claim_status: str,
    blocked_reason: str,
) -> TransferRow:
    return TransferRow(
        status="blocked",
        source=cohort.dataset.source,
        task=cohort.dataset.task,
        dataset=cohort.name,
        split_index=split_index,
        route=route,
        model=model,
        n_train=0,
        n_test=0,
        n_features=n_features,
        n_synthetic=n_synthetic,
        synthetic_preset=synthetic_preset,
        rmse=None,
        mae=None,
        r2=None,
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        b1_downstream_training_status=b1_downstream_training_status,
        b2_realism_risk=b2_realism_risk,
        synthetic_transfer_claim_status=synthetic_transfer_claim_status,
        blocked_reason=blocked_reason,
    )


def _split_indices(
    *,
    n_rows: int,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 6:
        raise ValueError(f"need at least 6 rows for split, got {n_rows}")
    test_size = max(2, int(round(n_rows * test_fraction)))
    test_size = min(test_size, n_rows - 2)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_rows)
    test_idx = np.sort(indices[:test_size])
    train_idx = np.sort(indices[test_size:])
    return train_idx, test_idx


def _pca_components(synthetic_X: np.ndarray, X_train: np.ndarray) -> int:
    return int(max(1, min(12, synthetic_X.shape[0] - 1, X_train.shape[0] - 1, synthetic_X.shape[1], X_train.shape[1])))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return (
        float(mean_squared_error(y_true, y_pred) ** 0.5),
        float(mean_absolute_error(y_true, y_pred)),
        float(r2_score(y_true, y_pred)),
    )


def _load_target_vector(path: Path) -> np.ndarray:
    y = np.genfromtxt(path, skip_header=1, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError(f"non-finite or empty target vector in {path}")
    return y


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError("sklearn is required for B3 transfer smoke baselines") from exc


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
