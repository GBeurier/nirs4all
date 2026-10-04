"""Independent test-only PLS-logistic/grouped OOF oracle.

The baseline-class multinomial Newton reference is preserved from the public
Methods parity suites.py oracle. SIMPLS uses the independent NumPy reference,
adapted to the registered Role
head's centered/unscaled profile; source encoders use sklearn. No native
state, controller result or fitted Methods model supplies oracle predictions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def _softmax_baseline_logits(design: np.ndarray,
                             beta: np.ndarray,
                             n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    logits_tail = design @ beta.T
    logits = np.column_stack([
        np.zeros(design.shape[0], dtype=np.float64),
        logits_tail,
    ])
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    if probabilities.shape[1] != int(n_classes):
        raise RuntimeError("probability shape mismatch")
    return logits, probabilities


def _baseline_logistic_objective(design: np.ndarray,
                                 labels: np.ndarray,
                                 beta: np.ndarray,
                                 ridge: float,
                                 n_classes: int) -> float:
    logits, _probabilities = _softmax_baseline_logits(design, beta, n_classes)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_den = np.max(logits, axis=1) + np.log(np.sum(np.exp(shifted), axis=1))
    chosen = np.zeros(labels.size, dtype=np.float64)
    non_baseline = labels > 0
    chosen[non_baseline] = logits[np.arange(labels.size)[non_baseline], labels[non_baseline]]
    penalty = 0.5 * float(ridge) * float(np.sum(beta[:, 1:] * beta[:, 1:]))
    return float(np.sum(log_den - chosen) + penalty)


def _fit_baseline_multinomial_logistic(
    scores: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    ridge: float = 1e-4,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    design = np.column_stack([np.ones(scores.shape[0], dtype=np.float64), scores])
    n, d = design.shape
    c = int(n_classes)
    m = c - 1
    beta = np.zeros((m, d), dtype=np.float64)

    for _iteration in range(int(max_iter)):
        _logits, probabilities = _softmax_baseline_logits(design, beta, c)
        gradient = np.zeros((m, d), dtype=np.float64)
        hessian = np.zeros((m * d, m * d), dtype=np.float64)
        for row in range(n):
            z = design[row, :]
            zz = np.outer(z, z)
            for cls_j in range(m):
                pj = probabilities[row, cls_j + 1]
                yj = 1.0 if int(y[row]) == cls_j + 1 else 0.0
                gradient[cls_j, :] += (pj - yj) * z
                for cls_l in range(m):
                    pl = probabilities[row, cls_l + 1]
                    weight = pj * ((1.0 if cls_j == cls_l else 0.0) - pl)
                    start_j = cls_j * d
                    start_l = cls_l * d
                    hessian[start_j:start_j + d, start_l:start_l + d] += weight * zz

        for cls_j in range(m):
            for feature in range(1, d):
                offset = cls_j * d + feature
                gradient[cls_j, feature] += float(ridge) * beta[cls_j, feature]
                hessian[offset, offset] += float(ridge)

        step = np.linalg.solve(hessian, gradient.reshape(-1, order="C")).reshape(m, d)
        max_step = float(np.max(np.abs(step)))
        current = _baseline_logistic_objective(design, y, beta, ridge, c)
        alpha = 1.0
        accepted = False
        candidate = beta.copy()
        for _line_search in range(32):
            candidate = beta - alpha * step
            trial = _baseline_logistic_objective(design, y, candidate, ridge, c)
            if math.isfinite(trial) and (trial <= current or alpha * max_step <= tol):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise RuntimeError("baseline multinomial logistic line search failed")
        beta = candidate
        if alpha * max_step <= tol:
            break

    logits, probabilities = _softmax_baseline_logits(design, beta, c)
    predictions = np.argmax(logits, axis=1).astype(np.int32)
    return beta[:, 0], beta[:, 1:], logits, probabilities, predictions


def head_probabilities(x: np.ndarray, y: np.ndarray, prediction: np.ndarray, *, n_classes: int, n_components: int, max_iter: int = 500) -> np.ndarray:
    dummy = np.eye(n_classes)[y.astype(np.int64)]
    mean = x.mean(axis=0)
    centered = x - mean
    centered_y = dummy - dummy.mean(axis=0)
    covariance = centered.T @ centered_y
    weights = np.zeros((x.shape[1], n_components))
    loadings = np.zeros_like(weights)
    bases = np.zeros_like(weights)
    for component in range(n_components):
        # Public Methods parity's NumPy SIMPLS reference, with unity scales.
        direction = np.linalg.svd(covariance, full_matrices=False)[0][:, 0]
        score = centered @ direction
        norm = np.linalg.norm(score)
        assert norm > np.finfo(float).eps
        score, direction = score / norm, direction / norm
        loading = centered.T @ score
        basis = loading - bases[:, :component] @ (bases[:, :component].T @ loading)
        assert np.linalg.norm(basis) > np.finfo(float).eps
        basis /= np.linalg.norm(basis)
        weights[:, component], loadings[:, component], bases[:, component] = direction, loading, basis
        covariance -= np.outer(basis, basis @ covariance)
    rotations = weights @ np.linalg.inv(loadings.T @ weights)
    scores = centered @ rotations
    intercept, coefficients, _logits, _probabilities, _predictions = _fit_baseline_multinomial_logistic(
        scores, y, n_classes, max_iter=max_iter)
    design = np.column_stack([np.ones(len(prediction)), (prediction - mean) @ rotations])
    _logits, probabilities = _softmax_baseline_logits(design, np.column_stack([intercept, coefficients]), n_classes)
    return probabilities


def branch_probabilities(model: Any, cohort: Any, train: np.ndarray, prediction: Any, rows: np.ndarray,
                         codes: np.ndarray, n_classes: int, components: int) -> np.ndarray:
    fit_blocks, prediction_blocks = [], []
    for name, transformer in model.transformers.items():
        encoder = clone(transformer).fit(cohort.sources[name].values[train], codes[train])
        weight = (model.source_weights or {}).get(name, 1.0)
        fit_blocks.append(np.asarray(encoder.transform(cohort.sources[name].values[train]), dtype=float) * weight)
        prediction_blocks.append(np.asarray(encoder.transform(prediction.sources[name].values[rows]), dtype=float) * weight)
    return head_probabilities(np.concatenate(fit_blocks, axis=1), codes[train], np.concatenate(prediction_blocks, axis=1),
                              n_classes=n_classes, n_components=components, max_iter=model.model.max_iter)


def topology_probabilities(sequence: list[Any], cohort: Any, train: np.ndarray, prediction: Any, rows: np.ndarray,
                           params: dict[str, Any], vocabulary: dict[str, Any]) -> np.ndarray:
    names = vocabulary["label_names"]
    index = {name: i for i, name in enumerate(names)}
    codes = np.asarray([index[label] for label in np.asarray(cohort.y).reshape(-1)])
    if len(sequence) == 1:
        return branch_probabilities(sequence[0]["model"], cohort, train, prediction, rows, codes, len(names), params["early.n_components"])
    branches = sequence[0]["branch"]
    oof = np.empty((len(train), len(branches) * len(names)))
    projected = []
    folds = list(GroupKFold(2).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
    for branch_index, (name, steps) in enumerate(branches.items()):
        model = steps[0]["model"]
        columns = slice(branch_index * len(names), (branch_index + 1) * len(names))
        for inner_train, inner_validation in folds:
            assert not set(np.asarray(cohort.groups)[train[inner_train]]).intersection(np.asarray(cohort.groups)[train[inner_validation]])
            oof[inner_validation, columns] = branch_probabilities(model, cohort, train[inner_train], cohort, train[inner_validation],
                codes, len(names), params[f"late.{name}.n_components"])
        projected.append(branch_probabilities(model, cohort, train, prediction, rows, codes, len(names), params[f"late.{name}.n_components"]))
    return head_probabilities(oof, codes[train], np.concatenate(projected, axis=1), n_classes=len(names),
                              n_components=params["late.meta.n_components"], max_iter=sequence[2]["model"].max_iter)
