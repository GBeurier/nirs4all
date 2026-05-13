"""Smoke experiment for Phase A2 prior-to-dataset generation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from nirsyntheticpfn.adapters.builder_adapter import (
    PriorDatasetAdapterError,
    SyntheticDatasetRun,
    build_synthetic_dataset_run,
)
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config

DEFAULT_OUTPUT = Path("bench/nirs_synthetic_pfn/reports/prior_to_dataset_smoke.md")
PRESETS = [
    ("grain", "regression", 1),
    ("forage", "classification", 3),
    ("oilseeds", "regression", 1),
    ("fruit", "classification", 2),
    ("dairy", "regression", 1),
    ("meat", "classification", 3),
    ("baking", "regression", 1),
    ("tablets", "classification", 4),
    ("powders", "regression", 1),
    ("fuel", "classification", 2),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = run_smoke(n_samples=args.n_samples, seed=args.seed)
    git_status = _git_status_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(results, args.n_samples, args.seed, git_status),
        encoding="utf-8",
    )
    print(args.output)


def run_smoke(*, n_samples: int, seed: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, preset in enumerate(PRESETS):
        domain_alias, target_type, target_size = preset
        run_seed = seed + idx
        source = _preset_source(
            domain_alias,
            target_type=target_type,
            target_size=target_size,
            seed=run_seed,
        )
        try:
            record = canonicalize_prior_config(source)
            run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=run_seed)
            results.append(_success_result(domain_alias, run))
        except PriorDatasetAdapterError as exc:
            results.append({
                "preset": domain_alias,
                "status": "failed",
                "validation_summary": exc.validation_summary,
            })
        except Exception as exc:
            results.append({
                "preset": domain_alias,
                "status": "failed",
                "validation_summary": {
                    "status": "failed",
                    "failures": [
                        {
                            "reason": "experiment_exception",
                            "field": "experiment",
                            "message": str(exc),
                        }
                    ],
                },
            })
    return results


def render_markdown(
    results: list[dict[str, Any]],
    n_samples: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py "
        f"--n-samples {n_samples} --seed {seed}"
    )
    passed = sum(1 for result in results if result["status"] == "passed")
    lines = [
        "# Prior-to-Dataset Smoke",
        "",
        "## Objective",
        "",
        "Generate 10 finite synthetic datasets from 10 canonical A1-style records.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Summary",
        "",
        f"- Seed base: {seed}",
        f"- Samples per dataset: {n_samples}",
        f"- Passed: {passed} / {len(results)}",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Dataset Summary",
        "",
        "| preset | domain | instrument | mode | target | X shape | y shape | y min | y max | status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if result["status"] != "passed":
            failures = result["validation_summary"].get("failures", [])
            lines.append(
                f"| `{result['preset']}` | _failed_ | _failed_ | _failed_ | _failed_ | "
                f"_failed_ | _failed_ | _failed_ | _failed_ | `{failures}` |"
            )
            continue
        summary = result["validation_summary"]["summary"]
        lines.append(
            f"| `{result['preset']}` | `{result['domain']}` | `{result['instrument']}` | "
            f"`{result['mode']}` | `{result['target_type']}` | "
            f"`{summary['X_shape']}` | `{summary['y_shape']}` | "
            f"{summary['y_min']:.4g} | {summary['y_max']:.4g} | `{result['status']}` |"
        )

    lines.extend([
        "",
        "## Contract Checks",
        "",
        "| preset | shape | finite | wavelengths | target | concentrations | seed |",
        "|---|---|---|---|---|---|---:|",
    ])
    for result in results:
        checks = result["validation_summary"].get("checks", {})
        lines.append(
            f"| `{result['preset']}` | `{checks.get('shape')}` | `{checks.get('finite')}` | "
            f"`{checks.get('wavelengths_monotonic')}` | `{checks.get('target_contract')}` | "
            f"`{checks.get('concentrations_row_normalized')}` | "
            f"{checks.get('seed', '')} |"
        )

    lines.extend([
        "",
        "## Unsupported Fields",
        "",
        "None for this smoke set. A2 maps target, row-normalized concentration mixtures, temperature, particle-size scatter, edge roll-off, batch, instrument, and mode fields explicitly.",
        "",
        "Note: `measurement_mode` is passed to `SyntheticNIRSGenerator` and preserved in metadata; this smoke validates the executable dataset contract, not mode-specific optical physics.",
        "",
        "## Residual Risks",
        "",
        "- `measurement_mode` is passed through and preserved, but A2 does not validate mode-specific optical physics.",
        "- Concentrations are row-normalized and should be interpreted as normalized latent fractions, not raw domain-prior magnitudes.",
        "- Smoke presets are curated; the repaired-prior sweep reduces risk but does not replace B1 prior predictive checks.",
        "- A2 target mapping is smoke-level; B1/B2 must validate target distributions and realism.",
        "",
        "## Provenance",
        "",
        "Each run metadata stores `prior_config`, `builder_config`, `validation_summary`, and A1 provenance fields `source_prior_config`, `_raw_prior_config`, and `_canonical_repairs` when present.",
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(_to_builtin({"results": results, "git_status": git_status}), indent=2, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        "Pass A2 smoke gate." if passed == len(results) else "Needs adapter fixes before A2 smoke gate.",
        "",
    ])
    return "\n".join(lines)


def _success_result(preset: str, run: SyntheticDatasetRun) -> dict[str, Any]:
    return {
        "preset": preset,
        "status": run.validation_summary["status"],
        "domain": run.metadata["domain"]["key"],
        "instrument": run.metadata["instrument"]["key"],
        "mode": run.metadata["mode"],
        "target_type": run.metadata["target"]["type"],
        "component_keys": run.metadata["builder_config"]["features"]["components"],
        "validation_summary": run.validation_summary,
        "metadata_keys": sorted(run.metadata),
        "provenance_a1": run.metadata["provenance_a1"],
    }


def _preset_source(
    domain_alias: str,
    *,
    target_type: str,
    target_size: int,
    seed: int,
) -> dict[str, object]:
    domain_key = canonicalize_domain(domain_alias)
    components = _first_valid_domain_components(domain_key, max(3, target_size))
    target_config: dict[str, object]
    if target_type == "classification":
        target_config = {
            "type": "classification",
            "n_classes": target_size,
            "separation": "moderate",
        }
    else:
        target_config = {
            "type": "regression",
            "n_targets": target_size,
            "nonlinearity": "none",
        }
    return {
        "domain": domain_alias,
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (400, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": components,
        "n_samples": 100,
        "target_config": target_config,
        "random_state": seed,
    }


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    components = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n_components:
            return components
    raise ValueError(f"Not enough executable components for {domain_key}")


def _git_status_summary() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    by_status: dict[str, int] = {}
    for line in lines:
        status = line[:2].strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "returncode": result.returncode,
        "entry_count": len(lines),
        "by_status": dict(sorted(by_status.items())),
        "sample": lines[:20],
    }


def _git_status_section(summary: dict[str, Any]) -> str:
    if not summary:
        return "_Not captured_"
    lines = [
        f"- Return code: {summary.get('returncode')}",
        f"- Entries: {summary.get('entry_count')}",
        f"- Status counts: `{summary.get('by_status', {})}`",
    ]
    sample = summary.get("sample") or []
    if sample:
        lines.append("- Sample:")
        lines.extend(f"  - `{line}`" for line in sample)
    return "\n".join(lines)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
