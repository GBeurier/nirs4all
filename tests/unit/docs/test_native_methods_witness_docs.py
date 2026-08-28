"""Documentation checks for the strict Methods live-execution evidence."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPOSITORY_ROOT / "docs" / "source"
PUBLIC_INTERFACES = DOCS_ROOT / "reference" / "public_interfaces.md"
MODULE_API = DOCS_ROOT / "api" / "module_api.md"
R2_AUDIT = DOCS_ROOT / "developer" / "r2_native_recovery_audit.md"
CHANGELOG = REPOSITORY_ROOT / "CHANGELOG.md"
COMPATIBILITY = REPOSITORY_ROOT / "docs" / "compatibility.md"
DOCUMENTED_LEDGER = REPOSITORY_ROOT / "docs" / "compatibility.json"
PACKAGED_LEDGER = REPOSITORY_ROOT / "nirs4all" / "compatibility_ledger.json"


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_native_methods_live_witness_is_documented_as_local_and_non_durable() -> None:
    """Public native docs distinguish the live observation from durable evidence."""

    public_interfaces = _normalized(PUBLIC_INTERFACES.read_text(encoding="utf-8"))
    module_api = _normalized(MODULE_API.read_text(encoding="utf-8"))

    required_public_interfaces = (
        "`native_execution_claim` and `native_execution_is_live`",
        "audit-only, process-local observation",
        "not a portable attestation",
        "`result.close()` or `result.detach()`",
        "`NativeMethodsSession.close()`",
        '`result.export("model.n4a")` may still write the Core Archive V2',
    )
    required_module_api = (
        "`result.native_execution_claim`",
        "process-local rather than a portable receipt",
        "`result.close()`, `result.detach()`",
        '`result.export("model.n4a")` remains valid',
    )

    missing_public_interfaces = [phrase for phrase in required_public_interfaces if phrase not in public_interfaces]
    missing_module_api = [phrase for phrase in required_module_api if phrase not in module_api]
    assert not missing_public_interfaces, "public Interfaces missing:\n" + "\n".join(missing_public_interfaces)
    assert not missing_module_api, "module API missing:\n" + "\n".join(missing_module_api)


def test_installed_methods_evidence_and_r2_release_record_are_published() -> None:
    """The documented CI proof preserves its narrow R2/default boundary."""

    compatibility = _normalized(COMPATIBILITY.read_text(encoding="utf-8"))
    changelog = _normalized(CHANGELOG.read_text(encoding="utf-8"))
    r2_audit = _normalized(R2_AUDIT.read_text(encoding="utf-8"))

    assert "not pinned in CI" not in compatibility
    assert "`methods-installed.yml` pins released `dag-ml==0.3.20` and `nirs4all-methods==1.0.13`" in compatibility
    assert "`test_native_methods_witness.py`" in compatibility
    assert "| → fall back to legacy (`EXPECTED_FALLBACK`) | **9**" in compatibility
    assert "| → semantic preflight refusal (`EXPECTED_PREFLIGHT_REFUSAL`) | **1**" in compatibility
    assert "| → run native on dag-ml | **85**" in compatibility

    assert "process-local live Methods execution observation" in changelog
    assert "`dag-ml==0.3.20` and `nirs4all-methods==1.0.13`" in changelog
    assert "The public default engine remains `legacy`" in changelog

    assert "Le lot #122 (merge `0f612509`)" in r2_audit
    assert "À l'instant historique du lot #122" in r2_audit
    assert "`dag-ml==0.3.19` et `nirs4all-methods==1.0.13`" in r2_audit
    assert "ne rend pas R2 complet" in r2_audit
    assert "ne change pas le défaut `legacy`" in r2_audit


def test_documented_compatibility_counts_follow_the_packaged_ledger() -> None:
    """The prose and byte-identical compatibility resources keep their scopes separate."""

    assert DOCUMENTED_LEDGER.read_bytes() == PACKAGED_LEDGER.read_bytes()
    ledger = json.loads(PACKAGED_LEDGER.read_text(encoding="utf-8"))
    assert ledger["coverage_meter"]["fallback"] == 9
    assert ledger["coverage_meter"]["preflight_refusal"] == 1
    assert ledger["coverage_meter"]["native"] == 85
    methods_installed = next(entry for entry in ledger["cross_engine_surfaces"] if entry["surface"] == "methods_installed")
    assert methods_installed["status"] == "partial"
