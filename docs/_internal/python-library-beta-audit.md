# nirs4all Python Library Beta Readiness Audit

Audit date: 2026-02-20  
Scope: `/home/delete/nirs4all/nirs4all` (Python library only, excluding webapp)  
Context: audit refresh after lint/type cleanup and recent refactors

---

## 1. Executive Verdict

The library is materially stronger than in the previous audit. Package-level quality gates are now clean (`ruff` and `mypy` on `nirs4all/`), and operational hygiene improved.

It is still **not beta-ready yet** because the remaining blockers are release-surface blockers, not algorithmic blockers:

- Documentation quality is still red at very high volume under strict mode.
- Release workflows do not consistently enforce docs warnings.
- Metadata/legal/version contracts remain contradictory across public artifacts.
- Typed package claim (`Typing :: Typed`) is still not truthful in the wheel.
- Dependency and governance contracts are still incomplete.

### Readiness score (today)

- Scientific capability: **9.0/10**
- API/product scope: **8.6/10**
- Test breadth: **8.5/10**
- CI/release engineering: **7.2/10**
- Documentation reliability: **2.8/10**
- Type/lint gate health: **8.8/10**
- OSS governance maturity: **5.4/10**
- Packaging/metadata correctness: **4.2/10**

**Overall beta readiness: 6.9/10**

---

## 2. Since The Previous Audit

### Major improvements

- Ruff closure reached zero:
  - `ruff check nirs4all` -> **All checks passed**.
  - `ruff check .` -> **All checks passed**.
- MyPy package scope closure reached zero:
  - `mypy nirs4all` -> **Success: no issues found in 434 source files**.
- `type: ignore` debt was fully removed:
  - Package: **9 -> 0**
  - Tests: **6 -> 0**
- `print(...)` usage was fully removed in scanned Python files:
  - Package: **421 -> 0**
  - Tests: **119 -> 0**
- Release rehearsal coverage improved with a dedicated pre-publish workflow including build + twine validation.

### Still critical / new blockers

- Strict docs build remains deeply red:
  - **3846 total problems** (`3546 WARNING` + `300 ERROR`) in strict run.
- Pre-publish type-check command drift:
  - `mypy nirs4all` passes.
  - `mypy .` fails immediately with `fck-pls is not a valid Python package name`.
- License/version/public metadata inconsistencies persist.
- Typed package claim remains inaccurate (`py.typed` absent from source and wheel).

---

## 3. Evidence Snapshot

### 3.1 Codebase footprint

- Package Python files: **434**
- Test Python files: **358**
- Unit test files: **285**
- Integration test files: **65**

### 3.2 Test collection

- `pytest tests --collect-only -q -n 0`: **7025 tests**
- `pytest tests/unit --collect-only -q -n 0`: **6444 tests**
- `pytest tests/integration --collect-only -q -n 0`: **581 tests**

### 3.3 Complexity indicators

- Files over 500 LOC: **150**
- Files over 1000 LOC: **51**
- Files over 1500 LOC: **24**
- Files over 2000 LOC: **13**

Largest files:

- `nirs4all/controllers/data/merge.py` (**5234** LOC)
- `nirs4all/synthesis/fitter.py` (**5203** LOC)
- `nirs4all/synthesis/_constants.py` (**2844** LOC)
- `nirs4all/pipeline/storage/workspace_store.py` (**2707** LOC)
- `nirs4all/data/predictions.py` (**2329** LOC)

### 3.4 Quality gate signals

- `ruff check nirs4all`: **PASSED**
- `ruff check .`: **PASSED**
- `mypy nirs4all`: **PASSED** (`no issues found in 434 source files`)
- `mypy .`: **FAILED** (`fck-pls is not a valid Python package name`)

### 3.5 Reliability/debt indicators

- `print(...)` occurrences:
  - Package: **0**
  - Tests: **0**
- `except Exception` occurrences:
  - Package: **269**
  - Tests: **17**
- bare `except:` occurrences:
  - Package: **0**
  - Tests: **0**
- `type: ignore` occurrences:
  - Package: **0**
  - Tests: **0**
- Test markers:
  - `pytest.mark.skip/skipif` references: **75**
  - `pytest.mark.xfail` references: **0**
  - `xdist_group` references: **19**

### 3.6 Docs/release/package checks

- Strict docs build (`sphinx-build -b html source _build/html -W --keep-going`): **FAILED**
  - Exit status: **1**
  - Warnings: **3546**
  - Errors: **300**
  - Total strict problems: **3846**
- Major docs issue families in strict log:
  - duplicate object descriptions: **3208**
  - docutils issues: **496**
  - missing cross-references (`myst.xref_missing`): **107**
  - toctree not included warnings: **12**
- Build artifacts (`python -m build --no-isolation`): **PASSED**
- Local twine availability (`python -m twine --version`): **FAILED** (module not installed in current local env)
- CLI health checks:
  - `nirs4all --test-install`: **PASSED**
  - `nirs4all --test-integration`: **PASSED**

### 3.7 Contract consistency checks

- Typed contract:
  - `nirs4all/py.typed`: **missing**
  - built wheel contains `nirs4all/py.typed`: **no**
- Legal/version drift:
  - `LICENSE`: default **AGPL-3.0-or-later** (dual-license model)
  - `README.md`: license badge + section claim **CeCILL-2.1**
  - `pyproject.toml`: classifier claims **CeCILL-2.1**
  - `README.md` citation version: **0.8.0**
  - package version (`pyproject.toml`): **0.7.1**
  - `SECURITY.md` support table: **0.8.x supported**
- Dependency drift:
  - In `requirements.txt` but not `project.dependencies`: **ikpls, pyopls, trendfitter**
  - In `project.dependencies` but not `requirements.txt`: **duckdb, umap-learn**
  - Docs extra in `pyproject.toml` does not include docs deps used by docs CI (for example `sphinxcontrib-mermaid`).
- Governance/support maturity:
  - Present: `SECURITY.md`
  - Missing: `SUPPORT.md`, `GOVERNANCE.md`, `.pre-commit-config.yaml`, `.github/CODEOWNERS`, PR template
  - Issue template still contains web/mobile fields (`OS iOS`, browser/smartphone prompts).

---

## 4. What Is Strong

- Core scientific breadth remains excellent.
- Large test inventory remains stable (7025 collected tests).
- Package-level lint and typing baseline is now clean.
- Build pipeline fundamentals are in place (dist build passes; pre-publish includes twine check job).
- CLI health smoke checks pass locally.
- Security disclosure policy exists.

---

## 5. Beta Blockers (Must Resolve Before Public Beta)

### B1. Documentation gate remains red and inconsistently enforced

Current state:

- Strict docs run fails with **3846** problems.
- `.github/workflows/docs.yml` still uses `continue-on-error: true` for docs build and linkcheck.
- Pre-publish/publish docs jobs run without `-W`, so warning-heavy docs can still pass release workflows.

Why this blocks beta:

- Docs are a first-class product surface for scientific users; current quality and gating policy are not release-grade.

Required beta outcome:

- A single enforced docs policy:
  - strict docs checks in release workflows,
  - no fail-open behavior,
  - warning/error budget driven to near-zero.

### B2. Type gate command mismatch in release workflow

Current state:

- `mypy nirs4all` is green.
- `mypy .` fails (`fck-pls is not a valid Python package name`).
- Pre-publish workflow currently uses `mypy .`.

Why this blocks beta:

- The advertised release gate is not reliably executable in current tree layout.

Required beta outcome:

- Make the enforced command deterministic and green:
  - either align to `mypy nirs4all`,
  - or keep `mypy .` and explicitly exclude non-package paths (for example `bench/fck-pls`).

### B3. Legal and version metadata remain contradictory

Current state:

- `LICENSE` default story and README/pyproject license claims conflict.
- README citation/version and SECURITY supported-version matrix are not aligned with package version `0.7.1`.

Why this blocks beta:

- Public legal/version ambiguity is a trust and compliance blocker.

Required beta outcome:

- One coherent, repo-wide legal/version contract across `LICENSE`, README, classifiers, SECURITY, and citation metadata.

### B4. Typed package claim is still inaccurate

Current state:

- `pyproject.toml` claims `Typing :: Typed` and package-data includes `py.typed`.
- `py.typed` is not present in source package and not present in wheel.

Why this blocks beta:

- Published typing contract is currently false.

Required beta outcome:

- Ship `py.typed` correctly, or remove typed claim until ready.

### B5. Dependency contract drift remains

Current state:

- `requirements.txt` and `project.dependencies` are not equivalent.
- Docs dependency source is split (`docs/readthedocs.requirements.txt` vs `project.optional-dependencies.docs`).

Why this blocks beta:

- Different installation paths can produce inconsistent runtime/doc behavior.

Required beta outcome:

- Define one authoritative dependency contract and enforce parity generation/checks.

### B6. OSS governance remains partial

Current state:

- Key maintainer process files are still missing.
- Issue templates remain partially web/mobile oriented rather than scientific reproducibility oriented.

Why this blocks beta:

- Community support and contribution contract is incomplete for public-beta operations.

Required beta outcome:

- Complete support/governance/CODEOWNERS/PR-template and modernize issue templates.

---

## 6. High-Priority Work (Beta Window)

### H1. Stabilize docs and enforce a single release gate

- Burn down duplicate autodoc targets and docutils formatting errors first.
- Convert release docs jobs to strict mode (`-W`) once issue volume is reduced.
- Remove docs fail-open behavior from `docs.yml`.

### H2. Align type-check command with repository structure

- Fix pre-publish mypy target mismatch (`mypy .` vs `mypy nirs4all`).
- Lock the chosen scope in CI and document it.

### H3. Normalize public metadata

- Align license statement across LICENSE/README/classifiers.
- Align version references in README citation and `SECURITY.md` to package metadata.

### H4. Repair typed and dependency contracts

- Add and ship `py.typed` or remove typed claim.
- Remove dependency drift between `pyproject.toml`, `requirements.txt`, and docs deps.

### H5. Finish governance surfaces

- Add `SUPPORT.md`, `GOVERNANCE.md`, `.github/CODEOWNERS`, PR template, and pre-commit config.
- Replace web/mobile bug template fields with scientific reproducibility fields.

---

## 7. Recommended 4-Week Beta Hardening Plan

### Week 1

- Resolve pre-publish mypy scope mismatch.
- Start docs triage on highest-volume classes (duplicate objects + docutils indentation issues).
- Decide canonical license/version source-of-truth.

### Week 2

- Continue docs burn-down and add strict docs dry-run in release rehearsal.
- Implement typed-contract fix (`py.typed` or classifier removal).
- Normalize dependency declarations and add parity checks.

### Week 3

- Remove docs fail-open (`continue-on-error`) and enforce strict docs in release path.
- Add missing governance/support templates and update bug-report template.

### Week 4

- Full pre-publish rehearsal from clean env:
  - ruff
  - mypy (enforced scope)
  - tests
  - strict docs
  - build + twine check
- Publish beta candidate with explicit known limitations (if any).

---

## 8. Beta Exit Checklist

- [x] Ruff is zero-error on package and repository scopes.
- [x] MyPy is green on `nirs4all/` package scope.
- [ ] MyPy command used in release workflow is deterministic and green.
- [ ] Strict docs build passes with no fail-open workflow behavior.
- [ ] License/version metadata is internally consistent across public surfaces.
- [ ] Typed package claim is truthful (`py.typed` shipped) or removed.
- [ ] Dependency declarations are normalized and reproducible.
- [ ] Governance/support/security docs and templates are complete.
- [ ] Release workflow (tests + strict docs + build + twine) is fully green.

---

## 9. Final Assessment

`nirs4all` is notably closer to beta than in the previous audit because lint and package-scope typing are now clean and sustained. Remaining blockers are concentrated in docs release quality, contract consistency (legal/version/typed/dependencies), and governance completeness. Once those release-surface items are closed, the project can credibly move to public scientific beta.
