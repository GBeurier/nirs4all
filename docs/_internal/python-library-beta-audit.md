# nirs4all Python Library Beta Readiness Audit

Audit date: 2026-02-19  
Scope: `d:\nirs4all\nirs4all` (Python library only, excluding webapp)  
Context: full audit refresh after major refactor/lint/type cleanup commits

---

## 1. Executive Verdict

The library has improved substantially since the previous audit, especially on lint quality and CI structure. It is still **not beta-ready yet** under top-tier production/open-source standards because core release gates remain red or inconsistent.

### Readiness score (today)

- Scientific capability: **9.0/10**
- API/product scope: **8.5/10**
- Test breadth: **8.5/10**
- CI/release engineering: **6.5/10**
- Documentation reliability: **3.5/10**
- Type/lint gate health: **6.0/10**
- OSS governance maturity: **5.5/10**
- Packaging/metadata correctness: **4.5/10**

**Overall beta readiness: 6.3/10**

---

## 2. Since The Previous Audit

## Major improvements

- Ruff debt collapsed from **8861 issues** to **2 issues**.
- MyPy debt reduced from **1490** to **1076** errors.
- Bare `except:` in package code reduced from **17** to **0**.
- `type: ignore` usage in package code reduced from **29** to **9**.
- CI now includes explicit lint and type-check jobs plus broader test matrix.
- `SECURITY.md` was added.
- Test inventory increased (`6723` -> `7025` collected tests).

## Regressions / still critical

- Strict docs build warning volume increased (`1909` -> **3624 warnings**).
- Docs workflow still uses `continue-on-error: true` for build/linkcheck.
- Legal and metadata consistency issues remain unresolved.
- Typed package contract (`Typing :: Typed`) is still not truthful in artifact contents.

---

## 3. Evidence Snapshot

## 3.1 Codebase footprint

- Package Python files: **434**
- Test Python files: **358**
  - Unit: **285**
  - Integration: **65**

## 3.2 Test collection

- `pytest --collect-only`: **7025 tests**
- `pytest tests/unit --collect-only`: **6444 tests**
- `pytest tests/integration --collect-only`: **581 tests**

## 3.3 Complexity indicators

- Files over 500 LOC: **149**
- Files over 1000 LOC: **50**
- Files over 1500 LOC: **23**
- Files over 2000 LOC: **12**

Largest files:

- `nirs4all/controllers/data/merge.py` (`5231` LOC)
- `nirs4all/synthesis/fitter.py` (`5203` LOC)
- `nirs4all/synthesis/_constants.py` (`2844` LOC)
- `nirs4all/pipeline/storage/workspace_store.py` (`2707` LOC)
- `nirs4all/data/predictions.py` (`2321` LOC)

## 3.4 Quality gate signals

- Ruff: **2 errors**
  - `B008` in `nirs4all/data/_indexer/index_store.py`
  - `I001` in `nirs4all/pipeline/storage/workspace_store.py`
- MyPy: **1076 errors in 240 files**
  - Top categories:
    - `import-untyped`: 309
    - `arg-type`: 192
    - `assignment`: 150
    - `union-attr`: 93
    - `import-not-found`: 70

## 3.5 Reliability/debt indicators

- `print(...)` occurrences:
  - Package: **421**
  - Tests: **119**
- `except Exception` occurrences:
  - Package: **268**
  - Tests: **17**
- bare `except:` occurrences:
  - Package: **0**
  - Tests: **0**
- `type: ignore` occurrences:
  - Package: **9**
  - Tests: **6**
- Test markers:
  - `skip/skipif`: **74**
  - `xfail`: **0**
  - `xdist_group`: **19**

## 3.6 Docs/release/package checks

- Strict docs build (`sphinx-build -W --keep-going`): **FAILED**
  - Warning count: **3624**
- Build artifacts (`python -m build --no-isolation`): **PASSED**
- Local `twine` availability (`python -m twine --version`): **FAILED** (module missing in current env)
- CLI health checks:
  - `nirs4all --test-install`: **PASSED** (optional frameworks absent)
  - `nirs4all --test-integration`: **PASSED**

---

## 4. What Is Strong

- Scientific and modeling breadth remains excellent.
- Test corpus is large and growing.
- CI architecture is materially improved:
  - lint job
  - mypy job
  - unit + integration jobs on multi-OS/multi-Python matrix
  - timeout control on CI tests job
- Security disclosure policy now exists (`SECURITY.md`).
- Build and packaging still succeed locally.

---

## 5. Beta Blockers (Must Resolve Before Public Beta)

## B1. Required CI gates are introduced but currently not green

Current state:

- `ruff check` still fails on 2 issues.
- `mypy nirs4all/` fails on 1076 errors.

Why this blocks beta:

- Public beta requires a stable, enforced, green baseline for merge safety.

Required beta outcome:

- Either:
  - Green ruff+mypy under current scope, or
  - Explicit staged scope in CI with clear roadmap and no silent bypass.

## B2. Documentation quality gate is still non-blocking and degraded

Current state:

- Strict docs build fails with **3624 warnings**.
- `.github/workflows/docs.yml` keeps `continue-on-error: true` on build and linkcheck.

Why this blocks beta:

- Docs are a core product surface for scientific users; unresolved API and reference warnings at this volume are incompatible with beta quality.

Required beta outcome:

- Remove fail-open behavior in release-critical docs checks.
- Reduce warning count to an explicit accepted budget (near-zero target).

## B3. License and metadata contract remains contradictory

Current state:

- `LICENSE` says default is AGPL-3.0-or-later.
- README badge/license text advertises CeCILL-2.1.
- `pyproject.toml` classifier also advertises CeCILL.

Related drift:

- README citation shows version `0.8.0` while package version is `0.7.1`.
- `SECURITY.md` supports `0.8.x` while current package metadata is `0.7.1`.

Why this blocks beta:

- Legal/version ambiguity reduces institutional trust and increases adoption risk.

Required beta outcome:

- One coherent license story and aligned versioning references across all public artifacts.

## B4. Typed package claim is still inaccurate

Current state:

- `pyproject.toml` declares `Typing :: Typed` and includes `py.typed` as package data.
- `nirs4all/nirs4all/py.typed` is missing.
- Built wheel does not contain `py.typed`.

Why this blocks beta:

- Published typing contract is currently misleading.

Required beta outcome:

- Ship valid `py.typed` marker and enforce typing contract, or remove typed claim until ready.

## B5. Dependency contract drift remains

Current state:

- In `requirements.txt` but not core dependencies: `ikpls`, `pyopls`, `trendfitter`.
- In core dependencies but missing from `requirements.txt`: `duckdb`, `umap-learn`.
- Docs extras still do not include docs-specific deps used in docs CI (for example `sphinxcontrib-mermaid`).

Why this blocks beta:

- Different install paths can create inconsistent runtime/doc behavior.

Required beta outcome:

- Define one authoritative dependency source and verify generated requirements parity.

## B6. OSS governance is still partial

Current state:

- Present: `SECURITY.md`.
- Missing: `SUPPORT.md`, `GOVERNANCE.md`, `.github/CODEOWNERS`, PR template, `.pre-commit-config.yaml`.
- Issue template still contains non-library web/mobile fields (`OS: iOS`, browser/smartphone references).

Why this blocks beta:

- Weak maintenance contract for community users and contributors.

Required beta outcome:

- Complete maintainer/governance documentation and modernize templates for scientific reproducibility reports.

---

## 6. High-Priority Work (Beta Window)

## H1. Complete lint closure and define mypy rollout policy

- Fix remaining 2 ruff issues immediately.
- Split mypy into enforceable tiers:
  - Tier 1: critical public API and core pipeline modules
  - Tier 2: expanded internal modules
  - Tier 3: optional backend/model families
- Keep CI hard-fail only on declared enforceable tiers.

## H2. Docs stabilization program

- Triage top warning classes (duplicate objects, unresolved refs, missing imports, toctree issues).
- Make API docs generation deterministic and avoid uncontrolled churn in tracked files.
- Keep strict mode in CI once warning budget is under control.

## H3. Metadata/legal normalization

- Align `LICENSE`, README badges/text, `pyproject` classifiers, SPDX guidance.
- Align version references in README citation and `SECURITY.md` support matrix.

## H4. Packaging truthfulness and release checks

- Add `py.typed` or drop typed classifier.
- Ensure twine validation is both in CI and documented for local release rehearsal.
- Update deprecated setuptools license metadata style (SPDX expression + modern fields).

---

## 7. Recommended 4-Week Beta Hardening Plan

## Week 1

- Close ruff to zero.
- Decide and document license/version canonical source.
- Add missing governance/support templates (CODEOWNERS, PR template, SUPPORT/GOVERNANCE docs).

## Week 2

- Establish mypy Tier-1 enforceable scope and make CI green on that scope.
- Align dependency declarations (`pyproject` vs requirements files).
- Add `py.typed` decision and implement.

## Week 3

- Execute docs warning burn-down for highest-volume classes.
- Remove docs `continue-on-error` once warning budget is acceptable.
- Update issue templates for scientific bug reports.

## Week 4

- Full pre-publish dry-run:
  - lint
  - mypy enforced tiers
  - test matrix
  - strict docs
  - build + twine check
- Publish beta candidate with explicit known limitations.

---

## 8. Beta Exit Checklist

- [ ] Ruff is zero-error on enforced CI scope.
- [ ] MyPy is green on declared enforceable scope, with documented expansion roadmap.
- [ ] Strict docs build passes without fail-open workflow behavior.
- [ ] License/version metadata is internally consistent across all public surfaces.
- [ ] Typed package claim is truthful (`py.typed` shipped) or removed.
- [ ] Dependency declarations are normalized and reproducible across install modes.
- [ ] Governance/support/security docs and templates are complete.
- [ ] Release workflow (tests + docs + build + twine) is fully green.

---

## 9. Final Assessment

The project has made real progress toward beta and is now much closer on code quality discipline. The remaining blockers are concentrated in docs quality, typing gate closure strategy, and metadata/legal consistency. Clearing those items will move `nirs4all` from “promising pre-beta” to a credible production-grade scientific beta.
