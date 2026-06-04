# Documentation & Examples Overhaul — Work Log

**Started:** 2026-06-04
**Owner:** Claude (Opus 4.8) — orchestrated, Codex-reviewed at each step
**Goal (verbatim from request):** nirs4all goes into intensive production → it needs *rock-solid documentation*. Deliver (1) a repo-hygiene check, (2) a detailed, structured, exhaustive audit of the docs + examples, (3) a complete rewrite/restructuring proposal at the level of best-in-class library docs. Every step reviewed by Codex agents. Keep a written trace. Justify every proposal.

This file is the running trace. Companion deliverables live in the same folder:

| File | Content |
|---|---|
| `00_worklog.md` | This running trace (decisions, methodology, Codex review rounds). |
| `01_repo_hygiene.md` | Phase 1 — repo state verification. |
| `02_audit.md` | Phase 2 — exhaustive audit of docs + examples. |
| `03_restructuring_proposal.md` | Phase 3 — target IA, migration plan, style guide. |
| `codex_reviews/` | Raw Codex review transcripts for the audit and the proposal. |

---

## Cross-cutting principle the user wants enforced

> "1 pipeline, des datasets, run()." Pipelines must be written **inline, in one shot**, with no external variables, for readability.

Canonical form to enforce everywhere (docs code blocks + every example):

```python
result = nirs4all.run(
    pipeline=[ ... steps inline ... ],
    dataset=...,
)
```

NOT:

```python
pipeline = [ ... ]          # ❌ external variable
result = nirs4all.run(pipeline, dataset)
```

This is treated as a first-class audit dimension ("style uniformity") and a deliverable of the proposal (a STYLE GUIDE).

---

## Methodology

- **Ultracode mode**: exhaustive multi-agent orchestration. The audit fans out one reader agent per documentation/examples segment; each reader grounds its claims against the *actual source code* (not memory), using a shared "ground-truth card" produced first.
- **Codex gate**: the audit and the proposal are each handed to Codex (`codex exec`, CLI 0.136.0) for an independent adversarial review before being finalized. Review transcripts are archived under `codex_reviews/`.
- Deliverables are written in **English** (the docs are English and these files will be acted upon in-repo). Status summaries to the user are in French.

---

## Timeline

### Phase 1 — Repo hygiene (DONE + Codex-reviewed)
- Verified `main` is clean and synced with `origin/main`. Catalogued local branches. See `01_repo_hygiene.md`.
- **Codex review round 1** (`codex exec`, CLI 0.136.0): verdict **MINOR ISSUES**. Codex independently re-ran every git check and confirmed all claims EXCEPT a counting slip — "15 merged branches" should be **14** (the 15th entry from `git branch --merged main` is `main` itself). Fixed in `01_repo_hygiene.md`. Transcript: `codex_reviews/phase1_hygiene_review.txt`.

> **Session resumed** (same day): the previous session was interrupted while launching the audit workflow. Phase 1 re-verified (still clean: `main` 0/0 vs `origin/main`, no stash, no merge state, single worktree — only this folder is untracked). Resuming at Phase 2 launch + Codex review of Phase 1.

### Phase 2 — Audit (in progress)
- Mapped the full doc surface: **478** auto-generated API `.rst` stubs + **~130** narrative `.md` pages + **~70** example `.py` files + root `.md` files.
- Early hand-spotted defects (confirmed before the workflow, used to calibrate the audit):
  - `docs/source/index.md` citation block pins `version = {0.2.1}` — real version is `0.9.1`.
  - `docs/source/index.md` states license "CECILL-2.1"; a recent commit (`7bd07280`) realigned the repo to **AGPL-3.0**. License statements are inconsistent across `index.md`, root `CLAUDE.md`, `README.md`, `LICENSE`.
  - `docs/source/onboarding/persona_paths.md` references files that do not exist: `U01_minimal.py` (real: `U01_hello_world.py`), `U02_workspace.py` (real: `U02_basic_regression.py`), `examples/developer/01_branching/` (real: `01_advanced_pipelines/`), `D04_stacking.py` (real: `D05_meta_stacking.py`), and `nirs4all.load_dataset(...)` which is **not** in the public API (`__init__.__all__`).
  - Three competing "front doors" in the landing toctree: `getting_started/`, `concepts/`, `onboarding/` (+ a `developer/` that overlaps `onboarding/`). Pipeline material is scattered across `concepts/`, `user_guide/pipelines/`, and `reference/`.
  - Example counts are inconsistent across pages ("50+", "67", "progressive").
- Launching the audit workflow (ground-truth → 15 segment readers → synthesis).

### Phase 2 — Audit (DONE, Codex review in flight)
- **Ground-truth card** (`GROUND_TRUTH.md`) built first from source: real `__all__`, signatures, result members, v0.9.1, AGPL-3.0. Every agent grounded against it.
- **Audit workflow** (`audit_workflow.js`): 20 segment readers (Explore, read-only) + cross-cutting synthesis + adversarial completeness critic. **22 agents, ~1.73M tokens, 1011 tool calls, ~11 min.** 20/20 returned. Raw data archived in `data/`.
- **Completeness critic caught 2 false "critical" blockers + 1 false "clean segment"** — all 3 re-verified against source and corrected:
  - bare-class pipeline form is *valid* (`parser.py:231`) → downgraded.
  - transfer-learning imports are `try/except`-guarded (no crash) but point to wrong module paths → downgraded to medium.
  - "2 clean segments / ~250 violations" was wrong → authoritative uniform grep gives **648 violations across 140 files, 0 clean segments**.
- **Gap-closing pass** (3 agents + manual): audited `predictions.ipynb` (stale/unrunnable, not in CI → 1/5), the 4 hand-written `api/*.md` pages (duplicate `reference/`, "Version 5.0"), and build hygiene (broken-ref suppression via `nitpick_ignore_regex`; `examples.yml` weekly-only, never per-PR; zero doc-snippet execution).
- Deliverable: `02_audit.md` (verdict: mean 2.80/5, not publication-ready; mechanical/structural rot, not wrong ideas; 4 critical + 15 high + register of 40).
- **Codex review round 2** (`codex_reviews/phase2_audit_review.txt`): verdict **SOLID-WITH-FIXES**. Confirmed #3/#4/#5/#7/#9/#10/#12/#13/#15/#19 against source and both severity downgrades. Forced 6 accuracy fixes (all re-verified + applied): license is *dual* not CeCILL-wrong (no token-ban gate); `bundle=` absorbed by `**runner_kwargs`; `ExplainResult.get_feature_importance()` is real (dropped fake-method claim); `PredictionAnalyzer(result)` is a 2nd bug; `target_column` belongs in `global_params` not a kwarg; stale `Predictions.load()` also in `visualization/predictions.py:76` docstring. `GROUND_TRUTH.md` + `02_audit.md` updated.

### Phase 2 — STATUS: COMPLETE & Codex-signed-off.

### Phase 3 — Proposal (DONE, Codex review in flight)
- **Proposal workflow** (`proposal_workflow.js`): 3 competing IA candidates (strict Diátaxis / tutorial-first FastAPI / task-persona) → 4-lens judge panel → synthesis → grounded 140-row migration map. **9 agents, ~812K tokens, ~15 min.**
- Panel result: Diátaxis won SSOT+enforceability (9), FastAPI won onboarding+fit (9), task-persona trailed (6). Synthesis = **hybrid**: Diátaxis SSOT backbone + FastAPI tutorial spine + generated reference/gallery + verb/goal routing as navigation-only.
- Deliverable `03_restructuring_proposal.md`: target IA, single front door, SSOT law, inline style guide + CI lint, examples-as-gallery plan, reference unification (api/ dissolved), 5 safety-net CI gates, 140-page migration map (14 deletes / 10 merges / 13 new pages), P0–P4 roadmap, 11 success metrics. Raw data in `data/proposal_*.json`.
- **Codex review round 3** (`codex_reviews/phase3_proposal_review.txt`): verdict **SOLID-WITH-FIXES** ("IA direction good, not executable as written"). Confirmed front-door diagnosis + canonical-snippet API validity. Forced 10 executability fixes (all applied, §12 of the proposal): sphinx-gallery is net-new tooling (render no-exec in docs build, execute in per-PR CI, skip heavy backends); "examples=tests" overclaimed (run.sh hand-manifest drifts, omits U06); **lint must catch keyword `run(pipeline=var)` not just positional** (AST-based); snippet needs imports+random_state; 357 are apidoc module stubs not autosummary symbols → need autodoc_mock_imports for TF/Torch/JAX; reconciled 140 JSON dispositions vs ~92-row digest + ~18 real file deletions; added ai_onboarding/CHANGELOG/CONTRIBUTING/AGENTS rows; augmentations.md → 3-way split; P1 sweep scoped to survivors; softened all "by construction"/"structurally impossible" to gate-backed claims.

### Phase 3 — STATUS: COMPLETE & Codex-signed-off (round 3, fixes applied).

### Planning phases COMPLETE. Deliverables: 00_worklog, 01_repo_hygiene, 02_audit, 03_restructuring_proposal, GROUND_TRUTH, data/, codex_reviews/ (3 rounds).

### EXECUTION — user said "lance les taches" + goal "refactor docs+examples to be production ready" (2026-06-04)
- **Proposal refined** for two user points: (a) inline non-tyrannical (simple→inline, complex→named var OK; lint advisory); (b) dataset-config + repetitions/aggregation as a first-class NIRS topic (§5b + nav). Memories saved.
- **P0 executed** on branch `docs/overhaul-p0-correctness` (19 files, +61/−52): metadata/license/version (incl. conf.py version-from-source), ~12 copy-paste-fatal wrong-API fixes on survivor pages + 1 source docstring + 2 example import paths. ruff green; post-edit sweeps clean. See `04_execution_log.md`.
- **Audit false-positive #25 (`PredictionResultsList`) caught during execution** — it's real; corrected in `02_audit.md`.
- **Not committed** (awaiting user). Deferred: notebook→R04 (needs venv), #32 (P2), style sweep (P1), structural migration (P3/P4).
