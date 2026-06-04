# Phase 1 — Repository Hygiene Check

**Date:** 2026-06-04
**Repo:** `nirs4all` (the Python library), `origin = https://github.com/GBeurier/nirs4all.git`

## Verdict

✅ **`main` is production-clean.** Nothing blocks the documentation work.
⚠️ **Branch hygiene is messy** — 21 stale local branches and 2 leftover worktree branches. Recommended optional cleanup (not done without your approval; deleting branches is destructive).

## Checks performed

| Check | Result |
|---|---|
| Working tree | **clean** — nothing to commit |
| `HEAD` vs `origin/main` | **0 ahead / 0 behind** — fully committed & pushed (`0f9e08cb`) |
| Stashes | **none** |
| Active worktrees | **1** (the main checkout only) — no dangling worktree dirs registered |
| Merge in progress | **none** (no `MERGE_HEAD`, no rebase/cherry-pick state) |
| Remote | single (`origin`); fetch OK |

## Local branches

**Fully merged into `main`** (0 commits ahead of `main` → safe to prune): `cache_refactoring`, `chore/bump-docker-metadata-action-v6`, `clean_legacy_deprec`, `duckdb_refactoring_2`, `fix/msc-standard-axis`, `fix/parquet-loader-categorical-mode`, `main2`, `parquets_refactoring`, `refactor_branching`, `sqlite_migration`, `tabpfn_pape`, `tabpfn_paper`, `worktree-2025-12-23T16-39-04`, `worktree-2025-12-23T17-21-13` *(**14 prunable branches**)*.

> **Codex review correction (2026-06-04):** an earlier draft said "15 branches". `git branch --merged main` lists **15** entries, but one of them is `main` itself — only **14** are prunable feature branches. Corrected above. All other claims in this file were independently re-verified as correct by Codex (`origin/main...HEAD = 0/0`, clean tree, no stash/worktree/merge state, merged-vs-unmerged spot-checks all matched). Transcript: `codex_reviews/phase1_hygiene_review.txt`.

**Carrying unmerged commits** (work not in `main`):

| Branch | Ahead of main | Behind main | Last commit | Note |
|---|---|---|---|---|
| `PP_Split_selectors` | +3 | 447 | 2025-11-29 | Very stale; far behind — likely abandoned/superseded. |
| `fix/issue-24-branch-preprocessing-chain` | +1 | 146 | 2026-02-25 | Old fix branch; would need rebase. |
| `dependabot/github_actions/actions/download-artifact-8` | +1 | 141 | 2026-02-26 | Automated dep bump. |
| `dependabot/github_actions/actions/upload-artifact-7` | +1 | 141 | 2026-02-26 | Automated dep bump. |
| `dependabot/github_actions/docker/build-push-action-7` | +1 | 141 | 2026-03-05 | Automated dep bump. |
| `dependabot/github_actions/docker/login-action-4` | +1 | 141 | 2026-03-05 | Automated dep bump. |
| `dependabot/github_actions/docker/setup-buildx-action-4` | +2 | 141 | 2026-03-10 | Automated dep bump. |

`main2`, `tabpfn_pape` (typo of `tabpfn_paper`), and the two `worktree-*` branches look like leftovers.

Remote-only branches include `0.4.1`, `0.4.2` (release branches), `branching`, `cli`, and several dependabot branches (17 total).

## Recommendation (optional, needs your go-ahead)

1. Prune the 14 fully-merged local feature branches: `git branch -d <name>` (safe; git refuses if not merged).
2. Decide on the 2 stale feature branches (`PP_Split_selectors`, `fix/issue-24-branch-preprocessing-chain`) — rebase-and-PR or delete.
3. Let dependabot branches be (GitHub manages them; they auto-close on merge/supersede).

None of this is required for the doc overhaul — `main` is the single source of truth and is clean.
