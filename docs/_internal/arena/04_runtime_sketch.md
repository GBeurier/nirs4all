# Runtime central — sketch architectural v0.1

> Sketch préliminaire, pas une spec d'implémentation. L'objectif est de fixer **les composants, leurs responsabilités et leurs contrats d'interface**, pas le détail des technologies retenues. Tout choix technique (ordonnanceur, file GPU, persistance) reste arbitrable.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §3.3, §5 |
| Liens | [grille v0.1b](03_canonical_grid_v0.1.md) ; [PLS-canon](02_pls_canon.md) |
| Version | v0.1 |
| Statut | Sketch — à compléter avant l'implémentation effective |
| Phase roadmap | sketch écrit en **Phase 0a** (Conception) ; runtime alpha (format A) implémenté en **Phase 2** ; extensions formats B/C en **Phase 3** ; scaling bêta en **Phase 5**. Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 1. Périmètre du v0.1

**Le runtime central v0.1 doit pouvoir :**

1. Recevoir une soumission (`pipeline.json` + `bundle.n4a` + `sanity_check.csv` + `CITATION.cff`).
2. La valider syntactiquement, sémantiquement, et reproduire le sanity-check.
3. Générer la grille v0.1b pour la tâche concernée (régression / classification).
4. Exécuter ≈ 3 660 runs atomiques en parallélisme contrôlé.
5. Persister chaque résultat atomique (métriques + prédictions + résidus + EnvCard).
6. Re-exécuter aléatoirement 1-2 runs pour vérifier la reproductibilité interne.
7. Indexer les résultats dans la DB, calculer les vues dérivées de base.
8. Notifier le contributeur et les auteurs de datasets/PP/augmenters concernés (cf §12 du manifeste).

**Hors v0.1** : interface web de soumission (la soumission v0.1 se fait par PR Git ou upload manuel) ; API REST complète ; système d'authentification ; hold-out server.

## 2. Composants

```
┌─────────────────────┐
│  Submission         │  Reception : Git PR, upload simple, ou CLI
│  Receiver           │  Validation : schéma, intégrité, sanity
└──────────┬──────────┘
           │ valid Submission
           ▼
┌─────────────────────┐
│  Grid Expander      │  pipeline + grille canonique → liste de RunSpec
└──────────┬──────────┘
           │ list[RunSpec]
           ▼
┌─────────────────────┐
│  Scheduler          │  RunSpec → tasks dans queue(s) (CPU / GPU)
└──────────┬──────────┘
           │ tasks
           ▼
┌─────────────────────┐
│  Executor pool      │  joblib loky CPU + queue GPU séparée
│  (CPU + GPU)        │  → exécute, mesure (temps, mémoire), capture failures
└──────────┬──────────┘
           │ run_atomic_result
           ▼
┌─────────────────────┐
│  Persister          │  store atomic result : Parquet + DuckDB + .n4a bundles
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Reproducibility    │  ré-exécute 1-2 runs aléatoires, compare bit-à-bit (avec
│  Verifier           │  tolérance pour float)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Indexer + Notifier │  invalide les vues dérivées, recalcule, notifie
└─────────────────────┘
```

## 3. Composant par composant

### 3.1 Submission Receiver — format-aware

**Responsabilités :**
- Recevoir une soumission. Le contenu varie par format (cf [06_submission_formats.md](06_submission_formats.md)) :
  - **Format A** : `manifest.json` (alias `pipeline.json` pour rétrocompatibilité) + `bundle.n4a` + `sanity_check.csv` + `CITATION.cff`.
  - **Format B** : `manifest.json` + `sanity_check.csv` + `CITATION.cff` (pas de bundle ; le paquet sera installé par `pip`).
  - **Format C** : `manifest.json` + `r_bridge.R` + `sanity_check.csv` + `CITATION.cff` (R installé via container).
- **Détecter le format** via le champ `format ∈ {A, B, C}` du manifeste.
- Valider contre le JSON Schema correspondant (`manifest_schema_A.json`, `_B.json`, ou `_C.json`).
- **Sandbox setup** :
  - Format A : aucun (déjà nirs4all natif).
  - Format B : créer un venv isolé ; `pip install <package_name><version_spec>` ; vérifier import + API sklearn.
  - Format C : pull container Docker `nirs4all-arena-r:v0.1` ; vérifier disponibilité du paquet R via `Rscript -e 'requireNamespace(...)'` (installer dans `R_LIBS_USER` si absent).
- Audit du pipeline :
  - Format A : `fit_on_train_only` automatique (audit du pipeline JSON).
  - Format B/C : déclaration `fit_on_train_only = true` requise ; détection rétrospective post-exécution.
- **Reproduction du sanity-check** : exécuter le modèle (selon format) sur `sample_data/regression` (seed 0) ; comparer au CSV soumis. Tolérance numérique : `1e-6` pour A et B, `1e-4` (relaxable) pour C.
- Valider `CITATION.cff` via `cffconvert` ou équivalent.

**Contrat d'erreur** :
```python
SubmissionRejected(
    reason: Literal["schema", "format_unsupported", "bundle_corrupt",
                    "pip_install_failed", "docker_pull_failed",
                    "r_package_unavailable", "sklearn_api_mismatch",
                    "citation_invalid", "audit_pipeline_unsafe",
                    "sanity_mismatch"],
    detail: str,
    format: Literal["A", "B", "C"],
)
```

**Sortie** : un dossier `submissions/<submission_id>/` avec :
- `manifest.json` (validé + métadonnées de réception + format)
- `bundle.n4a` (format A) **ou** `venv_lockfile.txt` (format B, snapshot post-install) **ou** `r_session_info.txt` (format C, post-sanity)
- `audit_log.json`

### 3.2 Grid Expander

**Responsabilités :**
- Charger la grille canonique active (`grid_v0.1b.json`) selon la tâche.
- Charger le registry des datasets (cf [01_dataset_cartography_tasks.md](01_dataset_cartography_tasks.md)).
- Charger les listes `selected_v0.1`, `fast12_transfer_core`, `audit20_transfer_core`, `scaling_datasets`, `cross_instrument_pairs`.
- Appeler `generate_canonical_grid(...)` (cf §7 du doc grille).
- Vérifier la compatibilité de chaque `RunSpec` avec les `input_constraints` du modèle déclarés dans `pipeline.json`. Marquer les incompatibles `skip_constraint` avant exécution.

**Sortie** : `submissions/<submission_id>/run_specs.parquet` (liste des RunSpec avec colonne `status_pre = "ok" | "skip_constraint"`).

### 3.3 Scheduler

**Responsabilités :**
- Allouer les `RunSpec` aux pools d'exécution selon `runtime_tier` du modèle :
  - `fast` (< 60 s) : pool CPU, n_jobs élevé.
  - `medium` (1-30 min) : pool CPU, n_jobs modéré.
  - `slow` (30 min - 2 h) : pool CPU avec timeout strict.
  - `very_slow` (> 2 h) : pool GPU si `requires_gpu=true`, sinon pool CPU long-running.
- Gérer la priorité : soumissions plus anciennes d'abord (FIFO), avec interruption possible pour soumissions urgentes (re-run bump majeur).
- État persistant : si le runtime crashe, reprendre où il s'est arrêté (cf §3.5).

**Choix technique v0.1 (proposition)** :
- File CPU : `joblib.Parallel(backend="loky", n_jobs=N_cores)` au-dessus de la file SQLite des RunSpec en attente.
- File GPU : queue Python simple (`asyncio.Queue` ou `multiprocessing.Queue`) avec workers GPU dédiés. Pas d'orchestrateur distribué (Airflow, Prefect, Celery) en v0.1 — ajouter en v0.2 si la traction l'exige.
- Pas de Kubernetes en v0.1 : le runtime tourne sur une seule machine bien dotée.

**À évaluer** : SLURM si le cluster est disponible.

### 3.4 Executor (CPU et GPU) — dispatch par format

**Responsabilités :**
- Pour chaque `RunSpec` :
  1. Charger le dataset : `nirs4all.benchmark.datasets.load(dataset_alias)`.
  2. Appliquer le split : charger le masque pré-calculé `splits/<dataset_alias>/<scheme>_seed<n>.parquet`.
  3. Appliquer le sous-échantillonnage `N` si bloc B6 (sur train seulement ; test inchangé).
  4. Construire le pipeline de run = `[augmenter] + pp_chain + [outlier_filter] + [target_processing] + model`.
     - **Format A** : le `model` est instancié depuis le bundle `.n4a` via `BundleLoader`.
     - **Format B** : le `model` est instancié depuis le venv via `from <import_path> import <class_name>` + `ClassName(**constructor_kwargs)` ; wrappé dans la chaîne nirs4all.
     - **Format C** : le `model` est représenté par une étape `{"format": "C", "manifest_path": ..., "r_bridge_script_path": ...}` qui sera routée vers `RModelController` (container Docker).
  5. Fixer les seeds globales via `nirs4all.benchmark.seed_env.set_global_seed(seed)` ; pour format C, écriture de `seed.txt` lue par le bridge R.
  6. Capturer `EnvCard` :
     - Format A : versions nirs4all + sklearn + BLAS + threads + GPU.
     - Format B : + venv `pip freeze --all` post-install + package_index.
     - Format C : + R version + `sessionInfo()` + container digest.
  7. `fit` puis `predict` ; mesurer `fit_time_s`, `predict_time_s`, `peak_rss_mb`, `peak_gpu_mb`.
     - Format C : la mesure mémoire/temps inclut l'overhead Docker (déclaré dans `notes`).
  8. Calculer toutes les métriques (primaire + secondaires §2.4 PLS-canon).
  9. Persister prédictions et résidus en Parquet.
  10. Retourner `RunAtomicResult`.
- Si échec : capturer code typé (`failed_nan`, `failed_oom`, `failed_timeout`, `failed_convergence`, `failed_dispatch`, `failed_pip_resolve`, `failed_docker`, `failed_r_bridge`) et message tronqué.

**Mesure temps/mémoire** :
- Warm-up : un fit blanc préalable pour modèles `fast` (pool allocator, JIT torch/jax).
- Répétitions ×3 pour `fast` et `medium` ; médiane + IQR.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` *fixés* avant le run, logués dans EnvCard.
- Timeout `runtime_tier_max` strict ; SIGTERM puis SIGKILL.

**Sandboxing** : v0.1 = isolation par process (joblib `loky` + worker pool). v0.2 envisagée : containers Docker.

### 3.5 Persister

**Responsabilités :**
- Persistance idempotente : un `RunSpec` déjà persisté n'est pas ré-exécuté si reprise du runtime.
- Schéma résultat : suit le `results.parquet` schema de `bench/harness/run_benchmark.py::ResultRow` étendu (cf manifeste §2.5).
- Bundles `.n4a` re-générés : un par `RunSpec` pour les `selected` ; un par dataset pour les autres (économie disque).

**Choix technique v0.1** :
- `WorkspaceStore` (SQLite WAL) déjà mature pour les métadonnées.
- `ArrayStore` (Parquet) déjà mature pour les prédictions/résidus.
- DB principale = DuckDB en lecture (analytics) ; SQLite WAL en écriture (concurrent runs).

### 3.6 Reproducibility Verifier

**Responsabilités :**
- À la fin de l'exécution d'une soumission, sélectionner aléatoirement 1-2 % des runs (au moins 2, au plus 50) et les ré-exécuter.
- Comparer bit-à-bit (avec tolérance `1e-6` absolu sur les prédictions, `0` pour les hashes d'EnvCard).
- Tout écart : flagger la soumission `under_review`, alerter le mainteneur.
- Si l'écart est imputable à un facteur connu non-déterministe (cuDNN, BLAS multi-thread), accepter mais documenter dans `reproducibility_caveats` du résultat.

### 3.7 Indexer + Notifier

**Responsabilités :**
- Invalider les vues dérivées (leaderboard, matrices) — implémentées comme requêtes DuckDB sur la base. Pas de précalcul lourd : la recompilation est rapide.
- Recalculer les agrégations basiques :
  - Leaderboard global (B1 : médiane sur datasets `selected` × 2 splits × 10 seeds, IC bootstrap).
  - Matrice modèle × dataset (`score_ratio_vs_pls_canon`).
  - Décomposition de variance.
  - Matrice de fragilité.
- Notifier :
  - Le contributeur soumissionnaire (résumé + lien vers la soumission publiée).
  - Les auteurs des `selected` datasets, `selected` PP, `selected` augmenters dont la soumission utilise le module (rétro-notification §12 du manifeste).
- Politique de notification : email + RSS feed du repository GitHub Pages.

## 4. Flow de données

```
Submission disk
    ↓
WorkspaceStore.submissions (SQLite WAL)  ← contributors, sanity_check, audit_log
    ↓
run_specs.parquet (Parquet, immutable)
    ↓
[ Executor pool ]
    ├── per-run atomic result → WorkspaceStore.runs (SQLite WAL)
    ├── per-run predictions → arrays/<dataset>/<run_id>.parquet
    ├── per-run residuals → arrays/<dataset>/<run_id>_res.parquet
    └── per-run bundle → artifacts/<bundle_hash>.n4a
    ↓
DuckDB views (lecture-only, computed on demand)
    ↓
Web pages (statiques générées par templating, voir [05_web_minimal_sketch.md])
```

## 5. États d'une soumission

```
received → validating → validated
                 │           │
                 ▼           ▼
              rejected     expanding (grid)
                            │
                            ▼
                          running ─→ paused (par operator)
                            │           │
                            ▼           ▼
                         completed   resumed
                            │
                            ▼
                          verifying (reproducibility check)
                            │
                            ▼
                          indexing
                            │
                            ▼
                          published
                            │
                            ▼
                  (event: bump majeur grille v(N) → vN+1)
                            │
                            ▼
                          retired (mais accessible avec tag v(N)-frozen)
```

## 6. Comptabilité (accountability) — invariants vérifiables

1. **Tout run persisté a un EnvCard non-vide.** Sinon, run invalide.
2. **Tout run a un `seed_card_hash` déterministe.** Le SeedCard inclut les états PRNG de numpy / random / torch / jax / sklearn (si présents) + `PYTHONHASHSEED`.
3. **Toute soumission publiée a passé le ReproducibilityVerifier.** Sinon, statut `under_review`.
4. **Toute soumission utilisant un dataset `retired` est elle-même flaggée.**
5. **Tout run de baseline (PLS-canon, Ridge-canon, RF-canon) sur un dataset `selected` doit avoir été calculé *avant* l'admission de la première soumission externe sur ce dataset.**

## 7. Concurrence et idempotence

Le runtime peut crasher (machine, kernel panic, OOM). Politique :
- Tout `RunSpec` est identifié par `run_spec_hash = blake2b(canonical_form(RunSpec))`.
- À la reprise, le scheduler liste les `run_spec_hash` déjà persistés ; les autres sont ré-enqueued.
- Pas de double-écriture : `WorkspaceStore.runs` a contrainte UNIQUE sur `run_spec_hash`.

## 8. Failure handling — politique opérationnelle

| Code | Action |
|---|---|
| `failed_nan` | persister avec status, ne pas re-tenter |
| `failed_oom` | persister, marquer le `RunSpec` `requires_more_ram` ; ne pas re-tenter sauf si nouvelle allocation |
| `failed_timeout` | persister, marquer ; ne pas re-tenter sauf nouveau tier |
| `failed_convergence` | persister ; ne pas re-tenter |
| `failed_dispatch` | persister + alerte mainteneur ; bug à fixer côté soumission ou côté runtime |
| `skip_constraint` | persister, exclu du leaderboard, indexé en couverture |

Aucune re-tentative automatique pour les bugs non-classés. Mainteneur statue.

## 9. Sécurité minimale (v0.1) — différenciée par format

V0.1 applique une politique de sandboxing **différenciée selon le format** (cf [06_submission_formats.md](06_submission_formats.md) §3.7 et §4.7) :

- **Format A** (bundle nirs4all) : pas de sandbox dédié. Hypothèse : soumissions de contributeurs identifiés (PR Git). Le bundle `.n4a` peut contenir joblib/pickle ; risque de désérialisation arbitraire. *Mitigation* : audit du contenu joblib ; remplacement par un format sans exec (ONNX, TF SavedModel) reporté en v0.2.
- **Format B** (Python lib externe) : **venv isolé obligatoire dès v0.1**. `pip install` exécute du code arbitraire (`setup.py`). *Mitigation* : venv séparé par soumission, pas d'accès réseau pendant l'exécution (uniquement pendant install), filesystem en lecture seule sauf `output_dir`, timeout strict global. v0.2 : container Docker.
- **Format C** (R package) : **container Docker obligatoire dès v0.1**. `install.packages` télécharge et compile du code C/C++/Fortran (Makevars) — risque structurellement plus élevé que B. Image `nirs4all-arena-r:v0.1` ; bibliothèque R sandboxée via `R_LIBS_USER` ; accès réseau restreint à CRAN/Bioconductor whitelistés ; filesystem read-only sauf `output_dir`.

## 10. Coût estimé d'une soumission (à valider empiriquement)

| Type modèle | Temps/run estimé | Total runs | Total temps |
|---|---|---|---|
| Léger (PLS, Ridge, RF) | 1-30 s | 3 660 | 1-30 h CPU |
| Moyen (sklearn ensemble, autres PLS variants) | 30 s - 2 min | 3 660 | 30-122 h CPU |
| Lourd (TabPFN, NN CPU) | 1-10 min | 3 660 | 60-610 h CPU |
| GPU (TabPFN GPU, NN GPU) | 10-60 s | 3 660 | 10-60 h GPU |

Sur une machine 32 cores : ≈ 1-4 j calendaire pour un modèle léger. Acceptable.

## 11. Dépendances externes minimales

**Côté runtime (toujours requis)** :
- Python 3.11+
- `nirs4all` (à jour)
- `joblib`, `numpy`, `scipy`, `pandas`, `pyarrow`
- `duckdb` (lecture analytique)
- `sklearn`
- `tqdm` (progress)
- `cffconvert` (validation CITATION.cff)

**Format B (Python lib externe)** :
- `virtualenv` ou `venv` (Python stdlib) — création du sandbox.
- `pip` — installation des paquets soumis.

**Format C (R package)** :
- **Docker** (engine + daemon accessibles au runtime).
- Image **`nirs4all-arena-r:v0.1`** (base `rocker/r-ver:4.3.x`, avec `arrow`, `jsonlite`, `mdatools`, `pls`, `prospectr`, `caret`, `e1071`, `randomForest` pré-installés).
- (optionnel v0.2) `rpy2` pour bridge in-process plus rapide que subprocess.

**Selon modèles soumis** (chargement *lazy* via `nirs4all`) :
- `torch`, `jax`, `tensorflow`, `catboost`, `xgboost`.

## 12. Tests d'acceptation v0.1

Avant le go-live v0.1, le runtime doit :

1. **Recevoir et valider** la soumission PLS-canon en mode "auto-soumission" : le runtime soumet *à lui-même* PLS-canon comme première soumission. Doit passer toutes les étapes.
2. **Exécuter B1 complet** sur les 26 datasets `selected` (PLS-canon) → ≈ 520 runs.
3. **Reproduire** un run aléatoire avec écart < 1e-6.
4. **Indexer** et publier la première page (leaderboard avec 1 modèle).
5. **Re-soumettre** PLS-canon sans changement → le runtime détecte les `run_spec_hash` existants et ne ré-exécute *aucun* run (idempotence).

Si ces 5 tests passent, v0.1 est utilisable pour des contributeurs invités.

## 13. Hors-périmètre v0.1, à planifier v0.2

- Soumission via web (endpoint REST + UI de upload).
- Authentification et gestion d'identité (OAuth GitHub probablement).
- Sandboxing Docker.
- Orchestration cluster (SLURM, Kubernetes, Prefect).
- Hold-out test set sacré (modèle Kaggle).
- Auto-recyclage des datasets `retired` (politique de remplacement).
- Diagnostic d'alignement physique automatique (vues qualitatives).

## 14. Décisions à arbitrer (pour Codex/comité)

- **Orchestrateur** : joblib pur, ou intégration légère Prefect/Dagster dès v0.1 ?
- **DB primaire** : SQLite + Parquet (déjà mature dans nirs4all) ou DuckDB natif ?
- **Web** : Hugo + JSON statique ? FastAPI + Vega-Lite ? Datasette ? (cf [05_web_minimal_sketch.md])
- **GPU queue** : workers Python isolés ou intégration vLLM-style ?
- **Politique des soumissions GPU lourdes** : limite x runs/jour ? file dédiée payante ?
