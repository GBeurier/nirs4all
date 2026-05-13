# Formats de soumission de modèles — A, B, C

> Trois formats de soumission pour abaisser la barrière d'entrée et ouvrir l'écosystème R. Toutes les soumissions traversent la même grille canonique ; seule la *façon dont le modèle est appelé* change.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §3.2, §5 |
| Liens | [grille v0.1b](03_canonical_grid_v0.1.md), [runtime sketch](04_runtime_sketch.md), [PLS-canon](02_pls_canon.md) |
| Version | v0.1c (intègre revue Codex initiale) |
| Statut | À soumettre à seconde revue avant freeze |
| Phase roadmap | spec rédigée en **Phase 0a** (Conception) ; format A supporté en **Phase 2** ; *dispatch spike* B/C en **Phase 2** ; formats B + C complets en **Phase 3**. Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 0. Changelog v0.1 → v0.1c (post-revue Codex)

1. **Sécurité format B durcie** : `venv` seul jugé insuffisant pour `setup.py` arbitraire. **Docker container obligatoire dès v0.1 pour B** (aligné avec C). L'image dédiée s'appelle `nirs4all-arena-py:v0.1`.
2. **Sources externes fermées** : pour v0.1, `pip` accepte uniquement **PyPI officiel** ; `R` accepte uniquement **CRAN + Bioconductor**. GitHub forbiddén par défaut. Exceptions via allowlist éditoriale avec **hash de commit immuable**.
3. **R bridge refactoré** : la coercition de prédictions est sortie du template et **doit être fournie par le contributeur** comme fonction `coerce_predictions(model, raw_pred, X, task, classes)` dans son `r_bridge.R`. Le template ne gère plus que le marshalling Parquet et le dispatching `fit/predict`.
4. **Contrat sortie format B précisé** : `predict(X)` doit retourner `(n,)` dtype `int64` pour classification (labels) ou `(n,)` dtype `float64` pour régression. `predict_proba(X)` si présent doit retourner `(n, k)` dtype `float64` dans `[0, 1]` avec ordre des classes `classes_` aligné à `model.classes_`. Toute ambiguïté → refus.
5. **Tolérances harmonisées** : `sanity_check.tolerance` est un champ unique du manifest, borné par format : A `<= 1e-6`, B `<= 1e-6`, C `<= 1e-4`. Les exemples de manifest avaient `0.001` incohérent — corrigé.
6. **Canary tests anti-leakage** : pour B et C, le runtime ajoute deux canaris à la validation : (a) **label permutation test** (fit sur y permuté → score doit ne pas être anormalement bon) ; (b) **test set access audit** (le modèle ne doit jamais avoir lu un fichier dont le chemin contient `_test` ou `holdout`).
7. **Politique GPL opérationnelle** : tag `gpl_derived: true` héritée à tout artefact. **Export utilisateur d'un artefact `gpl_derived`** : autorisé mais le bundle inclut un fichier `LICENSE_OBLIGATIONS.md` listant les obligations (notice, copyleft, source disponibilité). Mode export `no_gpl` : exclut tous les artefacts taggués.
8. **Multi-target régression reportée en v0.2** pour B et C. v0.1 = univariable uniquement.
9. **R bridge versioning** : `bridge_template_version` dans le manifest C ; le runtime garde l'historique des templates ; une nouvelle version du template n'invalide pas les soumissions précédentes (elles continuent avec leur version).
10. **Cache d'installation explicite** : `pip install` et `R install.packages` s'exécutent **une fois par soumission** (au début), pas par run atomique. Le venv/container R est conservé pour toute la grille de 3 660 runs.

## 1. Pourquoi trois formats

Le format unique "pipeline nirs4all + `.n4a`" garantit la reproductibilité la plus forte, mais exige du contributeur qu'il (i) connaisse nirs4all, (ii) implémente sa méthode dans le DSL nirs4all, (iii) exporte un bundle. C'est une barrière inutile pour :
- Un auteur d'une méthode déjà publiée sous forme de **paquet Python `pip install`-able** avec une classe `sklearn`-compatible (`fit/predict`).
- Un auteur d'une méthode publiée sous forme de **paquet R** (CRAN, Bioconductor) — communauté chimiométrie *très* présente.

Les formats A/B/C couvrent les trois cas, du plus fort au plus souple :

| Format | Source du modèle | Reproductibilité | Effort contributeur | Effort runtime |
|---|---|---|---|---|
| **A** — nirs4all bundle | pipeline nirs4all + `.n4a` | ★★★★★ | élevé | faible |
| **B** — Python lib externe | `pip` package PyPI + sklearn API | ★★★★☆ | modéré | modéré (Docker) |
| **C** — R package | CRAN/Bioconductor + R function | ★★★☆☆ | faible | élevé (Docker + R) |

Tous les formats sont **traités identiquement par la grille canonique** : chaque soumission, quel que soit son format, traverse les 3 660 runs de la grille v0.1b.

## 2. Format A — bundle nirs4all (référence)

### 2.1 Contenu de la soumission

```
submission.tar.gz
├── pipeline.json   (alias manifest.json, format = "A")
├── bundle.n4a
├── sanity_check.csv
└── CITATION.cff
```

### 2.2 Validation côté runtime

1. Charger `bundle.n4a` via `nirs4all.pipeline.bundle.BundleLoader`.
2. Re-exécuter sur `sample_data/regression` (seed=0) ; comparer à `sanity_check.csv`. Tolérance `1e-6`.
3. Audit du pipeline (fit-on-train-only automatique sur AST nirs4all).

### 2.3 Exécution dans la grille

```
[ augmenter du run ] → [ PP du run ] → [ outlier_filter ] → [ target_processing ] →  [ pipeline contributeur ]
```

### 2.4 Garanties

- Reproductibilité bit-à-bit modulo non-déterminismes déclarés.
- EnvCard complète (versions, BLAS, threads, GPU determinism).
- Audit automatique des étapes (rejet si fit-global détecté).

## 3. Format B — paquet Python externe + sklearn API

### 3.1 Contenu de la soumission

```
submission.tar.gz
├── manifest.json   (format = "B")
├── sanity_check.csv
└── CITATION.cff
```

### 3.2 Schéma `manifest.json` (format B)

```json
{
  "schema_version": "1.0",
  "format": "B",
  "submission_id": "...",
  "author": {"name": "...", "email": "...", "orcid": "..."},
  "license": "MIT|Apache-2.0|GPL-3.0|...",
  "citation_cff_path": "CITATION.cff",
  "model": {
    "canonical_name": "my-model",
    "source": "pypi",
    "package_name": "my-package",
    "package_version_spec": "==1.2.3",
    "extras": ["gpu"],
    "import_path": "my_package.estimators",
    "class_name": "MyEstimator",
    "constructor_kwargs": {"n_estimators": 200, "alpha": 0.1},
    "task_types": ["regression", "classification"],
    "input_constraints": {"min_n": 20, "max_features": 4096, "signal_types": ["absorbance", "reflectance"]},
    "runtime_tier": "medium",
    "compute_constraints": {"requires_gpu": false, "min_ram_gb": 4},
    "sklearn_compatibility": {
      "fit_signature": "fit(X, y, sample_weight=None) -> self",
      "predict_signature": "predict(X) -> ndarray of shape (n,)",
      "predict_proba_available": false,
      "predict_proba_signature": null,
      "clone_compatible": true,
      "tested_with_sklearn_version": ">=1.5,<2.0"
    },
    "output_contract": {
      "regression": {"dtype": "float64", "shape": ["n"]},
      "classification": {
        "predict": {"dtype": "int64", "shape": ["n"], "semantics": "class indices aligned to model.classes_"},
        "predict_proba": {"dtype": "float64", "shape": ["n", "k"], "range": [0.0, 1.0], "semantics": "columns aligned to model.classes_"}
      }
    }
  },
  "hyperparameter_search": null,
  "sanity_check": {
    "dataset_alias": "sample_data/regression",
    "seed": 0,
    "expected_rmse": 0.45,
    "tolerance": 1e-6
  },
  "declaration": {
    "fit_on_train_only": true,
    "reproducibility_caveats": ["..."],
    "pretraining_disclosure": "trained from scratch — no external data",
    "external_data_used": []
  }
}
```

**Source policy v0.1** : `source` peut être uniquement `"pypi"`. Toute exception (`testpypi`, `github`, …) exige une entrée préalable dans une **allowlist éditoriale** avec :
- Hash de commit immuable.
- Audit éditorial archivé.
- Justification (paquet pas encore sur PyPI, version dev nécessaire pour fix critique, …).

### 3.3 Validation côté runtime

1. **Container Docker** `nirs4all-arena-py:v0.1` instancié pour la soumission. Image base : `python:3.11-slim` + outils de build minimaux + `pip`.
2. **Résolution du paquet** : `pip install <package_name><version_spec>` *à l'intérieur du container*. Échec → `failed_pip_resolve`.
3. **Import et instanciation** : `from <import_path> import <class_name>` ; `model = ClassName(**constructor_kwargs)`. Échec → `failed_dispatch`.
4. **Vérification API sklearn** :
   - `model.fit(X, y)` callable.
   - `model.predict(X)` callable, retourne ndarray de `shape == (n,)` et `dtype` matching `output_contract`.
   - `sklearn.base.clone(model)` réussit.
   - Si `predict_proba_available=true` : `model.predict_proba(X)` retourne `(n, k)`, ses sommes par ligne ≈ 1 (tolérance `1e-6`).
   - **Refus si shape ou dtype divergent.**
5. **Sanity check** : exécuter le modèle sur `sample_data/regression` (seed=0) via pipeline minimal `[StandardScaler, model]`. Comparer à `sanity_check.csv`, tolérance `1e-6`.
6. **Audit fit-on-train-only** : non-trivial sur un modèle externe — déclaration sur l'honneur + canary tests rétrospectifs (§3.5).

### 3.4 Exécution dans la grille

Le modèle externe est encapsulé dans un pipeline nirs4all minimal :

```python
# pseudo-code côté runtime
from <import_path> import <class_name>
ExternalModel = <class_name>

pipeline = nirs4all.PipelineConfigs([
    {"y_processing": <target_processing du run>},
    *<augmenter_steps du run>,
    *<pp_chain_steps du run>,
    *<outlier_filter_steps du run>,
    {"model": ExternalModel(**constructor_kwargs)},
])
```

Le pipeline est traité comme n'importe quel pipeline nirs4all. Le bundle `.n4a` produit est archivé.

### 3.5 Canary tests (anti fit-on-train-only violation)

Pour chaque soumission format B (et C), le runtime exécute deux canaris **avant** d'autoriser l'entrée dans la grille :

- **Canary 1 — Label permutation test** : fit le modèle sur `(X_train, permutation(y_train))` ; mesurer le score sur `(X_test, y_test)`. Un modèle fit-on-train-only doit produire un score ≈ baseline aléatoire. Si le score est ≥ médiane(scores observés sur la même classe de modèles), flag `under_review`.
- **Canary 2 — Filesystem access audit** : pendant `fit` et `predict`, le container est surveillé via `strace` / `ptrace` ou équivalent (option pour v0.2). En v0.1, vérification statique : le code source du paquet ne doit pas contenir de patterns d'accès à des chemins suspects (`_test`, `holdout`, chemins absolus vers dataset registry).

Les canaris ne sont **pas absolus** (un modèle adverse motivé peut les contourner). Ils filtrent les violations naïves ; les soumissions à score anormalement bon restent flaggées et auditées éditorialement.

### 3.6 EnvCard format-B

Inclut :
- `python_version`, `package_versions` (de `pip freeze --all` post-install — capturé comme `venv_lockfile.txt`), `blas_backend`, `thread_env`, `gpu_determinism`, `cuda_version` (si applicable).
- `submission_package_version_resolved` (peut différer du `package_version_spec` selon le résolveur).
- `submission_package_index_url` (PyPI ou allowlistée).
- `container_image_digest` (digest immuable de `nirs4all-arena-py:v0.1`).

### 3.7 Sécurité format B

Docker obligatoire dès v0.1. Politique :
- Image base immuable, digest pinné.
- **Pas d'accès réseau** dans le container sauf pendant `pip install` (réseau coupé après installation).
- Filesystem en lecture seule sauf `output_dir` (montage volume restreint).
- Pas de secrets exposés (env vars, montage `/etc`, etc.).
- Timeout strict global par run atomique (`runtime_tier_max`).
- L'installation se fait **une fois par soumission**, pas par run. Le container du même état de venv exécute tous les runs de la soumission.

## 4. Format C — paquet R

### 4.1 Contenu de la soumission

```
submission.tar.gz
├── manifest.json   (format = "C")
├── r_bridge.R      (inclut la fonction coerce_predictions)
├── sanity_check.csv
└── CITATION.cff
```

### 4.2 Schéma `manifest.json` (format C)

```json
{
  "schema_version": "1.0",
  "format": "C",
  "submission_id": "...",
  "author": {"name": "...", "email": "...", "orcid": "..."},
  "license": "GPL-3.0|MIT|...",
  "citation_cff_path": "CITATION.cff",
  "model": {
    "canonical_name": "my-r-model",
    "source": "cran",
    "r_package_name": "mdatools",
    "r_package_version_spec": ">=0.14.0",
    "fit_function": "pls",
    "predict_function": "predict",
    "coerce_predictions_function": "coerce_predictions",
    "fit_args": {"ncomp": 10, "cv": 1},
    "predict_args": {},
    "task_types": ["regression"],
    "input_constraints": {"min_n": 20, "max_features": 4096, "signal_types": ["absorbance", "reflectance"]},
    "runtime_tier": "medium",
    "compute_constraints": {"requires_gpu": false, "min_ram_gb": 4},
    "r_bridge_script": "r_bridge.R",
    "bridge_template_version": "v0.1"
  },
  "sanity_check": {
    "dataset_alias": "sample_data/regression",
    "seed": 0,
    "expected_rmse": 0.45,
    "tolerance": 1e-4
  },
  "declaration": {
    "fit_on_train_only": true,
    "reproducibility_caveats": ["R stochastic — set.seed propagated; cv internal to mdatools"],
    "pretraining_disclosure": "no pretraining",
    "external_data_used": []
  }
}
```

**Source policy v0.1** : `source` peut être uniquement `"cran"` ou `"bioconductor"`. GitHub (`devtools::install_github`) **interdit** en v0.1. Exceptions via **allowlist éditoriale** avec **commit SHA immuable** + audit du paquet.

### 4.3 Bridge R par défaut (template — marshalling minimal)

Le template ne contient plus de logique de coercition spécifique. Le contributeur **doit** fournir une fonction `coerce_predictions(model, raw_pred, X, task, classes)` dans `r_bridge.R`.

```r
#!/usr/bin/env Rscript
# r_bridge.R — template fourni par nirs4all_arena (bridge_template_version: v0.1).
#
# Le contributeur doit fournir la fonction coerce_predictions() au-dessus de ce template.
# Le template ne gère que le marshalling Parquet et le dispatching fit/predict.

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

# -------- FONCTION FOURNIE PAR LE CONTRIBUTEUR --------
# Doit retourner un vecteur numerique (regression) ou un facteur (classification)
# de longueur length(predict(model, X)). Le contributeur est responsable de
# coercer correctement (model.classes_, structures complexes, etc.).
#
# Si manifest$model$coerce_predictions_function != "coerce_predictions",
# le bridge appelle la fonction nommee dans ce champ.
#
# coerce_predictions <- function(model, raw_pred, X, task, classes) {
#   # ... contributeur ...
# }
# ------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]                                    # "fit" | "predict"
input_dir <- args[2]                               # X.parquet, y.parquet (fit), seed.txt
output_dir <- args[3]                              # model.rds (fit) | y_pred.parquet (predict)
manifest_path <- args[4]
r_bridge_user_path <- args[5]                      # path vers le r_bridge.R du contributeur

manifest <- fromJSON(manifest_path)
source(r_bridge_user_path)   # charge la fonction coerce_predictions du contributeur

pkg <- manifest$model$r_package_name
fit_fn_name <- manifest$model$fit_function
predict_fn_name <- manifest$model$predict_function
coerce_fn_name <- if (!is.null(manifest$model$coerce_predictions_function)) {
  manifest$model$coerce_predictions_function
} else {
  "coerce_predictions"
}
fit_args <- manifest$model$fit_args
predict_args <- manifest$model$predict_args
task <- manifest$model$task_types[[1]]

library(pkg, character.only = TRUE)

# Seed externe
seed <- as.integer(readLines(file.path(input_dir, "seed.txt")))
set.seed(seed)

X <- as.matrix(read_parquet(file.path(input_dir, "X.parquet")))

if (mode == "fit") {
  y <- read_parquet(file.path(input_dir, "y.parquet"))[[1]]
  if (task != "regression") y <- as.factor(y)

  fit_call <- c(list(X, y), fit_args)
  model <- do.call(fit_fn_name, fit_call)

  # Persistance du modèle
  saveRDS(model, file.path(output_dir, "model.rds"))
  # Capture des classes pour coerce_predictions en mode predict
  if (task != "regression") {
    saveRDS(levels(y), file.path(output_dir, "classes.rds"))
  }
} else if (mode == "predict") {
  model <- readRDS(file.path(output_dir, "model.rds"))
  classes <- if (task != "regression" && file.exists(file.path(output_dir, "classes.rds"))) {
    readRDS(file.path(output_dir, "classes.rds"))
  } else {
    NULL
  }
  predict_call <- c(list(model, X), predict_args)
  raw_pred <- do.call(predict_fn_name, predict_call)

  # COERCION par le contributeur (le bridge ne sait pas comment extraire la pred)
  y_pred <- do.call(coerce_fn_name, list(model = model, raw_pred = raw_pred,
                                         X = X, task = task, classes = classes))

  out <- data.frame(y_pred = as.numeric(y_pred))
  write_parquet(out, file.path(output_dir, "y_pred.parquet"))
} else {
  stop("mode must be 'fit' or 'predict'")
}
```

### 4.4 Bridge utilisateur — exemple `mdatools::pls`

Le contributeur écrit dans son `r_bridge.R` (par-dessus le template) :

```r
coerce_predictions <- function(model, raw_pred, X, task, classes) {
  # mdatools::pls : la prediction est un objet 'plsres' contenant y.pred (array 3D)
  if (task == "regression") {
    # y.pred[, 1, n_components] — derniere LV, premier output
    n_lv <- dim(raw_pred$y.pred)[3]
    return(as.numeric(raw_pred$y.pred[, 1, n_lv]))
  }
  stop("Only regression supported for this submission.")
}
```

Pour un autre paquet R (ex. `pls::predict.mvr` ou `caret::predict`), le contributeur fournit la fonction adaptée — c'est *son* travail, pas celui du template.

### 4.5 Validation côté runtime

1. **R disponible via Docker** : container `nirs4all-arena-r:v0.1` instancié pour la soumission. Image base `rocker/r-ver:4.3.x`, avec `arrow`, `jsonlite`, et les paquets canon (cf §11) pré-installés.
2. **Paquet R installé** : `Rscript -e 'requireNamespace("<pkg>")'`. Si absent, `install.packages(pkg, repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))`. CRAN whitelistée ; refus si source = `github` sans allowlist.
3. **Sanity check** : `r_bridge.R fit` puis `predict` sur `sample_data/regression` (seed=0). Compare au `sanity_check.csv`, tolérance `1e-4`.
4. **Version pinning** : lecture de `sessionInfo()` post-exécution ; enregistrement dans EnvCard.
5. **Canary tests** : mêmes deux canaris que format B (§3.5).

### 4.6 Exécution dans la grille — via controller `RModelController`

Implémenté dans `nirs4all_arena.controllers.r_model.RModelController` (esquisse de code dans la version précédente du doc, simplifiée ici). Le controller est enregistré via `@register_controller` et intercepte les étapes `{"model": {"format": "C", ...}}`.

L'installation R se fait **une fois par soumission** (au sanity check) ; le container est conservé pour la totalité de la grille de 3 660 runs.

### 4.7 Reproductibilité format C

- **R version pinning** via image Docker digest immuable.
- **Paquets pinning** via `sessionInfo()` capturée + DESCRIPTION du paquet pinned.
- **Non-déterminisme R** : `set.seed(seed)` propagée pour PRNG R. Certains paquets utilisent du C-level RNG (`randomForest::randomForest`, par exemple) — `set.seed` propagée selon paquet, *non garantie universellement*. Caveats déclarés dans EnvCard.
- **Tolérance sanity** : `1e-4` par défaut, peut être resserrée à `1e-6` si le contributeur démontre le déterminisme dans son bridge.

### 4.8 Sécurité format C

Docker obligatoire dès v0.1. Politique :
- Image base immuable, digest pinné.
- Bibliothèque R sandboxée (`R_LIBS_USER` pointe vers un volume dédié à la soumission, isolé du système).
- Réseau autorisé uniquement vers CRAN/Bioconductor (whitelist DNS ou proxy) pendant `install.packages`.
- Filesystem en lecture seule sauf `output_dir`.
- Timeout strict.

## 5. Tableau récapitulatif des trois formats

| Aspect | Format A | Format B | Format C |
|---|---|---|---|
| Manifest | pipeline.json (nirs4all-DSL) | manifest.json (format-B) | manifest.json (format-C) |
| Source modèle | bundle.n4a | PyPI (allowlist hors PyPI) | CRAN/Bioconductor (allowlist hors) |
| Sandbox v0.1 | aucun (nirs4all natif) | **Docker** | **Docker** |
| Validation | bundle loadable + sanity + audit | pip install + sklearn API check + sanity + canary | R install + bridge + sanity + canary |
| Exécution | direct via PipelineRunner | wrap dans pipeline minimal | controller `RModelController` |
| Cache installation | n/a | venv conservé pour 3 660 runs | container R conservé pour 3 660 runs |
| EnvCard | versions nirs4all, BLAS | + venv lockfile, package index, container digest | + R version, `sessionInfo()`, container digest |
| Bundle archivé | `.n4a` original | `.n4a` régénéré + venv_lockfile | `.n4a` régénéré + `model.rds` + `r_bridge.R` |
| Tolérance sanity | `1e-6` | `1e-6` | `1e-4` (resserrable à `1e-6`) |
| Audit fit-on-train-only | automatique (AST) | déclaration + canary | déclaration + canary |
| Multi-target régression | ✓ | reporté v0.2 | reporté v0.2 |
| Reproductibilité | ★★★★★ | ★★★★☆ | ★★★☆☆ |

## 6. Auto-détection du format à la soumission

Le runtime détecte le format via le champ `format ∈ {"A", "B", "C"}`. Le manifest est validé contre le JSON Schema correspondant (`manifest_schema_A.json`, `_B.json`, `_C.json`).

Pour format A, rétrocompatibilité : fichier nommé `pipeline.json` *ou* `manifest.json` avec `format = "A"`.

## 7. Impact sur la grille

**Aucun changement.** Les 3 660 runs canoniques sont identiques, seul l'invocateur du modèle change. Les conclusions sur (PP × modèle), (aug × modèle), etc. sont alimentées identiquement.

## 8. Restrictions par format

| Restriction | A | B | C |
|---|---|---|---|
| Modèle peut contenir son propre PP | ✓ | ✓ (déconseillé — pollue les blocs PP/aug/outliers) | ✓ (déconseillé) |
| Modèle GPU | ✓ | ✓ (declarer `requires_gpu`) | difficile (R + GPU = niche), à déclarer |
| Modèle stochastique | ✓ (seed propagée) | ✓ (si `random_state` accepté) | ✓ (set.seed dans bridge ; caveat C-level RNG) |
| Régression multi-target | ✓ | **reporté v0.2** | **reporté v0.2** |
| Classification multi-classe | ✓ | ✓ (output contract §3.2) | ✓ (factor → coerce dans le bridge) |
| Probabilités calibrées | ✓ si `predict_proba` exposée | ✓ si exposée + contrat output | ✓ si la fonction R les retourne |

## 9. Tests d'acceptation v0.1 — par format

Avant ouverture publique, le runtime doit avoir traité au moins une soumission par format **par lui-même** :

- **A** : PLS-canon v0.1b (déjà spec) → traverse la grille.
- **B** : wrapper de `sklearn.linear_model.Lasso` empaqueté `lasso_canon_b` publié sur PyPI test (en allowlist exception) → traverse la grille.
- **C** : `mdatools::pls` empaqueté avec son `r_bridge.R` (incluant `coerce_predictions`) → traverse la grille.

Les trois sanity-checks passent ; les trois canaris (label permutation, FS audit) passent ; les trois soumissions sont indexées et publiables.

## 10. Évolutions v0.2+

- **Format D — Notebook Jupyter** reproductible (papermill).
- **Format E — Endpoint HTTP** (modèle servi par le contributeur). Sécurité élevée. Reporté.
- **Format F — Container Docker** complet fourni par le contributeur. Maximum d'isolation, maximum de barrière. v0.2.
- **rpy2** comme alternative à subprocess pour format C (performance).
- **Multi-target régression** pour formats B et C.
- **Strace / ptrace** FS access audit pour canaris (au lieu d'audit statique).

## 11. Risques et limites — v0.1c

**R-A1** — Code arbitraire au build/install (B/C). *Mitigation v0.1c* : Docker dès v0.1 pour B et C, image base immuable, network restreint, FS read-only. Le risque résiduel est l'évasion de container — peu plausible avec un attaquant non-motivé.

**R-A2** — Reproductibilité dégradée (B/C). Versions transitives peuvent dériver. *Mitigation* : EnvCard exhaustive + re-vérification périodique des baselines.

**R-A3** — Format C exige R installé. *Mitigation* : image Docker `nirs4all-arena-r` maintenue centralement.

**R-A4** — Fit-on-train-only violation. *Mitigation v0.1c* : canary tests (label permutation + FS audit) ajoutés. Pas infaillibles ; audit éditorial sur soumissions au-dessus du percentile 99.

**R-A5** — License GPL contamination. *Mitigation v0.1c* : politique opérationnelle :
- Métadonnée `gpl_derived: true` héritée à tout bundle.
- Export utilisateur du bundle inclut `LICENSE_OBLIGATIONS.md` (notice, copyleft, source disponibilité).
- Mode export `no_gpl` exclut les artefacts taggués.
- Le leaderboard public n'est *jamais* bloqué par la licence ; seul l'export d'artefact est encadré.

**R-A6** — Timeout d'installation disproportionné. *Mitigation v0.1c* : `pip install` / `install.packages` sont *hors* du timeout `runtime_tier_max`. Comptabilisés séparément (`install_time_s`). Installation par soumission, pas par run.

**R-A7** *(nouveau)* — R bridge fragile aux paquets non-`mdatools`. *Mitigation v0.1c* : la coercition est responsabilité du contributeur ; le template ne gère que le marshalling.

**R-A8** *(nouveau)* — Multi-target régression non couvert en v0.1 pour B/C. *Mitigation* : reporté v0.2 ; en v0.1, refus à l'admission si la soumission déclare `multi_target = true` pour B ou C.

**R-A9** *(nouveau)* — Versioning bridge R. *Mitigation v0.1c* : `bridge_template_version` dans le manifest C. Les soumissions exécutées sous `v0.1` restent figées ; un bump du template ne re-exécute pas automatiquement.

## 12. Coordination avec la cartographie des datasets

Aspects critiques pour B/C :
- **Datasets multi-source** (`n_sources > 1`) : le runtime expose `X` comme matrice 2D `(n_samples, n_features_total)` *ou* refuse l'exécution si la soumission ne le supporte pas. Décision v0.1 : le runtime concatène les sources sur l'axe features avant marshalling Parquet, et l'EnvCard logue cette opération.
- **Datasets avec répétitions** (`sample_id` groupes) : le split honore les groupes via les masques pré-calculés. Les formats B/C reçoivent `(X_train, y_train)` déjà groupé correctement — pas leur responsabilité.
- **Datasets multi-target** : refus à l'admission pour B/C (v0.1) — cf R-A8.

## 13. Implications sur le runtime sketch

Toutes intégrées dans [04_runtime_sketch.md](04_runtime_sketch.md) §3.1, §3.4, §9, §11 (v0.1b patch).

## 14. Livrables

- `manifest_schema_A.json`, `manifest_schema_B.json`, `manifest_schema_C.json` (JSON Schemas v1.0 versionnés).
- `r_bridge_template_v0.1.R` (template par défaut).
- `nirs4all_arena.controllers.r_model.RModelController`.
- `nirs4all_arena.runtime.executors.{format_a, format_b, format_c}`.
- Tests d'acceptation : un sanity par format + un canary fail-test par format dans `tests/acceptance/`.
- Docker images : `nirs4all-arena-py:v0.1`, `nirs4all-arena-r:v0.1`.
- `LICENSE_OBLIGATIONS.md` (template pour mode export `gpl_derived`).
- Allowlist éditoriale : `editorial_allowlists.json` (sources hors PyPI/CRAN/Bioconductor autorisées avec hash commit).
