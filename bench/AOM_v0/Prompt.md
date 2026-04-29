# Prompt maître — AOM_v0 Operator-Adaptive PLS

Tu es Claude Code, agent principal d'implémentation. Tu dois livrer en une
exécution un projet scientifique complet dans `bench/AOM_v0` pour
Operator-Adaptive PLS, incluant AOM-PLS, POP-PLS, NIPALS, SIMPLS, PLS-DA,
backends NumPy et Torch, benchmarks contre `master_results.csv`, et dossier de
publication complet.

Le rôle attendu est :

- Claude code.
- Codex vérifie comme reviewer externe aux checkpoints définis.
- Claude reste responsable de l'intégration, des tests et des livrables.

Ne demande pas de choix utilisateur. Si un choix est nécessaire, applique les
defaults définis ici et documente la décision dans
`docs/AOMPLS_IMPLEMENTATION_LOG.md`.

## 0. Répertoire et limites

Répertoire de travail obligatoire :

```text
bench/AOM_v0
```

Ne modifie pas le package de production `nirs4all/` pendant cette passe. Tu peux
le lire pour référence. Toute nouvelle implémentation doit vivre sous
`bench/AOM_v0/aompls`.

Lis d'abord ces documents, dans cet ordre :

1. `README.md`
2. `docs/CONTEXT_REVIEW.md`
3. `docs/AOMPLS_MATH_SPEC.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/BENCHMARK_PROTOCOL.md`
6. `docs/PUBLICATION_REPO_PLAN.md`
7. `publication/manuscript/PAPER_DRAFT.md`
8. `source_materials/AOM/PLAN_V1.md`
9. `source_materials/AOM/ROADMAP.md`
10. `source_materials/AOM/PUBLICATION_PLAN.md`
11. `source_materials/AOM/PUBLICATION_BACKLOG.md`
12. `source_materials/AOM/report.md`
13. `source_materials/AOM/advanced_architectures.md`
14. `source_materials/fck_pls/FCK_PLS_README.md`
15. `source_materials/tabpfn/SPECTRAL_LATENT_FEATURES.md`
16. `source_materials/tabpfn/MASTER_RESULTS_PROFILE.md`
17. `source_materials/tabpfn/TABPFN_PAPER_PROTOCOL_NOTES.md`

Lis aussi, en read-only, les références existantes :

```text
nirs4all/operators/models/sklearn/aom_pls.py
nirs4all/operators/models/sklearn/pop_pls.py
nirs4all/operators/models/sklearn/aom_pls_classifier.py
nirs4all/operators/models/sklearn/pop_pls_classifier.py
nirs4all/operators/models/sklearn/plsda.py
nirs4all/operators/models/sklearn/simpls.py
nirs4all/operators/models/pytorch/aom_pls.py
bench/fck_pls/fckpls_torch.py
bench/_tabpfn/search_space.py
bench/_tabpfn/spectral_latent_features.py
bench/tabpfn_paper/run_reg_aom.py
bench/tabpfn_paper/run_reg_pls.py
bench/tabpfn_paper/run_reg_tabpfn.py
bench/tabpfn_paper/master_results.csv
bench/tabpfn_paper/data/DatabaseDetail.xlsx
bench/tabpfn_paper/metrics.xlsx
```

## 1. Résultat final attendu

À la fin, `bench/AOM_v0` doit contenir :

1. Une implémentation Python importable `aompls`.
2. Des estimators sklearn-like pour régression et classification.
3. Des moteurs NumPy et Torch.
4. Une suite pytest complète.
5. Des générateurs synthétiques déterministes.
6. Un benchmark smoke exécuté.
7. Un benchmark full prêt, resumable, compatible avec `master_results.csv`.
8. Des cohortes regression/classification construites ou des fichiers skip
   détaillés.
9. Une documentation math/API/validation.
10. Des prompts/reviews Codex.
11. Un repo de publication sous `publication/`.
12. Un manuscrit complet compilable ou, si LaTeX n'est pas disponible, un
    `main.tex` complet plus un log indiquant la commande de compilation non
    disponible.

## 2. Architecture obligatoire

Crée exactement cette structure si elle n'existe pas :

```text
bench/AOM_v0/
  aompls/
    __init__.py
    operators.py
    banks.py
    preprocessing.py
    centering.py
    nipals.py
    simpls.py
    scorers.py
    selection.py
    estimators.py
    classification.py
    torch_backend.py
    synthetic.py
    metrics.py
    diagnostics.py
  benchmarks/
    build_cohorts.py
    run_aompls_benchmark.py
    summarize_results.py
    run_smoke_benchmark.py
  tests/
    test_operators.py
    test_nipals.py
    test_simpls.py
    test_estimators.py
    test_selection.py
    test_classification.py
    test_torch_parity.py
    test_benchmark_schema.py
  docs/
    AOMPLS_IMPLEMENTATION_LOG.md
    AOMPLS_API.md
    AOMPLS_VALIDATION.md
    CODEX_REVIEWS.md
  publication/
    manuscript/
      main.tex
      references.bib
    supplement/
    figures/
    tables/
    scripts/
    arxiv/
    journal/
```

## 3. Math et conventions non négociables

Utilise toujours :

```text
X in R^{n x p}
Y in R^{n x q}
X_b = X A_b^T
X_b^T Y = A_b X^T Y
```

Tout opérateur strictement linéaire doit fournir :

- `fit(X, y=None)`
- `transform(X)` pour `X A^T`
- `apply_cov(S)` pour `A S`
- `adjoint_vec(v)` pour `A^T v`
- `matrix(p)` pour petits tests
- `is_linear_at_apply()`
- `fitted_parameters()`

SNV, MSC, EMSC, OSC et EPO ne sont pas des opérateurs stricts fixes par défaut.
Ils ne doivent pas entrer dans les tests de covariance SIMPLS stricte sauf
preuve explicite. S'ils sont inclus, ils le sont comme preprocessors ou wrappers
expérimentaux fitted linear-at-apply, fit uniquement sur train folds.

Les coefficients finaux prédisent depuis l'espace original :

```text
Y_hat = (X - x_mean) @ coef_ + intercept_
B = Z (P^T Z)^+ Q^T
```

`Z` est toujours exposé comme `x_effective_weights_`.

## 4. Variantes à implémenter

### Régression

Classes publiques :

- `AOMPLSRegressor`
- `POPPLSRegressor`

Politiques :

- `selection="none"`
- `selection="global"`
- `selection="per_component"`
- `selection="soft"` expérimental
- `selection="superblock"` expérimental/baseline

Engines NumPy obligatoires :

- `pls_standard`
- `nipals_materialized`
- `nipals_adjoint`
- `simpls_materialized`
- `simpls_covariance`
- `superblock_simpls`

Engines Torch obligatoires :

- `nipals_adjoint`
- `simpls_covariance`
- `superblock_simpls`

Critères :

- `covariance`
- `cv`
- `approx_press`
- `hybrid`
- `holdout` legacy uniquement

N'implémente `press` sans préfixe que si la formule est validée par tests. Sinon
nomme explicitement `approx_press`.

### Classification

Classes publiques :

- `AOMPLSDAClassifier`
- `POPPLSDAClassifier`

Implémente une vraie version classification, pas seulement un wrapper minimal :

1. Encode labels en one-hot class-balanced :
   `Y_ic = 1 / sqrt(pi_c)` si `y_i = c`, sinon 0.
2. Centre `Y`.
3. Fit AOM/POP comme PLS2 discriminant.
4. Transforme `X` en scores latents.
5. Fit `LogisticRegression(class_weight="balanced", max_iter=2000)` sur les
   scores latents.
6. `predict_proba` utilise ce calibrateur.
7. Si la calibration logistique échoue, fallback déterministe :
   temperature-scaled softmax fit sur train uniquement.

Metrics classification :

- balanced accuracy,
- macro-F1,
- log loss,
- Brier score binaire,
- expected calibration error.

## 5. Defaults imposés

Utilise ces defaults :

```python
random_state = 0
center = True
scale = False
n_components = "auto"
max_components = min(25, n_samples - 1, n_features)
engine = "simpls_covariance"
selection = "global"       # AOM
selection = "per_component" # POP
criterion = "cv"
cv = 5
orthogonalization = "auto"
```

Résolution de `orthogonalization="auto"` :

- `transformed` pour `selection in {"none", "global"}`.
- `original` pour `selection="per_component"`.
- `original` pour classification POP.

Torch doit tourner sur CPU si CUDA est absent.

## 6. Tests obligatoires

Crée et fais passer les tests suivants :

### Opérateurs

- shape transform,
- linéarité,
- adjoint `<A x, y> = <x, A^T y>`,
- covariance `(X A^T)^T Y = A X^T Y`,
- matrice explicite vs `transform/apply_cov/adjoint_vec`,
- compositions.

### PLS et équivalences

- PLS1, PLS2 shapes,
- identity-only AOM = PLS standard,
- single fixed operator materialized = covariance/adjoint fast en
  `orthogonalization="transformed"`,
- fixed POP sequence materialized = fast,
- NIPALS convention `X_b^T y = A_b X^T y`,
- coefficients prédictifs sur `X` original.

### Sélection

- global retourne un opérateur unique,
- per-component retourne une séquence de longueur `n_components_`,
- scores de tous les opérateurs stockés,
- `n_components="auto"` valide,
- `max_components` respecté,
- critères CV sans fuite.

### Classification

- binaire et multiclasses,
- `predict`, `predict_proba`,
- probabilités bornées et somme à 1,
- class imbalance,
- calibration fit uniquement sur train fold,
- metrics balanced accuracy/log loss/ECE.

### Torch

- import sans CUDA,
- parity NumPy/Torch identity-only,
- parity NumPy/Torch petit opérateur explicite,
- pas de NaN float32/float64.

## 7. Benchmark obligatoire

Lis `docs/BENCHMARK_PROTOCOL.md` et implémente exactement :

- `benchmarks/build_cohorts.py`
- `benchmarks/run_aompls_benchmark.py`
- `benchmarks/run_smoke_benchmark.py`
- `benchmarks/summarize_results.py`

Regression :

- Utilise `bench/tabpfn_paper/master_results.csv`.
- Construis `benchmarks/cohort_regression.csv` depuis les 61 splits.
- Écris des résultats compatibles avec le schéma de `master_results.csv`.
- Ajoute les colonnes AOM diagnostics définies dans le protocole.
- Compare aux modèles PLS, TabPFN-Raw et TabPFN-opt.

Classification :

- Scanne `bench/tabpfn_paper/data/classification`.
- Construis `benchmarks/cohort_classification.csv`.
- Si un dataset est illisible, ne le supprime pas silencieusement : écris une
  ligne `status="skipped"` et une raison.
- Compare PLS-DA, AOM-PLS-DA et POP-PLS-DA.

Exécute au minimum :

```bash
PYTHONPATH=bench/AOM_v0 pytest bench/AOM_v0/tests -q
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_smoke_benchmark.py
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/summarize_results.py \
  --results bench/AOM_v0/benchmark_runs/smoke/results.csv \
  --master bench/tabpfn_paper/master_results.csv \
  --out bench/AOM_v0/publication/tables
```

Règle déterministe pour le full benchmark :

- Lance le full benchmark si l'environnement local le permet sans blocage
  prévisible.
- Si une dépendance dataset/runner manque ou si la durée attendue dépasse 8h,
  n'échoue pas le projet : écris les commandes exactes de reprise dans
  `docs/AOMPLS_VALIDATION.md` et vérifie que le runner est resumable.

## 8. Publication obligatoire

Lis `docs/PUBLICATION_REPO_PLAN.md`.

Crée :

```text
publication/manuscript/main.tex
publication/manuscript/references.bib
publication/supplement/supplement.tex
publication/scripts/make_figures.py
publication/scripts/make_tables.py
publication/scripts/export_arxiv.sh
publication/scripts/check_submission.sh
publication/journal/cover_letter.md
publication/journal/highlights.md
publication/arxiv/README.md
```

Journal principal imposé :

```text
Chemometrics and Intelligent Laboratory Systems
```

Preprint imposé :

```text
arXiv avant soumission journal, avec disclosure dans cover letter.
```

Le papier doit être rédigé intégralement, pas seulement un plan. Les sections
obligatoires :

1. Abstract.
2. Introduction.
3. Related work.
4. Operator-Adaptive PLS framework.
5. Algorithms.
6. Classification and probability calibration.
7. Experimental protocol.
8. Regression benchmark results.
9. Classification benchmark results.
10. Ablations and equivalence validation.
11. Discussion.
12. Limitations.
13. Reproducibility statement.
14. Conclusion.

Si les résultats full ne sont pas disponibles, le manuscrit doit intégrer les
résultats smoke clairement marqués comme validation technique et laisser les
tables full générées automatiquement par `make_tables.py` lorsqu'elles seront
présentes. Ne fais aucune claim de supériorité full sans résultats full.

## 9. Protocole Codex

Cherche d'abord si un CLI Codex est disponible :

```bash
codex --version
```

Si Codex est disponible, demande une review aux checkpoints suivants et colle
les réponses dans `docs/CODEX_REVIEWS.md` :

1. Après design math + squelettes, avec
   `docs/codex_review_prompts/math_review.md`.
2. Après implémentation des cores, avec
   `docs/codex_review_prompts/code_review.md`.
3. Après tests, avec `docs/codex_review_prompts/test_review.md`.
4. Après manuscrit, avec
   `docs/codex_review_prompts/publication_review.md`.

Si Codex CLI n'est pas disponible, ne bloque pas. Les prompts existent déjà :
copie leur présence dans `docs/CODEX_REVIEWS.md` et continue.

## 10. Ordre de travail

Procède strictement par phases :

### Phase 0 — Inspection

- Inspecte repo et dépendances.
- Écris `docs/AOMPLS_IMPLEMENTATION_LOG.md`.
- Note versions Python, numpy, scipy, sklearn, torch si disponible.

### Phase 1 — Operators

- Implémente operators et banks.
- Tests opérateurs verts.

### Phase 2 — Références lentes

- PLS standard.
- NIPALS materialized.
- SIMPLS materialized.
- Tests PLS1/PLS2.

### Phase 3 — Fast engines

- NIPALS adjoint.
- SIMPLS covariance.
- Tests équivalences.

### Phase 4 — Selection unifiée

- No/global/per_component/soft/superblock.
- Criteria covariance/cv/approx_press/hybrid/holdout legacy.
- Diagnostics complets.

### Phase 5 — API sklearn

- `AOMPLSRegressor`.
- `POPPLSRegressor`.
- `get_params/set_params`.
- `predict/transform`.

### Phase 6 — Classification

- `AOMPLSDAClassifier`.
- `POPPLSDAClassifier`.
- Class-balanced coding.
- Probability calibration.
- Classification tests.

### Phase 7 — Torch

- Torch backend.
- CPU fallback.
- NumPy/Torch parity tests.

### Phase 8 — Benchmarks

- Cohorts.
- Smoke benchmark exécuté.
- Full benchmark resumable.
- Summary tables.

### Phase 9 — Documentation

- `docs/AOMPLS_API.md`.
- `docs/AOMPLS_VALIDATION.md`.
- Update implementation log.

### Phase 10 — Publication

- Manuscript full.
- Supplement.
- Figures/tables scripts.
- arXiv export.
- Journal files.

### Phase 11 — Final verification

Exécute :

```bash
PYTHONPATH=bench/AOM_v0 python -m compileall bench/AOM_v0/aompls
PYTHONPATH=bench/AOM_v0 pytest bench/AOM_v0/tests -q
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_smoke_benchmark.py
```

Si une commande échoue, corrige. Si elle échoue pour une dépendance externe
absente, documente précisément la dépendance et vérifie le reste.

## 11. Definition of Done

Le travail est terminé seulement si :

1. Les tests AOM_v0 passent ou les seuls échecs restants sont des dépendances
   externes documentées.
2. Les variantes principales fonctionnent :
   - PLS standard,
   - AOM global NIPALS,
   - POP NIPALS,
   - AOM global SIMPLS,
   - POP SIMPLS,
   - Superblock SIMPLS,
   - AOM/POP PLS-DA.
3. NumPy et Torch existent.
4. PLS1 et PLS2 sont testés.
5. Les opérateurs sélectionnés sont inspectables.
6. Les coefficients prédisent depuis `X` original.
7. Les équivalences identity et single-operator passent.
8. Les critères CV ne fuient pas.
9. Smoke benchmark exécuté et résultats écrits.
10. Full benchmark prêt et resumable.
11. Paper repo complet.
12. `main.tex` complet existe.
13. arXiv and journal files existent.
14. `docs/CODEX_REVIEWS.md` contient les reviews Codex ou la trace des prompts
    prêts à utiliser.

## 12. Résumé final à produire

À la fin, réponds avec :

- fichiers créés/modifiés,
- variantes implémentées,
- tests exécutés et résultats,
- benchmark smoke observé,
- statut du full benchmark,
- statut Codex,
- statut du papier/arXiv/journal,
- limites restantes,
- commandes exactes pour relancer tests et benchmarks.

Commence maintenant par la Phase 0.
