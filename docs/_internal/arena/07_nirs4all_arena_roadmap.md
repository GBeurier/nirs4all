# Feuille de route du projet `nirs4all_arena`

> Plan de mise en œuvre du protocole **depuis la base de preuves interne jusqu'à l'Arena publique et la publication méthodologique**. Le court terme n'est pas un leaderboard : c'est une machine de criblage pour l'equipe. Objectif operationnel : passer au minimum 10 datasets par mois, accumuler des preuves sur les recettes `(preprocessing, modele, hyperparams)`, identifier les anti-patterns, et recommander une trousse a outils NIRS. L'Arena est le website qui publie ensuite une partie de cette base. Si l'effort tient 2 ans, avec ≈300 datasets, ≈300 recettes/methodes, des collegues contributeurs et une analyse approfondie, le corpus peut devenir un papier methodologique ambitieux.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md), [00_arena_concept_storage.md](00_arena_concept_storage.md), [04_runtime_sketch.md](04_runtime_sketch.md), [06_submission_formats.md](06_submission_formats.md) |
| Version | v0.4 (recentrage base de preuves interne + Arena website) |
| Statut | À soumettre à seconde revue avant freeze |
| Phase courante au moment de l'écriture | **Phase 0a — Conception**, en cours |
| Prochaine phase | **Phase 0b — Bootstrap repo** dès validation de la conception |

## 0. Changelog v0.3 → v0.4

1. **Court terme clarifie** : le produit initial est une base de reference pour cribler les methodes dans l'equipe, pas le site public.
2. **Arena redefinie** : Arena designe le website public et les vues de publication construites au-dessus de la base de preuves.
3. **Evidence cards ajoutees** : les sorties prioritaires sont des fiches recettes/methodes, recommandations et anti-patterns.
4. **Cadence explicite** : objectif minimal de 10 datasets/mois ; trajectoire longue ≈300 datasets et ≈300 methodes/recettes.
5. **Ambition publication reformulee** : Nature Methods devient credible seulement comme synthese de 2 ans de donnees, contributions et analyses, pas comme promesse de v0.1.

## 0.1 Changelog v0.2b → v0.3

1. **Finalité explicitée** : l'arène produit à la fois un support de comparaison communautaire et un dataset de runs pour analyser/sélectionner les méthodes.
2. **Pivot stockage** : abandon du couple "base analytique séparée + extension `WorkspaceStore`". Le schema actif est `WorkspaceStore` SQLite existant + sidecar `arena.sqlite`.
3. **Impact `nirs4all` minimisé** : v0.1 ne doit pas ajouter de tables arena au coeur ; `nirs4all_arena` lit/écrit des workspaces standard et garde ses propres tables protocolaires.
4. **Registry datasets circonscrite** : `datasets_registry_v0.1.sqlite` ou tables dataset dans `arena.sqlite`, avec split masks Parquet versionnés.
5. **Idempotence déplacée** : `run_bindings(submission_id, run_spec_hash, workspace_uri, run_id, pipeline_id, prediction_id)` devient le point de reprise, pas une contrainte dans `WorkspaceStore`.

## 0.2 Changelog v0.1 → v0.2b (post-revue Codex)

1. **Numérotation des sous-sections corrigée** : Codex a flaggé `§4 contient 3.1, §5 contient 4.1`, etc. — toutes les sous-sections sont renumérotées en cohérence avec leur section parent.
2. **Périmètre v0.1 documenté** : nouveau document `SCOPE_v0.1.md` (à créer en Phase 0b) sépare *must-have* vs *backlog v0.2* par phase, avec budget jours-personnes par livrable. Règle : tout livrable sans budget ni owner sort de v0.1.
3. **Phase 1 découpée en 3 lots** : 8 datasets "smoke" → 18 restants → pool `all`. Gate Phase 1 = 8 DatasetCards validées avant Phase 2.
4. **Phase 2 enrichie d'un "dispatch spike" B/C** : dummy Python estimator + dummy R no-op traversent 2 datasets × 2 splits sans entrer au leaderboard. Valide manifeste parsing, isolation Docker, EnvCard format-aware avant l'implémentation complète Phase 3.
5. **Gates de transition phase X → X+1 explicites** : chaque phase a maintenant un **gate no-go** avec critères auditables, responsable, artefact attendu, date.
6. **Phase 5 à deux niveaux** : "bêta minimale" (≥ 5 labos / ≥ 10 soumissions) vs "paper-ready" (≥ 30 soumissions, ≥ 60 datasets `all`). Nature Methods reporté de v0.1 à *paper-ready*, qui peut glisser en v0.2 selon traction.
7. **`nirs4all` pin strict** : `nirs4all==0.8.11` pour v0.1 (pas `>=0.8.11`). ADR `API_CONTRACT_NIRS4ALL.md` en Phase 2 + matrice CI testant pin + branche dev.
8. **Compute budget par phase** : nouvelle section §12 estimant CPU/GPU-heures et stockage par phase.
9. **Parties prenantes** : nouvelle section §14 listant les rôles (mainteneur, reviewer manuscrit, arbitre dataset, …).
10. **Plan communication** : nouvelle section §15 (blog, mailing-list, conférences).
11. **Pivot cost stockage analytique** : historique ; remplacé en v0.3 par le choix SQLite sidecar.

## 1. Positionnement du repo `nirs4all_arena`

```
github.com/<owner>/
├── nirs4all/                  # bibliothèque NIRS (cur. v0.8.11)
├── nirs4all_arena/            # NOUVEAU — base de preuves + arena.sqlite + runtime + web
├── nirs4all-datasets-registry/  # NOUVEAU — registry datasets curé (séparé pour licences)
└── nirs4all_webapp/           # webapp existante (workspace UI, indépendant de l'arène)
```

### 1.1 Périmètre de `nirs4all_arena`

- **Base de preuves autour de nirs4all** : exécuter dans des workspaces standard (`store.sqlite`, `arrays/`, `artifacts/`) et maintenir une base annexe `arena.sqlite` pour `recipe_definitions`, `submissions`, `run_specs`, `run_bindings`, `env_cards`, `seed_cards`, `verifications`, `evidence_snapshots`, `citations`, `publication_snapshots`.
- **Controllers** étendant ceux de nirs4all : notamment `RModelController` (format C), `ExternalSklearnController` (format B), `RuntimeAccountingController` (mesure temps/mémoire/EnvCard).
- **Baselines** (PLS-canon, Ridge-canon, RF-canon) implémentées comme un sous-paquet `nirs4all_arena.baselines`.
- **Grille canonique** : code générateur (`nirs4all_arena.protocol.grid_v01b`), fixtures de tests, exports JSON.
- **Runtime central** : Submission Receiver, Grid Expander, Scheduler, Executor (CPU + GPU), Persister, ReproducibilityVerifier, Indexer + Notifier.
- **Evidence engine** : generation de recipe cards, recommandations, anti-patterns et matrices internes de decision.
- **Arena website** : génération des 4 vues critiques publiques (Option A statique, cf [05_web_minimal_sketch.md](05_web_minimal_sketch.md)).
- **CLI** : `arena submit`, `arena run`, `arena audit`, `arena replay`, `arena evidence`, `arena freeze`.

### 1.2 Hors-périmètre `nirs4all_arena` (reste dans `nirs4all`)

- Loaders de données, transforms, splitters, augmenters, models, generators, pipeline runner.
- `SpectroDataset`, `RunResult`, `BundleGenerator/BundleLoader`.
- UI workspace (`nirs4all_webapp`).
- Schema interne `WorkspaceStore` de `nirs4all` : l'arène ne doit pas y ajouter de tables spécifiques.

### 1.3 Dépendances

`nirs4all_arena` dépend de :
- **`nirs4all==0.8.11`** (pin strict pour v0.1 ; révisable à `>=0.8.11,<0.9` une fois `0.8.12+` validé).
- `sqlite3` (stdlib), `pyarrow`, `pandas` ou `polars`, `joblib`, `tqdm`.
- `cffconvert` (validation citation).
- `docker` (sandbox formats B et C).
- (Format C) `R >= 4.3` + paquets `arrow`, `jsonlite` (dans le container).
- `rpy2` *optionnel* (alternative au subprocess R, v0.2).

## 2. Phases — vue d'ensemble

```
                                                            chemin critique
Phase 0a — Conception                      ┃ T-1m → T0      ╔══════════════╗
                                           ┃                ║ design docs  ║
Phase 0b — Bootstrap repo                  ┃ T0   → T+1.5m  ║ + ADRs       ║
Phase 1  — Registry + cartographie         ┃ T+1.5m → T+3m  ║              ║
Phase 2  — Baselines + runtime alpha + spike B/C ┃ T+3m  → T+5m   ║ runtime      ║
Phase 3  — Formats B et C complets         ┃ T+5m  → T+7m   ║              ║
Phase 4  — Evidence reports + Web v0.1      ┃ T+7m  → T+9m   ╚══════════════╝
Phase 5  — Bêta minimale → paper-ready     ┃ T+9m  → T+12m
Phase 6+ — Stabilisation + publication     ┃ T+12m → T+18m
```

**Convention temporelle** : `T0` = jour d'ouverture du repo `nirs4all_arena`. Tout ce qui est antérieur (`T-Xm`) est la phase de conception.

**Charge estimée** : ≈ 0.7-0.8 ETP soutenu sur T0 → T+12m. **À valider impérativement avec le document `SCOPE_v0.1.md`** par lot must-have/nice-to-have (cf §0). Sans budget par livrable, le risque RT1 (mainteneur unique surchargé) reste critique.

**Précédents pour comparaison de charge** (à valider) : MoleculeNet, Open Graph Benchmark, ChemBench, HELM — équipes multi-personnes sur 1-2 ans pour des périmètres comparables. Notre planning à 0.7-0.8 ETP est ambitieux ; le périmètre doit donc être *strictement* réduit aux livrables critiques.

## 3. Phase 0a — Conception (T-1m → T0, **en cours**)

### 3.1 Objectif

Produire un protocole de benchmarking suffisamment **précis et défendable** pour que :
- la mise en œuvre (Phases suivantes) ne soit pas un terrain d'arbitrages ad hoc ;
- la communauté NIRS puisse en lire la spécification avant le repo, et formuler ses retours ;
- la première campagne (Phase 2) puisse exécuter exactement ce qui est écrit.

### 3.2 Livrables (état au moment d'écrire cette section)

| Livrable | Doc | Statut |
|---|---|---|
| Manifeste de protocole | [`systematic_benchmarking_protocol.md`](../systematic_benchmarking_protocol.md) | rédigé, 2 revues Codex MAJOR_REVISIONS intégrées |
| Plan de cartographie des datasets | [`01_dataset_cartography_tasks.md`](01_dataset_cartography_tasks.md) | rédigé, en attente d'exécution Phase 1 |
| Définition opérationnelle des baselines | [`02_pls_canon.md`](02_pls_canon.md) | rédigé v0.1b, revue Codex MAJOR_REVISIONS intégrée |
| Grille canonique v0.1b + structure d'aliasing | [`03_canonical_grid_v0.1.md`](03_canonical_grid_v0.1.md) | rédigé v0.1b, revue Codex MAJOR_REVISIONS intégrée |
| Architecture runtime minimal | [`04_runtime_sketch.md`](04_runtime_sketch.md) | rédigé, **non encore revu par Codex** |
| Vues web critiques v0.1 | [`05_web_minimal_sketch.md`](05_web_minimal_sketch.md) | rédigé, **non encore revu par Codex** |
| Formats de soumission A/B/C + R controller | [`06_submission_formats.md`](06_submission_formats.md) | rédigé v0.1c, revue Codex MAJOR_REVISIONS intégrée |
| Cette feuille de route | [`07_nirs4all_arena_roadmap.md`](07_nirs4all_arena_roadmap.md) | rédigé v0.2b, revue Codex MAJOR_REVISIONS intégrée |

### 3.3 Décisions structurantes prises en Phase 0a

1. **Vision base de preuves** plutôt que reconstruction de l'évidence empirique du master CSV `bench/`.
2. **Soumission hybride** : pipeline + bundle, re-exécution centrale obligatoire.
3. **Pool ouvert + sous-ensemble `selected` curé** : double vue rankings.
4. **Plan factoriel fractionnaire par blocs** : ≈ 3 660 runs par soumission.
5. **Trois formats A/B/C** : bundle nirs4all, Python lib externe sklearn, R package via Docker.
6. **Séparation `nirs4all_arena` ↔ `nirs4all`**.
7. **Stitching interdit programmatiquement** dans la dataviz v0.1.
8. **Trois piliers scientifiques** : calibration transfer, robustesse physique, reproductibilité externalisée.
9. **Deux niveaux de sortie** : base de preuves interne + Arena website public.
10. **Stockage minimal** : workspaces `nirs4all` comme journal d'exécution, `arena.sqlite` comme index protocolaire.
11. **Cadence de criblage** : objectif interne minimal de 10 datasets/mois.

### 3.4 Jalons de sortie de Phase 0a — auditable

| Critère | Type | Responsable | Artefact | Échéance |
|---|---|---|---|---|
| Manifeste rédigé + 2 revues Codex intégrées | livré | Claude + Codex | `systematic_benchmarking_protocol.md` | atteint |
| 4 design docs (01-03, 06) revus Codex + intégrés | livré | Claude + Codex | docs/_internal/arena/ | atteint |
| 2 design docs sketches (04, 05) **revus Codex** | **bloquant** | Claude + Codex | review intégrée | **à faire avant Phase 0b** |
| Décision finale **nom PyPI** | bloquant | utilisateur | choix figé | à faire |
| Décision finale **licence** repo | bloquant | utilisateur | choix figé | à faire |
| Décision finale **stratégie DOI** | bloquant | utilisateur | choix figé | à faire |
| Décision finale **liste initiale `selected_v0.1`** | bloquant | utilisateur | JSON freezé | à faire |
| `SCOPE_v0.1.md` rédigé avec budget par livrable | bloquant | utilisateur + Claude | doc | à faire |

### 3.5 Gate Phase 0a → Phase 0b

**No-go tant que :**
- Tous les bloquants §3.4 ne sont pas résolus.
- Pas de revue Codex MAJOR_REVISIONS non-intégrée pour les docs 04 et 05.
- `selected_v0.1` non figé (liste exacte des 26 datasets attendus pour la cartographie).

**Go quand :**
- Tous les checks §3.4 sont verts (`atteint` ou `résolu`).
- `SCOPE_v0.1.md` figé.

### 3.6 Risques de Phase 0a

- **Sur-spécification** : risque de figer des choix qui devraient rester ouverts. Mitigation : versionnage explicite (v0.1b, v0.1c, v0.2b) ; bumps autorisés en Phase 1-2.
- **Sous-spécification** : risque inverse. Mitigation : multiples cycles de revue Codex.
- **Manque de feedback externe** : conception solitaire. Mitigation : Phase 5 (bêta) ouvre formellement au feedback.

## 4. Phase 0b — Bootstrap repo (T0 → T+1.5m)

### 4.1 Livrables

1. **Création du repo `nirs4all_arena`** :
   - `pyproject.toml` avec deps + extras (`arena`, `r`).
   - `README.md` (pitch + roadmap pointer + setup).
   - LICENSE (choix figé en Phase 0a).
   - CI minimale (`ruff`, `mypy`, `pytest`).
2. **Migration des design docs** depuis `nirs4all/docs/_internal/arena/` vers `nirs4all_arena/docs/design/`.
3. **Création du repo `nirs4all-datasets-registry`** — scaffolding seulement.
4. **Document `SCOPE_v0.1.md`** : périmètre v0.1 vs backlog v0.2, budget jours-personnes par livrable.
5. **Document `GOVERNANCE.md` v0** : toi seul comité éditorial initial + processus de recrutement.
6. **Premier ADR** : stockage minimal `WorkspaceStore SQLite + arena.sqlite sidecar` et règles de non-extension du coeur `nirs4all`.
7. **Conventions internes** :
   - Style (PEP 8, ruff, type hints).
   - Tests (≥ 80 % couverture runtime).
   - Versioning (semver strict).
   - Branches.

### 4.2 Gate Phase 0b → Phase 1

**No-go tant que :**
- Repo non `pip install -e .` fonctionnel.
- CI non verte sur PR vide.
- `SCOPE_v0.1.md` non figé.

**Go quand :**
- Repo opérationnel, CI verte.
- Périmètre v0.1 figé, signed-off.
- ADR stockage minimal approuvé.

### 4.3 Risques de Phase 0b

- **Allocation de temps** : phase d'amorce souvent sous-estimée. Mitigation : reuser au maximum les conventions de `nirs4all`.

## 5. Phase 1 — Registry et cartographie des datasets (T+1.5m → T+3m)

### 5.1 Livrables — découpés en 3 lots

**Lot 1 — Smoke datasets (8 datasets)**. Sous-ensemble strict de `selected_v0.1`, choisi pour la diversité minimale (small/large `n_samples`, signal-type, instrument). Cartographie complète T1-T7 + audit licences. *Doit être terminé avant tout déblocage Phase 2.*

**Lot 2 — Reste de `selected_v0.1` (18 datasets)**. Cartographie complète des 18 restants. Tolère que certains finissent après Phase 1 si non-bloquants pour Phase 2.

**Lot 3 — Pool `all`**. Cartographie minimale (struct + cible + licence) des 29 régression + 11 classification de `bench/tabpfn_paper/data/`. Profil qualité signal (T4) peut rester en seconde passe. *Pas un bloquant Phase 1 → Phase 2.*

### 5.2 Livrables transversaux Phase 1

1. **Schema sidecar `arena.sqlite`** :
   - Tables essentielles : `arena_meta`, `protocol_versions`, `dataset_cards`, `dataset_versions`, `split_masks`, `recipe_definitions`, `submissions`, `submission_files`, `run_specs`, `run_bindings`, `env_cards`, `seed_cards`, `verification_checks`, `evidence_snapshots`, `publication_snapshots`.
   - Les scores, durées, predictions, residus et artefacts restent dans les workspaces `nirs4all`.
   - Les vues `leaderboard_global`, `matrix_model_dataset`, `variance_decomposition`, `fragility_matrix` sont des exports JSON regenerables, pas des tables primaires.
   - Migrations SQLite versionnées.
2. **`nirs4all-datasets-registry`** rempli :
   - Datasets `selected` avec leurs métadonnées et splits.
   - Licences et CITATION.cff par dataset.
   - DOI Zenodo de la release v0.1 de la registry.

### 5.3 Gate Phase 1 → Phase 2

**No-go tant que :**
- Lot 1 (8 smoke datasets) non complet.
- Schéma `arena.sqlite` non validé (tests d'intégration verts avec au moins un workspace `store.sqlite` attaché).
- Licences non vérifiées sur Lot 1.
- `selected_v0.1` non figé bit-à-bit (hash de la liste publié).

**Go quand :**
- 8 DatasetCards Lot 1 validées (Codex/humain).
- Splits canoniques reproductibles (test = générer 2 fois et comparer).
- Crosswalk Lot 1 audité.
- Anti-leakage Lot 1 résolu ou taggué `accepted_with_caveat`.

### 5.4 Risques de Phase 1

- **Crosswalk complexe** : 60+ `dataset_alias` à mapper aux dossiers physiques. Mitigation : automatiser au maximum, `decisions_log.md` pour les cas ambigus.
- **Licences ambiguës** : exclusion du pool public, conservation en `private` pour audit interne.
- **Cartographie chronophage** : Lot 1 strict d'abord ; Lots 2-3 peuvent glisser sans bloquer Phase 2.
- **Mauvais couplage stockage/runtime** : risque de dupliquer les scores dans `arena.sqlite`. Mitigation : tests d'intégration qui vérifient que les exports publics se reconstruisent depuis `arena.sqlite` + workspaces.

## 6. Phase 2 — Baselines + runtime alpha + dispatch spike B/C (T+3m → T+5m)

### 6.1 Livrables

1. **Baselines implémentées** (cf [02_pls_canon.md](02_pls_canon.md) v0.1b) :
   - `pls_canon_regression`, `plsda_canon_classification`.
   - `ridge_canon_regression`, `ridge_canon_classification`.
   - `rf_canon_regression`, `rf_canon_classification`.
   - Sanity tests verts, `canon_sanity_v0.1b.yaml` gelé.
2. **Grille canonique v0.1b** :
   - `nirs4all_arena.protocol.grid_v01b.generate_canonical_grid()`.
   - JSON Schema `manifest_schema_A.json`.
   - Tests déterminisme, cardinalité, couverture, ordre canonique.
3. **Runtime alpha (format A complet)** :
   - Submission Receiver (validation format A).
   - Grid Expander.
   - Scheduler joblib loky CPU.
   - Executor + Persister + ReproducibilityVerifier (re-run 1-2 %).
4. **Dispatch spike B et C** *(nouveau ; demande Codex)* :
   - Schemas `manifest_schema_B.json` et `manifest_schema_C.json` figés.
   - Submission Receiver détecte format B/C, instancie le container Docker (`nirs4all-arena-py:v0.1`, `nirs4all-arena-r:v0.1`).
   - **Dummy B** : un estimator `sklearn.dummy.DummyRegressor` empaqueté, soumis comme format B → traverse 2 datasets × 2 splits du B1 *sans entrée au leaderboard public* (statut `spike`).
   - **Dummy C** : un script R no-op (`function(X, y) lm(y ~ 1)`) soumis comme format C → traverse 2 datasets × 2 splits.
   - Le spike teste : manifest parsing, isolation Docker, EnvCard format-aware, persistance workspace + `run_bindings`, dispatch RModelController.
5. **ADR `API_CONTRACT_NIRS4ALL.md`** : documente le contrat d'API nirs4all utilisé par arena.
6. **Matrice CI** :
   - Job `nirs4all-pinned` : `nirs4all==0.8.11`.
   - Job `nirs4all-dev` : branche dev de `nirs4all` (signal précoce de cassure).
7. **Première campagne** : exécuter PLS-canon, Ridge-canon, RF-canon sur 8 smoke datasets (puis 18 restants au fur et à mesure de Lot 2). Persister les résultats. Calculer les premiers `score_ratio_vs_pls_canon`.
8. **CLI minimale** : `arena submit <path>`, `arena run --resume`, `arena status <submission_id>`.

### 6.2 Gate Phase 2 → Phase 3

**No-go tant que :**
- 3 baselines n'ont pas passé leur sanity test gelé.
- Runtime alpha n'a pas réussi à exécuter PLS-canon sur les 8 smoke datasets (test d'acceptation §12 du runtime).
- Dispatch spike B et C n'a pas réussi sur 2 datasets × 2 splits chacun.
- Matrice CI `nirs4all-pinned` non verte.

**Go quand :**
- Test d'acceptation runtime alpha passé.
- Dispatch spike B/C confirmé : 4 runs B + 4 runs C écrits dans un workspace standard, liés dans `arena.sqlite` avec EnvCard correcte, container Docker fonctionnel.
- Baselines indexées sur ≥ 8 datasets.

### 6.3 Risques de Phase 2

- **Bug double-scaling** (déjà connu cf v0.1b PLS-canon) — à valider empiriquement sur synthétique avant freeze.
- **Performance scheduler joblib** : pour > 64 cores, prévoir fallback Prefect/Dagster en Phase 5/6.
- **Stockage Parquet** : estimer empiriquement la taille des prédictions/résidus sur 8 smoke datasets ; extrapoler ; ajuster la stratégie de compression si besoin.
- **API nirs4all instable** : si `nirs4all 0.8.12` casse l'API durant Phase 2, le pin v0.8.11 protège ; mais l'ADR doit aussi prévoir un plan de mise à jour.

## 7. Phase 3 — Formats B et C complets (T+5m → T+7m)

### 7.1 Livrables

1. **Format B complet** :
   - Parser `manifest_schema_B.json` v1.0.
   - Installeur pip dans container `nirs4all-arena-py:v0.1`.
   - Wrapper sklearn dans pipeline nirs4all.
   - Canary tests (label permutation + FS access audit).
   - Sanity test : `lasso_canon_b` (`sklearn.linear_model.Lasso` empaqueté) → traverse la grille.
2. **Format C complet** :
   - `nirs4all_arena.controllers.r_model.RModelController` enregistré au registry nirs4all.
   - `r_bridge_template_v0.1.R` (template marshalling, sans coercition spécifique).
   - Image Docker `nirs4all-arena-r:v0.1` (rocker/r-ver:4.3.x + paquets canon).
   - Marshalling Parquet ↔ R matrix.
   - Canary tests.
   - Sanity test : `mdatools::pls` (bridge contributeur fourni) → traverse la grille.
3. **EnvCard format-aware** : capture complète par format.
4. **Politique GPL opérationnelle** : `LICENSE_OBLIGATIONS.md` template + mode export `no_gpl` implémenté.
5. **CLI étendue** : `arena submit --format <A|B|C>`.

### 7.2 Gate Phase 3 → Phase 4

**No-go tant que :**
- 3 soumissions test (une par format) n'ont pas traversé la grille complète avec succès.
- Canary tests ne sont pas verts sur Dummy B et Dummy C.
- Bug `nirs4all_arena.controllers.r_model.RModelController` non corrigé.
- Politique GPL non documentée et testée.

**Go quand :**
- Sanity tests par format passent.
- ≥ 1 soumission externe format B et ≥ 1 format C traversent la grille (en mode beta closed).
- Workspaces + `arena.sqlite` persistent les artefacts (`python_env_lockfile`, `r_session_info`, `container_image_digest`) sans erreur.

### 7.3 Risques de Phase 3

- **R toolchain instable** : Docker image versionnée, image-as-code.
- **Sécurité format B/C** : Docker pour les deux dès v0.1 (durci suite à revue Codex format).
- **License contamination R** : tag `gpl_derived` héritée + mode export `no_gpl` implémenté en Phase 3.
- **`pip` / `install.packages` lents** : cache par soumission, pas par run (cf §06).

## 8. Phase 4 — Evidence reports + Arena Web v0.1 (T+7m → T+9m)

### 8.1 Livrables

1. **Générateurs evidence internes** :
   - `recipe_cards/*.json` : une fiche par recette importante (`ASLSBaseline + PLS`, `SNV + Ridge`, etc.).
   - `anti_patterns.json` : combinaisons rarement utiles ou fragiles.
   - `toolkit_recommendations.json` : defaults recommandes par contexte.
   - `coverage_report.json` : datasets/mois, domaines couverts, methodes testees, trous restants.
2. **Générateurs JSON publics** (cf [05_web_minimal_sketch.md](05_web_minimal_sketch.md)) :
   - `submissions.json`, `leaderboard_global.json`, `matrix_model_dataset.json`, `variance_decomposition.json`, `fragility_matrix.json`.
   - Régénérés à chaque indexation.
3. **Site statique Arena (Option A)** :
   - Hugo/Eleventy/Astro + Vega-Lite.
   - 4 vues V1-V4 fonctionnelles.
   - Pages `/`, `/leaderboard/`, `/matrix/`, `/variance/`, `/fragility/`, `/models/<id>/`, `/datasets/<alias>/`, `/protocol/`, `/grid/`, `/about/`, `/submit/`.
4. **Déploiement** : GitHub Pages ou équivalent.
5. **i18n FR/EN minimal** (must-have : leaderboard + matrix + about ; nice-to-have : reste).
6. **Tests d'acceptation web** : cf §8 du web sketch.

### 8.2 Gate Phase 4 → Phase 5

**No-go tant que :**
- Les evidence cards internes ne produisent pas au moins 5 conclusions actionnables relisibles par l'equipe.
- Les 4 vues V1-V4 ne s'affichent pas correctement avec ≥ 3 soumissions (les baselines).
- Pas de cycle complet validé : nouvelle soumission → indexation → site mis à jour.
- Site non accessible publiquement.

**Go quand :**
- Evidence reports utiles pour le criblage interne.
- 4 vues fonctionnelles.
- Cycle complet validé.
- Site sur URL publique.
- Tests responsivité OK (mobile minimum sur leaderboard + matrix + fragility).

### 8.3 Risques de Phase 4

- **Pas de Web designer** : prioriser fonctionnel sur esthétique en v0.1.
- **Vega-Lite limites** : si vues exigent des interactions complexes, fallback Plotly ou D3.

## 9. Phase 5 — Bêta minimale → paper-ready (T+9m → T+12m)

### 9.1 Sous-phase 5a — Bêta minimale (T+9m → T+10.5m)

**Objectifs** :
- ≥ 5 labos invités.
- ≥ 10 soumissions externes (mix format A/B/C).
- Pool `selected` consolidé (16-20 datasets).
- Site publié, leaderboard avec ≥ 13 entrées (3 baselines + 10 externes).
- Premiers retours qualitatifs (issues GitHub).
- Cadence interne maintenue : ≥ 10 datasets/mois ou justification documentee.

### 9.2 Sous-phase 5b — Paper-ready (T+10.5m → T+12m, sous conditions)

**Objectifs paper-ready** :
- ≥ 30 soumissions externes.
- ≥ 30 datasets dans `selected`, ≥ 60 dans `all`.
- Trajectoire interne claire vers ≈300 datasets et ≈300 recettes/methodes testees si le projet continue sur 24 mois.
- ≥ 5 retours qualitatifs publiés.
- Site avec DOI Zenodo v0.1.
- Comité éditorial constitué (3-5 personnes minimum).
- Réplication indépendante sur ≥ 10 datasets organisée.

**Si ces critères ne sont pas atteints à T+12m, la sous-phase 5b glisse de v0.1 vers v0.2.** Le manuscrit méthodologique attend. Une cible type Nature Methods ne devient réaliste qu'avec un corpus longitudinal substantiel, des contributions externes et une analyse au-dela des rankings.

### 9.3 Livrables Phase 5

1. **Documentation publique** :
   - Guide de soumission par format.
   - Tutoriels (un par format, avec exemple complet).
   - Politique de gouvernance v1.
   - FAQ.
2. **Contributeurs invités** :
   - Outreach (mailing-listes NIRS, conférences chimiométrie).
   - Support technique.
3. **Datasets étendus** : Lot 2 et Lot 3 complétés.
4. **Re-exécution des baselines** sur le pool consolidé.
5. **Première campagne de communication** : annonces, slides standard.

### 9.4 Gate Phase 5 → Phase 6 (paper-ready)

**No-go tant que :**
- Critères §9.2 non atteints.
- Comité éditorial non constitué.
- Réplication indépendante non organisée.

**Go quand :**
- Critères §9.2 atteints.

### 9.5 Risques de Phase 5

- **Adoption nulle** : recrutement actif obligatoire.
- **Surcharge support** : limiter Phase 5a à 5-10 contributeurs pour ne pas saturer.
- **Demandes hors-périmètre** : roadmap publique, "v0.2 backlog".

## 10. Phase 6+ — Stabilisation et publication (T+12m → T+18m)

### 10.1 Livrables

1. **v0.1 freeze** : `master_csv_freeze_v0.1.yaml`, DOI Zenodo gelé.
2. **Calibration de puissance** : PLS-canon × 30 seeds sur fast12 → recalibrer K en B1.
3. **Bump v0.2** : grille étendue, format D (notebook), hold-out test set sacré, rpy2 alternative R, vues web v0.2.
4. **Publication méthodologique** :
   - Preprint ChemRxiv / ArXiv (T+12 → T+15m).
   - Soumission Nature Methods / Anal. Chem. / NeurIPS D&B (T+15 → T+18m, **sous conditions paper-ready**).
   - Réplication tierce confirmée.

### 10.2 Gate publication

**Aucune soumission journal avant** :
- Comité éditorial constitué + actif.
- Réplication indépendante validée ≥ 10 datasets.
- ≥ 20-30 datasets externes open-licence intégrés (au-delà des datasets internes initiaux).
- ≥ 3 cas multi-capteur démontrés (calibration transfer).

Si non-atteints, retourner en Phase 5b ; la publication méthodologique glisse vers v0.2.

## 11. Vue d'ensemble — chemin critique

```
              Cartographie       Baselines + spike   Runtime A          Site
              Lot 1 8 datasets   B/C (Phase 2)       complet (Ph 2)     (Ph 4)
              (Ph 1)
T0──Ph0b──┬───┬──────────┬──────────────────┬──────────────┬──────────────┬────► T+9m
          │   │          │                  │              │              │
          │   │          │                  │              │              │
          │   └──Ph1 Lot 2/3 ────────────────────────────────              │
          │                                                                │
          └──Ph3 (formats B/C complets) ────────────────────────────────────┘
                                                                       
                            Bêta minimale (Ph 5a) → paper-ready (Ph 5b) ──► T+12m
                                                                              
                                                      v0.2 (Ph 6+) ──► T+18m
```

**Chemin critique strict** : Phase 0b → cartographie Lot 1 → baselines + runtime alpha → spike B/C → formats B/C complets → web v0.1 → bêta minimale → paper-ready.

## 12. Compute budget par phase (estimations à valider empiriquement)

| Phase | CPU-h estimés | GPU-h estimés | Stockage cumulé |
|---|---:|---:|---:|
| Phase 1 | ≤ 50 (cartographie + tests) | 0 | < 5 Go |
| Phase 2 (3 baselines × 26 datasets × 600 runs ≈ 47 000 runs ; ≈ 1-30 s/run) | 200-2 000 | 0 | 10-50 Go |
| Phase 3 (dummy B + dummy C + sanity tests sur 8 smoke) | 50-100 | 0 | +5 Go |
| Phase 4 | < 10 (génération sites) | 0 | +1 Go |
| Phase 5a (10 soumissions × 3 660 runs ≈ 36 600 runs ; mix runtime_tier) | 1 000-10 000 | 50-500 | +50-200 Go |
| Phase 5b (30 soumissions cumulées) | 5 000-50 000 | 200-2 000 | +200 Go |
| Phase 6 (re-runs bump majeur sur tous les selected) | jusqu'à 2× total Phase 5 | idem | +400 Go |

**Hypothèses** :
- Modèles légers (PLS, Ridge, RF) dominants en Phase 2-3.
- TabPFN GPU et NN profonds apparaissent en Phase 5.
- Stockage : prédictions + résidus Parquet + bundles `.n4a`.

**Budget total estimé Phase 5b** : ≈ 50 000 CPU-h + 2 000 GPU-h + 250 Go stockage. À un coût cloud médian (≈ 0.05 USD/CPU-h, ≈ 0.5 USD/GPU-h), ≈ 3 500 USD pour la première année *si* tout est sur le cloud commercial. Avec accès à un cluster académique : marginal.

**Action en Phase 0b** : confirmer l'accès au cluster cible (CPU + GPU + stockage). Sans cela, le périmètre v0.1 doit être réduit.

## 13. Risques transverses

**RT1 — Disponibilité unique du mainteneur**. Mitigation : documentation très poussée, code lisible, ADRs, tests, automatisation. *Le `SCOPE_v0.1.md` est le levier principal — il borne le périmètre à ce qui est tenable.*

**RT2 — Drift de nirs4all**. Mitigation : pin strict `==0.8.11` ; matrice CI testant pin + branche dev ; ADR `API_CONTRACT_NIRS4ALL.md`.

**RT3 — Dérive du protocole**. Mitigation : v0.1 fige la grille v0.1b ; tout ajout = bump majeur (donc re-run). Roadmap publique.

**RT4 — Coût compute exponentiel**. Mitigation : tiering, queue, transparence sur les délais. Budget §12 à valider.

**RT5 — Communauté inerte**. Mitigation : co-auteurship des contributeurs `selected`, communication active, challenges trimestriels.

**RT6 — Sécurité formats B/C**. Mitigation : Docker systématique dès v0.1 ; canary tests ; audit éditorial des nouveaux paquets.

**RT7 — Couplage stockage inadapté**. Risque : `arena.sqlite` devient une copie partielle et divergente des workspaces. Mitigation : contrat de stockage minimal, `run_bindings` comme mapping unique, exports reconstruits depuis les sources primaires, tests de reconstruction sur workspaces réels.

**RT8 — Publication méthodologique trop ambitieuse à T+12m**. Mitigation : structuration en Phase 5a / 5b ; soumission type Nature Methods seulement si la base accumule assez de datasets, recettes, contributions et analyses. Sinon, publier d'abord la ressource/protocole.

## 14. Parties prenantes — rôles à pourvoir

| Rôle | Responsabilités | Pourvu en Phase | État courant |
|---|---|---|---|
| Mainteneur principal | architecture, code, runtime, web, doc | 0a | toi (utilisateur) |
| Comité éditorial | curation `selected`, arbitrages, gouvernance | 5a (3 personnes) → 5b (5-7) | non pourvu — recrutement en cours côté utilisateur |
| Reviewer du manuscrit méthodologique | relecture pre-submission, validation scientifique | 5b → 6 | non pourvu |
| Arbitre des litiges leaderboard / dataset | appels, conflits d'intérêts | 5a+ | mainteneur initialement, déléguer à comité dès 5b |
| Operateur cluster compute | provisioning, monitoring, budget | 2+ | à décider en Phase 0b (cluster cible) |
| Communication / outreach | blogs, mailings, conférences | 4+ | mainteneur jusqu'à 5b, déléguer si possible |

## 15. Plan communication

| Phase | Canal | Action | Cadence |
|---|---|---|---|
| 0b | issue tracker public | annonce projet, scope, contact | 1× |
| 2-3 | blog dev (Hashnode / GitHub Pages) | journal des décisions, ADRs | mensuel |
| 4 | mailing-lists NIRS (NIRS-L, mdatools-users, chemometrics-L) | annonce site v0.1 | 1× |
| 5a | conférences (Chimiométrie ASRC, NIR Symposium SCIX) | démonstration ; appel à contributeurs | dès calendrier |
| 5b | preprint + post LinkedIn / Twitter académique | annonce manuscrit | 1× preprint |
| 6+ | journal scientifique | soumission ; suite communication active | continu |

## 16. Décisions à arbitrer avant Phase 0b

- **Nom de package PyPI** : `nirs4all-arena` (cohérent avec `nirs4all`) ?
- **Hébergement** : GitHub + GitHub Pages au début ?
- **Licence du repo** : CeCILL-2.1 (cohérent nirs4all) ?
- **Naming des baselines** : `PLS-canon` vs `PLSCanon` vs `arena-pls-canon` ?
- **DOI strategy** : Zenodo classique ou OSF ?
- **CI hosting** : GitHub Actions ?
- **Cluster compute** : académique (gratuit, contraint) vs cloud commercial (payant, flexible) ?

## 17. Livrables finaux v0.1 (avant publication)

- Repo `nirs4all_arena` publié + DOI Zenodo.
- Repo `nirs4all-datasets-registry` publié + DOI Zenodo.
- Site `nirs-arena.org` (ou équivalent) accessible.
- Documentation complète (protocole, tutoriels, guide de soumission par format).
- ≥ 30 soumissions externes, ≥ 30 datasets `selected`, ≥ 60 datasets `all`.
- Manuscrit méthodologique soumis.
- Comité éditorial constitué et fonctionnel.

## 18. Au-delà — vision long terme (T+18m+)

- **Format D** (notebook) et **F** (Docker complet) selon traction.
- **Hold-out servers** sur sous-ensemble sacré (anti-cherry-picking).
- **Extension à d'autres spectroscopies** : Raman, MIR, FTIR.
- **Méta-modèle zero-shot** : prédicteur du "meilleur candidat" sur un nouveau dataset.
- **Synthetic stress-test corpus** : datasets synthétiques contrôlés (drift, n_samples, SNR).
- **Publication empirique** (Papier 2 du manifeste §5.6.2), sous conditions strictes d'indépendance.
- **Workshop conférence dédié** (Chimiométrie / NIR Symposium, NeurIPS DBT, EuroAnal).

Ces éléments ne sont pas planifiés en jalons ; ce sont des opportunités à exploiter selon la traction réelle.
