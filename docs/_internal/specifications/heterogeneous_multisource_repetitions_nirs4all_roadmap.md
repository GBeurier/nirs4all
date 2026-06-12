# Roadmap nirs4all - Répétitions multisources hétérogènes

Date: 2026-06-12
Statut: roadmap d'implémentation, à reviewer avant développement
Design de référence: `docs/_internal/specifications/heterogeneous_multisource_repetitions.md`
Périmètre: nirs4all uniquement. Le challenge 2026 est hors périmètre.

## Objectif

Implémenter dans nirs4all le support natif des répétitions spectrales hétérogènes par source, par exemple `MIR=2`, `RAMAN=3`, `NIRS=2`, sans régression des chemins existants:

- `repetition=` uniforme legacy;
- `aggregate` / `by_repetition`;
- `rep_to_sources` et `rep_to_pp`;
- branching/merge existants;
- refit simple, refit stacking, bundles `.n4a`, API `RunResult` / `PredictResult`;
- workspace et visualisations existantes.

La sortie principale reste une prédiction sample-level (`physical_sample_id`). Les niveaux observation/source/combo sont des niveaux intermédiaires, diagnostics, ou méta-features.

## Décisions de design déjà tranchées

- Les répétitions sont échangeables par défaut.
- Les cibles sont sample-level.
- `per_source_aggregate` et late fusion sont tous les deux nécessaires; l'ordre de livraison doit suivre les dépendances techniques, pas une préférence ML arbitraire.
- Le cartésien est une représentation de première classe, pas un hack: il approxime `E[f(x)]` et permet des interactions feature-level.
- Une source absente en prédiction doit passer par une policy configurable avec warning (`impute_declared`, padding/mask, modèle partiel, ou `strict`).
- La provenance doit toujours être conservée, en réutilisant l'esprit du mécanisme `origin` / augmentation existant.
- nirs4all et DAG-ML ont chacun leur implémentation. nirs4all ne doit pas attendre DAG-ML.
- Le projet n'est pas en v1 publique stable, mais il existe déjà des contrats de régression pour API publique, schéma workspace et bundle. Toute modification visible de signature/API/format doit être explicitement soumise à décision et couverte par `tests/regression/`.

## Changements publics à décider avant implémentation

Ces points ne doivent pas être modifiés silencieusement:

1. Ajouter des paramètres publics à `Predictions.top()` / `get_best()`:
   - `evaluation_scope`;
   - `reduction_plan` ou `reduction`;
   - `refit_slot` / `selection_level`.
2. Ajouter une syntaxe YAML expérimentale:
   - `relations`;
   - `representations`;
   - `reducers`;
   - `fit_influence`;
   - `meta_features`;
   - `refit_slots`.
3. Ajouter des colonnes persistées dans les prédictions/workspace:
   - `prediction_scope`;
   - `prediction_level`;
   - `evaluation_scope`;
   - `reduction_role`;
   - `reduction_id`;
   - `physical_sample_id`;
   - `origin_sample_id`;
   - `derived_unit_id`;
   - `unit_level`;
   - `unit_id`;
   - `row_id`;
   - `sample_influence_weight`.
4. Étendre les bundles `.n4a` pour porter la relation table, les représentation plans, reducers et policies de missingness.
5. Introduire ou exposer des classes publiques:
   - `RepetitionSpec`;
   - `SampleRelationPlan`;
   - `RepresentationPlan`;
   - `ReductionPlan`;
   - `FitInfluencePolicy`;
   - `StackingFitContract`.
6. Étendre `api/explain`, `RunResult`, `PredictResult`, exports et visualisations pour porter la provenance des features agrégées.
7. Acter les comportements visibles qui deviennent des erreurs ou refus explicites dans le profil relationnel:
   - sources hétérogènes legacy sans `link_by`/relation plan;
   - `link_by` présent mais non exécuté comme vraie jointure;
   - sources de même longueur avec IDs divergents, shuffle non déclaré ou IDs non uniques;
   - `merge.unsafe=True`;
   - `validate_fold_alignment=False`;
   - imputation de couverture silencieuse type `CoverageStrategy.IMPUTE_MEAN`;
   - warning de fuite/alignment promu en erreur dans `experimental_relation_pipeline`.

Décision recommandée: exposer tout cela d'abord sous un namespace expérimental (`experimental_relation_pipeline: true`) et garder les signatures legacy inchangées tant que possible.

## Architecture existante à réutiliser

Points d'ancrage déjà présents:

- `nirs4all/data/predictions.py`
  - `Predictions.aggregate()`;
  - `Predictions.top(..., by_repetition=...)`;
  - `Predictions.get_best()`;
  - agrégation de répétitions et ranking déjà partiellement couplés.
- `nirs4all/operators/models/meta.py`
  - `TestAggregation` pour mean / weighted / best-fold.
- `nirs4all/controllers/models/meta_model.py` et `controllers/models/stacking/reconstructor.py`
  - reconstruction et agrégation des prédictions de stacking.
- `nirs4all/controllers/splitters/split.py`
  - `compute_effective_groups()` et logique de composantes connexes.
- `nirs4all/data/dataset.py`
  - `repetition`, `aggregate`, `reshape_reps_to_sources()`, `reshape_reps_to_preprocessings()`.
- `nirs4all/operators/data/repetition.py`
  - `RepetitionConfig` legacy.
- `nirs4all/data/loaders/loader.py`, `data/schema/config.py`, `data/parsers/files_parser.py`
  - `link_by` existe dans la config mais doit devenir un vrai contrat de jointure.
- `nirs4all/data/_indexer/augmentation_tracker.py`
  - provenance d'augmentation à réutiliser pour les lignes dérivées.
- `nirs4all/pipeline/execution/refit/*`
  - sélection/refit à étendre vers `EvaluationScope` et `RefitSlotPlan`.
- `nirs4all/pipeline/storage/workspace_store.py`, `pipeline/storage/array_store.py`
  - persistance à étendre.
- `nirs4all/pipeline/storage/store_schema.py`
  - version de schéma workspace à traiter explicitement, avec tests regression.
- `nirs4all/pipeline/bundle/*`
  - replay predict à étendre pour relations/reducers/policies.
- `nirs4all/api/explain.py`, explain results, `NIRSPipeline` / sklearn wrappers
  - provenance et explanations à préserver quand les features sont agrégées ou dérivées.

## Principes de non-régression

- Ne pas changer le comportement de `repetition=` uniforme si aucune `RepetitionSpec` source-aware n'est fournie.
- Ne pas supprimer `fold_id="final"`, `avg`, `w_avg`, ou `_agg` au premier passage. Ajouter les nouveaux champs et un accessor typé.
- Ne pas transformer `rep_to_sources` / `rep_to_pp`; ajouter `rep_fusion` comme chemin relationnel séparé avec matrice d'exclusivité.
- Garder les agrégations existantes comme adaptateurs vers `ReductionPlan`.
- Garder les configs existantes valides, sauf configs déjà silencieusement fausses: sources multisources de longueurs incompatibles, `link_by` présent mais non exécuté, sources de même longueur avec IDs divergents, ou merge positionnel ambigu doivent devenir des erreurs.
- Tout warning de fuite/alignment dans le profil relationnel doit devenir une erreur. `merge.unsafe`, `validate_fold_alignment=False` et les imputations de couverture type `IMPUTE_MEAN` doivent être refusés par défaut dans un plan relationnel, sauf opt-in expérimental explicite hors validation finale.

## Phase N0 - Baseline, garde-fous et contrats de non-régression

But: figer le comportement actuel avant de changer les fondations.

Travaux:

- Ajouter un audit de démarrage pour détecter:
  - sources multisources avec longueurs différentes sans `link_by`/relation plan;
  - `link_by` configuré mais non exécuté comme vraie jointure;
  - sources de même longueur mais IDs divergents, shuffled ou non uniques;
  - `rep_fusion` mélangé avec `rep_to_sources` ou `rep_to_pp`;
  - `repetition=` global utilisé sur sources non alignées;
  - `merge.unsafe=True`, `validate_fold_alignment=False`, ou `CoverageStrategy.IMPUTE_MEAN` dans un profil relationnel.
- Initialiser le socle de réduction avant toute représentation:
  - `ReductionPlan` minimal comme adaptateur autour de `Predictions.aggregate()`, `top(by_repetition=...)` et `TestAggregation`;
  - `PredictionLevel` et `PredictionUnitId` en métadonnées internes;
  - hooks dans `top()`, `get_best()`, `extract_top_configs()` et refit pour accepter plus tard `evaluation_scope` sans changer immédiatement les comportements legacy.
- Ajouter une matrice d'exclusivité:
  - `repetition=` legacy;
  - `rep_to_sources`;
  - `rep_to_pp`;
  - `rep_fusion` relationnel.
- Documenter les valeurs de `fold_id` legacy et créer un helper interne:
  - `is_real_cv_fold(fold_id)`;
  - `prediction_scope_from_legacy(fold_id, refit_context, reduction metadata)`.

Modules:

- `nirs4all/data/config.py`
- `nirs4all/data/schema/config.py`
- `nirs4all/data/loaders/loader.py`
- `nirs4all/operators/data/repetition.py`
- `nirs4all/controllers/data/repetition.py`
- `nirs4all/data/predictions.py`
- `nirs4all/controllers/data/merge.py`
- `nirs4all/pipeline/execution/refit/config_extractor.py`
- `nirs4all/pipeline/execution/refit/model_selector.py`

Tests:

- non-régression `test_repetition.py`;
- non-régression `test_aggregation_integration.py`;
- erreur claire pour multisource `2N/3N/2N` sans relation;
- erreur claire pour `link_by` présent mais non exécutable;
- erreur claire pour sources même longueur mais `sample_id` divergents;
- erreur claire pour `merge.unsafe` / OOF imputation silencieuse en profil relationnel;
- parity minimale `Predictions.aggregate()` / `ReductionPlan`;
- configs legacy simples inchangées.

Definition of Done:

- aucun pipeline legacy valide ne change de résultat;
- les cas silencieusement incohérents échouent avec message actionnable;
- `ReductionPlan`, `PredictionLevel`, `PredictionUnitId` et l'accessor `fold_id` typé existent comme socle interne;
- le design relationnel peut être activé par flag sans polluer le legacy.

## Phase N1 - Identités relationnelles sample/source/observation

But: résoudre une fois les identités au lieu de laisser split, reshape, scoring et export recalculer chacun leur notion de sample.

Travaux:

- Introduire un modèle interne:
  - `physical_sample_id`;
  - `internal_sample_id`;
  - `unit_level`;
  - `unit_id`;
  - `source_id`;
  - `observation_id`;
  - `rep_id`;
  - `origin_sample_id`;
  - `derived_unit_id`;
  - `row_id`;
  - `partition`;
  - `target_id`;
  - `sample_influence_weight`;
  - `quality_flag`.
- Ajouter `SampleRelationPlan` et une `NormalizedObservationTable` en mémoire.
- Stabiliser le mapping `sample_id` YAML -> `physical_sample_id`.
- Ajouter le lineage:
  - `component_observation_ids` pour combos;
  - fingerprints de relation table;
  - provenance `origin` pour lignes dérivées/augmentées.
- Valider:
  - unicité `(physical_sample_id, source_id, rep_id)`;
  - cible constante par sample;
  - metadata sample-level non contradictoires;
  - cardinalités attendues si déclarées.

Modules:

- nouveau module probable: `nirs4all/data/relations.py`
- `nirs4all/data/dataset.py`
- `nirs4all/data/indexer.py`
- `nirs4all/data/_indexer/sample_manager.py`
- `nirs4all/data/_indexer/augmentation_tracker.py`
- `nirs4all/data/loaders/loader.py`

Tests:

- relation table nominale `A=2/B=3/C=2`;
- répétitions désordonnées;
- doublon `(sample, source, rep)`;
- metadata contradictoires;
- target contradictoire;
- `unit_level`/`unit_id` cohérents;
- `sample_influence_weight` présent même si égal à `1.0`;
- lineage combo complet;
- reload workspace.

Definition of Done:

- un dataset relationnel peut être construit sans produire encore de `Features`;
- tous les samples ont une clé physique stable et persistable;
- la table relationnelle reprend le schéma canonique complet du design;
- l'ancien index interne reste utilisable pour les pipelines legacy.

## Phase N2 - `RepetitionSpec` source-aware et `link_by` réel

But: rendre la configuration explicite.

Travaux:

- Ajouter `RepetitionSpec`:
  - `sample_id`;
  - `target_level=physical_sample`;
  - `sources.<source>.rep_col`;
  - `sources.<source>.expected`;
  - `missing_repetition_policy`;
  - `missing_source_policy`;
  - `rep_order=exchangeable|ordered`;
  - `strict_cardinality`.
- Faire de `link_by` une vraie jointure validée, pas seulement un champ parsé.
- Supporter les sources shuffled par `link_by`.
- Refuser les jointures positionnelles quand `link_by` est requis.

Modules:

- `nirs4all/data/schema/config.py`
- `nirs4all/data/parsers/files_parser.py`
- `nirs4all/data/loaders/loader.py`
- `nirs4all/data/config.py`

Tests:

- source order permuté;
- source même longueur mais IDs divergents;
- source manquante;
- `link_by` absent;
- `link_by` non unique;
- shared targets via `link_by`.

Definition of Done:

- un utilisateur peut décrire `MIR=2`, `RAMAN=3`, `NIRS=2`;
- le loader produit une `NormalizedObservationTable`;
- aucun dataset hétérogène n'est chargé par accident comme sources alignées.

## Phase N3 - Staging source-specific avant `Features`

But: nirs4all doit porter son propre staging source-specific, sans attendre DAG-ML, et sans refondre `Features` en ragged natif au premier lot.

Travaux:

- Ajouter `RawMultiSourceDataset`:
  - `X_by_source`;
  - `headers_by_source`;
  - `targets_by_sample`;
  - metadata par niveau.
- Ajouter la matérialisation vers `SpectroDataset` / `Features` uniquement via `RepresentationPlan`.
- Ne pas refondre `Features` en ragged au premier lot: `Features` reste rectangulaire/aligné, et le ragged natif reste un refactor long terme séparé.

Modules:

- nouveau module probable: `nirs4all/data/raw_multisource.py`
- `nirs4all/data/features.py`
- `nirs4all/data/dataset.py`
- `nirs4all/data/loaders/loader.py`

Tests:

- staging source-specific avec cardinalités différentes;
- conversion vers représentation alignée;
- fingerprint relation table;
- ordre déterministe.

Definition of Done:

- la donnée hétérogène peut exister en mémoire sans matrice rectangulaire fausse;
- tout passage vers `Features` déclare une représentation;
- aucun refactor ragged natif de `Features` n'est requis pour les premiers livrables.

## Phase N4 - `ReductionPlan` complet comme généralisation de l'existant

But: compléter le socle minimal N0 pour ne pas créer un `output_reducer` parallèle.

Travaux:

- Compléter `ReductionPlan`:
  - `role=score|persist|fold_ensemble|meta_feature|final_output`;
  - `axis=unit|fold|model|metric`;
  - `input_level`;
  - `output_level`;
  - `method=mean|median|vote|weighted_mean|robust|custom`;
  - `weight_source`;
  - `task_compatibility`.
- Adapter:
  - `Predictions.aggregate()`;
  - `Predictions.top(... by_repetition ...)`;
  - `Predictions.get_best()`;
  - `TestAggregation`;
  - reporting/visualisations.
- Ajouter les métadonnées de prédiction:
  - `prediction_level`;
  - `prediction_scope`;
  - `evaluation_scope`;
  - `reduction_role`;
  - `reduction_id`.
- Relier explicitement ces métadonnées à l'accessor `fold_id` typé introduit en N0.
- Ajouter un contrat d'état fit/replay pour tout reducer ou transform qui estime des paramètres:
  - `fit_scope=stateless|fold_train|full_train_refit`;
  - `fit_partition`;
  - `fold_id`;
  - `state_id`;
  - fingerprint des paramètres appris.
- Appliquer ce contrat aux reducers non triviaux, QC/outlier, trimmed/robust statistics, imputers, padding/mask statistics et calibrations. Les paramètres sont fités sur train-fold puis rejoués sur validation; le refit final produit un état séparé full-train.

Modules:

- `nirs4all/data/predictions.py`
- `nirs4all/controllers/shared/prediction_aggregator.py`
- `nirs4all/controllers/models/meta_model.py`
- `nirs4all/controllers/models/stacking/reconstructor.py`
- `nirs4all/operators/models/meta.py`
- `nirs4all/visualization/*`

Tests:

- parity `Predictions.aggregate()` vs `ReductionPlan`;
- `top(by_repetition=True)` inchangé;
- `TestAggregation.MEAN/WEIGHTED/BEST` inchangé;
- scoring sample-level à partir de combos;
- classification mean-proba et vote.
- reducers/imputers/QC fités train-fold uniquement et rejoués sur validation/predict;
- refus si un state fit sur validation/test est détecté.

Definition of Done:

- le même reducer peut servir au score, au ranking, au refit, au reporting et au predict final;
- les anciens champs restent supportés comme adaptateurs;
- aucun reducer ou imputer fitable ne peut être évalué avec un état appris hors fold.

## Phase N5 - `RepresentationPlan` et opérateur `rep_fusion`

But: produire des matrices alignées à partir du staging relationnel.

Travaux:

- Ajouter opérateur/config `rep_fusion`.
- Implémenter représentations non cartésiennes en N5a:
  - `per_source_aggregate`;
  - `per_source_observation`;
  - `sample_aggregate`;
  - `stack_fixed`;
  - `stack_padded_masked`.
- Déclarer mais ne pas rendre exécutables avant N6 + N7 + N9 minimal les représentations cartésiennes N5b:
  - `cartesian_full`;
  - `cartesian_mc`;
  - `cartesian_augmentation`.
- Chaque représentation déclare:
  - `unit_level`;
  - `stage`;
  - `lineage`;
  - `CombinationPlan` si elle dérive des combos;
  - `combo_selection=deterministic_all|random_seeded|stratified|match_by|zip`;
  - `missing_source_policy`;
  - `missing_repetition_policy`;
  - `max_combos_per_sample`;
  - `max_total_combos`;
  - `max_total_rows`;
  - `memory_budget`;
  - `random_state`;
  - manifest de représentation rejouable;
  - `fingerprint`.

Modules:

- `nirs4all/operators/data/repetition.py`
- nouveau `nirs4all/operators/data/rep_fusion.py`
- nouveau `nirs4all/controllers/data/rep_fusion.py`
- `nirs4all/data/dataset.py`
- `nirs4all/data/features.py`

Tests:

- `per_source_aggregate -> branch by_source -> merge_sources -> model`;
- `per_source_aggregate` en mode prédiction (`supports_prediction_mode=True`) rejoue le plan sur train/test avec cardinalités différentes;
- `cartesian_full -> grouped CV -> reduce combo->sample`;
- `cartesian_mc` seedé;
- refus si le cap global lignes/mémoire est dépassé;
- replay du manifest de combos sans régénération non déterministe;
- `cartesian_augmentation` avec `origin`, combos seulement en train et validation/prédiction sur la base non augmentée;
- `stack_fixed` refuse cardinalités incompatibles;
- `stack_padded_masked` accepte missing avec mask.

Definition of Done:

- `rep_fusion` est le seul point de passage relationnel vers `Features`;
- le cartésien a un lineage complet et une borne de coût;
- `cartesian_full`, `cartesian_mc` et `cartesian_augmentation` restent non exécutables tant que N6, N7 et le replay N9 minimal ne sont pas passés;
- aucun `rep_fusion` n'est considéré livrable tant que le replay workspace/`.n4a` minimal de N9 pour cette représentation ne passe pas;
- la provenance d'augmentation est conservée.

## Phase N6 - Split, OOF, scoring et refit par scope

But: sélectionner/refitter sur l'unité correcte.

Travaux:

- Ajouter `EvaluationScope=row|observation|combo|sample`.
- Auto-grouping par `physical_sample_id` dès qu'une représentation produit plusieurs lignes par sample.
- Passer les warnings de leakage/alignment en erreurs dans le profil relationnel.
- Étendre:
  - `Predictions.top()`;
  - `Predictions.get_best()`;
  - `extract_top_configs()`;
  - `model_selector.py`;
  - `RefitConfig`;
  - `orchestrator`.
- Ajouter `RefitSlotPlan`:
  - `refit_one`;
  - `refit_ensemble`;
  - `selection_level`;
  - `selection_metric`;
  - `reduction_plan`.

Modules:

- `nirs4all/controllers/splitters/split.py`
- `nirs4all/data/predictions.py`
- `nirs4all/pipeline/execution/refit/config_extractor.py`
- `nirs4all/pipeline/execution/refit/model_selector.py`
- `nirs4all/pipeline/execution/refit/executor.py`
- `nirs4all/pipeline/execution/orchestrator.py`

Tests:

- aucun combo du même sample dans deux folds;
- `best row-level` différent de `best sample-level`;
- refit choisi sur score sample-level réduit;
- `refit_one` et `refit_ensemble` ne s'écrasent pas;
- nested vs exploratory `reuse_oof`.

Definition of Done:

- un score combo-level ne peut pas être utilisé par erreur comme score principal sample-level;
- un refit porte le scope et le reducer qui l'ont sélectionné.

## Phase N7 - `FitInfluencePolicy`

But: séparer influence au fit et agrégation des prédictions.

Travaux:

- Ajouter:
  - `auto`;
  - `uniform_rows`;
  - `equal_sample_influence`;
  - `resample_equalized`;
  - `scorer_only`;
  - `backend_loss_weight`;
  - `strict_weight_support`.
- Consommer `sample_influence_weight` déjà porté par la relation table N1.
- Plumber `sample_weight` uniquement dans les backends qui le supportent.
- Pour les backends sans poids:
  - warning + `resample_equalized` si `auto`;
  - erreur si `strict_weight_support`.
- Ne pas confondre `strict_weight_support`, `strict_cardinality` et `missing_source_policy=strict`.
- Règles déterministes de `auto`:
  - cardinalités constantes et pas de lignes dérivées: `uniform_rows`;
  - cardinalités constantes avec cartésien accepté scientifiquement: `uniform_rows` avec diagnostic explicite;
  - cardinalités variables et backend avec poids: `equal_sample_influence`;
  - cardinalités variables et backend sans poids: `resample_equalized` avec warning persistant;
  - si aucun fallback déclaré n'est possible: erreur.
- Ne pas figer `sample_influence_weight` comme vérité unique dans N1: la relation table porte une colonne calculable/auditable, mais le poids effectif est dérivé par représentation, fold, backend et policy puis persistant dans le run manifest.

Modules:

- `nirs4all/controllers/models/base_model.py`
- contrôleurs sklearn / keras / torch / tabular selon support réel;
- `nirs4all/operators/models/*`
- `nirs4all/data/relations.py`

Tests:

- cartésien PLS accepté en `uniform_rows`;
- Ridge/sklearn avec poids si supporté;
- backend sans poids en `auto`;
- backend sans poids en `strict_weight_support` échoue;
- cardinalités variables égalisent bien l'influence sample.

Definition of Done:

- les poids de fit ne sont jamais confondus avec les poids de reducer;
- le run manifest permet d'auditer l'influence choisie.

## Phase N8 - Late fusion et stacking multi-domaines

But: supporter proprement les branches par source et les merges de prédictions.

Travaux:

- Ajouter `MetaFeaturePlan`:
  - `meta_row_domain=sample|combo`;
  - adapters `direct|reduce|broadcast|source_to_combo`;
  - `alignment_key=physical_sample_id|combo_id`.
- Ajouter `StackingFitContract`:
  - `meta_training_features=oof`;
  - `inference_features=refit_base_predictions`;
  - `selection_protocol=nested|holdout|reuse_oof`;
  - `base_prediction_calibration=none|rank|calibrated_oof_to_refit`.
- Ajouter `SelectionProtocol`:
  - `branch_level`: chaque branche choisit ses refits localement puis le meta est entraîné;
  - `global`: le meta choisit la combinaison complète des sous-modèles;
  - `meta_aware`: les sous-modèles gardés sont ceux qui améliorent le meta, pas ceux qui gagnent localement.
- Par défaut: `nested` pour validation finale; `reuse_oof` est exploratoire et interdit dans le profil final-validation.
- Interdire par défaut:
  - méta-features in-sample;
  - last-write-wins sur OOF combo-level;
  - merge positionnel si les unités ne correspondent pas.
- Implémenter missing prediction policy:
  - `strict`;
  - `impute_declared`;
  - `drop_incomplete`;
  - `drop_branch`;
  - `mask`;
  - `pad`;
  - `partial_model`.
- Unifier `missing_source_policy`, `missing_repetition_policy` et missing prediction policy dans une même taxonomie persistée. Chaque représentation déclare son défaut et ses fallbacks autorisés; le predict/serve ne peut pas choisir une stratégie implicite.

Modules:

- `nirs4all/controllers/data/branch.py`
- `nirs4all/controllers/data/merge.py`
- `nirs4all/operators/data/merge.py`
- `nirs4all/controllers/models/meta_model.py`
- `nirs4all/controllers/models/stacking/reconstructor.py`
- `nirs4all/pipeline/execution/refit/stacking_refit.py`

Tests:

- `by_source -> modèles source -> OOF sample-level -> meta`;
- `sample-meta`;
- `combo-meta` avec reduce final obligatoire;
- source absente au predict avec imputation/padding;
- `strict`, `drop_incomplete`, `mask/pad`, `partial_model` et `impute_declared` testés séparément;
- validation fold alignment stricte;
- meta final entraîné sur OOF, pas in-sample.
- sélection `branch_level`, `global` et `meta_aware`;
- `reuse_oof` refusé en profil final-validation.

Definition of Done:

- late fusion est utilisable pour production sample-level;
- le stacking ne peut pas joindre silencieusement des unités incompatibles.

## Phase N9 - Workspace, bundle `.n4a`, replay predict

But: rendre la feature déployable et rejouable.

Règle de séquencement: N9 a un mode minimal obligatoire dès le premier `rep_fusion` livré. Une représentation qui ne peut pas être reloadée, exportée en `.n4a` et rejouée en prédiction ne doit pas être marquée terminée.

Travaux:

- Persister:
  - `NormalizedObservationTable` ou son manifest;
  - `RepresentationPlan`;
  - `ReductionPlan`;
  - `FitInfluencePolicy`;
  - `MissingnessPolicy`;
  - `StackingFitContract`;
  - fingerprints.
- Versionner explicitement les changements via `store_schema.py` et les tests de contrat.
- Ajouter les colonnes prédictions de façon additive.
- Étendre les exports.
- Étendre explainability:
  - niveau d'explication (`raw_observation`, `source_aggregate`, `combo`, `sample`);
  - mapping feature agrégée -> observations/répétitions sources;
  - avertissement si SHAP/explain porte sur features agrégées plutôt que spectres bruts.
- Étendre bundle generation / loading:
  - rejouer `rep_fusion`;
  - rejouer reducers;
  - valider cardinalités train/predict;
  - appliquer fallback source absente;
  - préserver provenance.

Modules:

- `nirs4all/pipeline/storage/workspace_store.py`
- `nirs4all/pipeline/storage/store_schema.py`
- `nirs4all/pipeline/storage/array_store.py`
- `nirs4all/pipeline/bundle/generator.py`
- `nirs4all/pipeline/bundle/loader.py`
- `nirs4all/api/predict.py`
- `nirs4all/api/result.py`
- `nirs4all/api/explain.py`
- `nirs4all/pipeline/nirs_pipeline.py` ou wrappers sklearn si impactés par explain/predict.

Tests:

- workspace reload;
- `.n4a` export/predict même cardinalité;
- `.n4a` export/predict cardinalité différente;
- `.n4a` `per_source_aggregate`: même répétitions, répétitions différentes, source manquante;
- `.n4a` late fusion/stacking: OOF sample-level, refit bases, predict avec missing branch;
- `.n4a` `cartesian_full`: `CombinationPlan`, seed, caps et reducers `combo -> sample` rejoués identiquement;
- `.n4a` `cartesian_mc`: seed et `max_combos_per_sample` déterministes;
- `.n4a` `cartesian_augmentation`: validation/prédiction sur base non augmentée, combos train-only non servis;
- missing source en policy warning+impute;
- missing source en `strict`, `drop_incomplete`, `mask/pad`, `partial_model`;
- classification reducers: mean-proba, vote, seuil/calibration persistés hors test;
- `FitInfluencePolicy`: poids effectifs et fallback backend persistés;
- export CSV sample-level sans lignes combo indésirables.
- tests regression API publique, schéma workspace et bundle;
- explainability sur `per_source_aggregate` avec provenance.

Definition of Done:

- un run relationnel peut être sauvegardé, reloadé, exporté et utilisé en prédiction;
- les anciens bundles restent lisibles.
- les explications n'attribuent pas une feature agrégée à une observation brute sans lineage.

## Phase N10 - API utilisateur, docs et exemples

But: rendre la feature compréhensible.

Travaux:

- Documenter:
  - différence répétition uniforme vs source-aware;
  - `FitInfluencePolicy` en termes simples;
  - `StackingFitContract` en termes simples;
  - `per_source_aggregate`, late fusion, cartésien, stack fixe;
  - impact explainability/provenance quand les répétitions sont agrégées.
- Ajouter exemples YAML minimaux:
  - pré-agrégation par source;
  - late fusion;
  - cartésien complet;
  - source absente en prédiction.
- Ajouter guides de migration:
  - `repetition=` legacy vers `relations`;
  - `aggregate` vers `reducers`.

Modules:

- `docs/source/*`
- `examples/sample_configs/*`
- `examples/configs/*`

Tests:

- tests de parsing exemples;
- smoke examples.
- exemple explain/predict sur features agrégées.

Definition of Done:

- un utilisateur peut configurer le cas `A=2/B=3/C=2` sans lire le code;
- les limites et erreurs sont explicites.

## Ordre recommandé de livraison

1. N0 + N1 + N2: garde-fous, identités, config relationnelle.
2. N4 socle complet: reducers, niveaux de prédiction et hooks ranking/refit.
3. N3: staging ragged et table relationnelle persistable.
4. N5 part 1 + N9 minimal: `per_source_aggregate`, `per_source_observation`, manifest et replay `.n4a`.
5. N6: scoring/refit sample-level.
6. N8 part 1 + N9 minimal: late fusion sample-level, OOF strict, replay bundle.
7. N5 part 2 + N7: cartésien complet/MC/augmentation, `CombinationPlan`, influence et caps globaux.
8. N8 part 2: combo-meta expérimental.
9. N10: docs publiques, exemples et explainability.

## Hors scope ou différé

- Refactor ragged natif de `Features`: différé. Les premiers livrables passent par un staging source-specific puis des représentations rectangulaires déclarées.
- Convergence/export DAG-ML: différé. Les noms et invariants doivent rester compatibles, mais nirs4all ne doit pas attendre DAG-ML pour implémenter la feature.
- `combo-meta` par défaut: refusé. Il reste expérimental et doit exiger `combo_id`, OOF strict, `FitInfluencePolicy`, reducer final `combo -> sample` et opt-in explicite.
- Suppression physique des conventions legacy (`fold_id` surchargé, suffixes `_agg`, vieux `repetition=`): différée tant que les adaptateurs et tests de régression ne prouvent pas la migration.

## Risques principaux

- Sous-estimer l'impact de `fold_id` legacy sur API, webapp, reports et bundle.
- Laisser deux notions de sample coexister.
- Implémenter `rep_fusion` directement dans `Features` et perdre le staging relationnel.
- Faire du cartésien sans bornes de coût.
- Sélectionner/refitter sur le mauvais scope.
- Entraîner un meta-modèle final sur prédictions in-sample.
- Ajouter des colonnes workspace sans migration/reload tests.
- Oublier les tests regression publics (`tests/regression`) alors que les formats API/workspace/bundle changent.
- Faire de l'explainability une réflexion tardive alors que les features agrégées changent le sens scientifique des attributions.

## Gate de validation avant merge global

Commandes minimales:

```bash
ruff check .
mypy nirs4all
pytest tests/unit/
pytest tests/integration/
pytest tests/regression/
pytest tests/ --cov=nirs4all --cov-report=xml
```

Suites ciblées à garder vertes à chaque phase:

- `tests/unit/data/test_group_split_validation.py`
- `tests/integration/pipeline/test_aggregation_integration.py`
- `tests/integration/pipeline/test_merge_per_branch.py`
- `tests/integration/pipeline/test_meta_stacking_integration.py`
- `tests/unit/pipeline/execution/refit/*`
- `tests/unit/pipeline/storage/*`
- `tests/regression/test_public_api_contract.py`
- `tests/regression/test_storage_schema_contract.py`
- `tests/regression/test_bundle_contract.py`
- nouveaux tests `tests/unit/data/test_relation_table.py`
- nouveaux tests `tests/integration/pipeline/test_heterogeneous_multisource_repetitions.py`
- nouveaux tests explainability/provenance sur features agrégées
