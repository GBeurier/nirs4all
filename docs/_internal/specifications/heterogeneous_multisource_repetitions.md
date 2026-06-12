# Répétitions spectrales hétérogènes par source

Date: 2026-06-12
Statut: analyse de design pour implémentation nirs4all / DAG-ML, non implémentée, revues Claude/fable ultracode et Codex hors contexte intégrées, annotations utilisateur du 2026-06-12 intégrées

Cette note synthétise les analyses initiales sur nirs4all, une analyse dédiée à DAG-ML (`../dag-ml`), une seconde analyse Codex hors contexte et une revue Claude/fable en lecture seule. Le cas déclencheur est un dataset multisource avec trois modalités spectrales, par exemple `MIR=2`, `RAMAN=3`, `NIRS=2` répétitions par échantillon physique.

## Résumé décisionnel

Le problème ne doit pas être modélisé comme "trois sources avec des longueurs différentes". Le vrai problème est que nirs4all suppose aujourd'hui un axe ligne commun à toutes les sources, alors que les répétitions hétérogènes créent plusieurs axes observationnels:

- `physical_sample_id`: l'échantillon physique, unité statistique de split, d'évaluation et de sélection. `sample_id` reste seulement un alias YAML utilisateur.
- `source_id`: la source/modality/instrument, par exemple `MIR`, `RAMAN`, `NIRS`.
- `observation_id`: une mesure spectrale réelle d'une source pour un échantillon.
- `rep_id`: l'indice de répétition dans une source donnée.
- `derived_unit_id` / `combo_id`: une unité dérivée, par exemple une des 12 combinaisons `MIR_i x RAMAN_j x NIRS_k`.

La recommandation est de ne pas étendre silencieusement `repetition="sample_id"` pour couvrir ce cas. Il faut garder ce chemin comme legacy uniforme, et ajouter une spécification explicite des répétitions par source.

Chemin court terme recommandé:

1. Bloquer le chargement silencieusement faux: aujourd'hui `link_by` est perdu et les sources de longueurs différentes peuvent être chargées dans un état incohérent.
2. Introduire une clé sample externe stable, distincte de l'id interne auto-incrémenté, puis une `RepetitionSpec` source-aware en lecture/validation.
3. Ajouter un nouvel opérateur `rep_fusion`, sans changer `rep_to_sources` ni `rep_to_pp`, avec une matrice d'exclusivité explicite entre ces mécanismes.
4. Implémenter `rep_fusion(mode="per_source_aggregate")` comme premier jalon de représentation alignée: une ligne par sample physique, par source, avec provenance des répétitions.
5. Forcer split, métriques, ranking, refit et API `top/export` sur un `evaluation_scope` explicite, en gardant les défauts legacy.
6. Livrer la fusion tardive OOF sample-level dans le même périmètre fonctionnel que la pré-agrégation, puis ajouter le cartésien contrôlé. La fusion tardive est le meilleur défaut général quand aucun appariement brut inter-source n'est requis; le cartésien est scientifiquement légitime quand on veut apprendre des interactions feature-level ou approximer `E[f(x)]`. Il exige split sample-level, scoring/refit sample-level et une politique explicite d'influence des lignes dérivées; la pondération de fit est un moyen, pas une condition universelle.

Nuance importante ajoutée après les revues indépendantes: il ne faut pas opposer "pré-agrégation sûre" à "cartésien dangereux". La pré-agrégation choisit `f(E[x])`; le cartésien complet avec agrégation des prédictions approxime `E[f(x)]`. Pour des modèles non linéaires, ces deux quantités diffèrent. Le choix est donc un compromis biais/coût/robustesse de déploiement, pas un simple garde-fou de structure.

Le chemin long terme correct est plus proche du modèle DAG-ML: relations explicites `sample_id` / `observation_id` / `source_id`, granularité de plan, agrégation observation -> sample, et contrats OOF/fold-alignment avant stacking ou évaluation finale.

Apports ajoutés après la seconde passe indépendante:

- formalisation du problème d'appariement avant le choix d'architecture;
- justification positive du cartésien par `E[f(x)]` vs `f(E[x])`;
- distinction cartésien complet, Monte-Carlo et cartésien train-only via `origin`/augmentation;
- séparation stricte fuite de données versus pseudo-réplication/pondération;
- ajout des niveaux de sélection branch-level, global et meta-aware;
- nuance DAG-ML/data: les contrats relationnels aident, mais la fusion feature-level ragged reste à expliciter et ne doit pas devenir une jointure implicite.

## Addendum - révision après use case ultime

Cet addendum corrige plusieurs points du design initial et sert de base pour la roadmap nirs4all / DAG-ML. Le design vise le moteur générique; les cas applicatifs externes servent seulement de stress tests et ne doivent pas contraindre la forme de l'API.

### Clarifications imposées par le use case

1. **Réduction de sortie existante.** nirs4all possède déjà plusieurs mécanismes proches d'un `output_reducer`:
   - `Predictions.aggregate()` réduit des prédictions par groupe (`mean`, `median`, `vote`, exclusion outliers);
   - `Predictions.get_best()` / `top()` peuvent recalculer le score après agrégation via `by_repetition`;
   - l'orchestrateur peut créer des twins `fold_id="<orig>_agg"` pour les prédictions agrégées;
   - `MetaModel` / `TrainingSetReconstructor` ont `TestAggregation` pour agréger les prédictions test de folds (`mean`, `weighted`, `best`).

   Il ne faut donc pas ajouter un concept parallèle nommé `output_reducer`. Le bon design est de généraliser ces mécanismes en un seul objet canonique `ReductionPlan`, avec `role={score,persist,fold_ensemble,meta_feature,final_output}` et `axis={unit,fold,model,metric}`. Les anciens noms (`aggregate`, `by_repetition`, `_agg`, `TestAggregation`) deviennent des adaptateurs legacy vers ce contrat.

2. **Cartésien et poids.** Le cartésien complet ne nécessite pas mathématiquement des poids pour fitter un PLS. Les 12 combinaisons sont des lignes spectrales standards, à condition que les folds soient groupés par sample biologique pour éviter la fuite. La pondération répond à une autre question: quelle influence statistique donne-t-on à un sample biologique qui génère plus de lignes dérivées qu'un autre?

   Donc:
   - si tous les samples ont exactement `2 x 3 x 2 = 12` combos, un fit non pondéré ne déséquilibre pas les samples entre eux;
   - si les cardinalités varient ou si des sources/répétitions manquent, une loss non pondérée surpondère les samples riches en combos;
   - le score principal, le ranking et le refit doivent rester sample-level, pondérés ou agrégés correctement;
   - `sample_weight` est une politique de fit (`fit_influence_policy`), pas une barrière absolue au mode cartésien;
   - pour les backends sans poids, on peut autoriser `fit_influence_policy="uniform_rows"` avec avertissement scientifique, `scorer_only`, ou `resample_equalized`; on ne refuse que si la config exige explicitement `equal_sample_influence`.

3. **CNN > feature reduction > TabPFN.** En syntaxe nirs4all, cette chaîne signifie que la feature reduction repart des samples post-preprocessing; elle ne consomme pas la sortie du CNN. Chainer des sorties modèle comme features d'un autre modèle est un problème de merge/stacking de prédictions et doit passer par le mécanisme de merge prévu. Le design ne doit donc pas déclarer automatiquement une fuite interne CNN->TabPFN pour cette écriture.

4. **DAG-ML est le futur moteur, pas une cible legacy.** Les limitations actuelles du prototype Rust ne doivent pas contraindre la forme finale. Il faut concevoir un moteur relationnel propre: unités explicites, relations, ports typés par niveau, reducers déclarés, OOF/fold contracts, lineage et replay. Si cela implique de recoder des pans de `dag-ml`, c'est acceptable.

### Frontiere feature-plane / prediction-plane

Le design doit rendre explicite une frontière qui existe déjà implicitement dans nirs4all:

- **feature-plane**: matrices spectrales, preprocessings, feature reduction, augmentations, représentation cartésienne ou wide;
- **prediction-plane**: prédictions OOF/test/refit, reducers, agrégation fold/model, scoring, ranking, refit selection.

Un `ReductionPlan` ne traverse jamais seul de prediction-plane vers feature-plane. Le seul mécanisme autorisé pour réinjecter une sortie modèle comme feature est un noeud de merge/stacking de prédictions, avec OOF contract et alignment key explicites. Cette règle évite de relire `CNN > feature_reduction > TabPFN` comme un chaînage de sorties modèle et évite aussi de cacher des méta-features in-sample derrière une étape de preprocessing.

### Modèle de staging obligatoire

nirs4all ne peut pas appliquer proprement `rep_fusion` directement sur `Features`, car `Features` représente aujourd'hui des sources déjà alignées sur un même axe ligne. Le support hétérogène exige donc une couche de staging avant toute matérialisation rectangulaire:

- `RawMultiSourceDataset` ou `NormalizedObservationTable` porte `X_by_source: dict[source_id, array]`;
- une table relationnelle porte au minimum `physical_sample_id`, `internal_sample_id`, `source_id`, `observation_id`, `rep_id`, `derived_unit_id`, `target_id`, `partition`, `quality_flag`;
- les targets et métadonnées sample-level vivent une seule fois au niveau `physical_sample_id`;
- les métadonnées source-level et observation-level sont typées explicitement;
- `rep_fusion` consomme cette table de staging et produit ensuite seulement un `SpectroDataset` / `Features` aligné.

Conséquence de roadmap: il faut ajouter une Phase N1.5 obligatoire avant les représentations (`per_source_aggregate`, `cartesian_full`, `stack_fixed`). Le support long terme de tables source-specific dans `Features` est une refonte ragged séparée; il ne doit pas être présenté comme prérequis du premier livrable.

### Primitives de design révisées

Le design doit distinguer les primitives suivantes:

| Primitive | Rôle | Exemple |
| --- | --- | --- |
| `RawMultiSourceDataset` | staging ragged avant tout `Features` aligné | `X_by_source["MIR"]`, arrays par source |
| `NormalizedObservationTable` | table relationnelle canonique des observations, combos et samples | lignes `observation`, `combo`, `sample` avec lineage |
| `SampleRelationPlan` | graphe relationnel entre sample physique, source, observation, répétition et unité dérivée | `physical_sample_id=S1`, `source_id=A`, `rep=A2`, `observation=A:S1:2`, `combo=A2:B3:C1` |
| `RepresentationPlan` | matérialisation de lignes/features pour un noeud, avec `stage` explicite | `raw_before_preprocessing`, `after_observation_preprocessing`, `fold_fitted_reducer`, `refit_fitted_reducer` |
| `CombinationPlan` | stratégie de construction d'observations dérivées multisources | `cartesian`, `zip`, `match_by`, `sample_k`, budgets, seed, politique missing |
| `PredictionUnitId` | clé typée des prédictions et réductions | `source_observation:A:S1:2`, `derived_combo:S1:A2:B3:C1`, `sample:S1` |
| `ReductionPlan` | unique contrat de réduction; déclare `role`, `axis`, entrée, sortie, méthode, poids et compatibilité tâche | `role=score, axis=unit, combo -> sample mean`; `role=fold_ensemble, axis=fold, mean` |
| `FitInfluencePolicy` / `SampleInfluencePolicy` | poids ou influence des lignes au fit, séparé des poids de réduction | `uniform_rows`, `equal_sample_influence`, `resample_equalized`, `backend_loss_weight` |
| `EvaluationScope` | unité de scoring et de ranking | `sample` pour le use case principal |
| `RefitSlotPlan` | définition des candidats refit | `best_rmsecv_sample`, `best_mean_repetition_rmse`, `top_k_per_metric` |
| `MetaFeaturePlan` | domaine et colonnes vues par le meta-modèle | `meta_row_domain=sample` ou `combo`, adapters par branche |
| `StackingFitContract` | contrat entre OOF, refit des bases et features d'inférence du meta | `meta_training_features=oof`, `inference_features=refit_base_predictions`, `selection_protocol=nested|reuse_oof|holdout`, `base_prediction_calibration=rank` |
| `FoldContract` | unité de split et compatibilité OOF | `split_unit=physical_sample`, `requires_fold_alignment=true` |
| `LineagePlan` | replay et audit | `origin_sample_id`, `combo_manifest`, seeds, reducers, feature schema fingerprint |

Axes de `ReductionPlan`: `unit` réduit entre niveaux d'unité (`combo -> sample`, `observation -> source_sample`), `fold` réduit des prédictions de folds/refits, `model` réduit ou assemble plusieurs modèles, et `metric` ne sert qu'à construire un critère de ranking/reporting composite, jamais à transformer des lignes de prédiction.

Noms canoniques: `physical_sample_id`, `internal_sample_id`, `source_id`, `observation_id`, `derived_unit_id`, `origin_sample_id`. `sample_id` et `source_name` ne doivent subsister que comme alias d'entrée ou affichage utilisateur. Par défaut, `combo` est une observation/unité dérivée portée par la table relationnelle (`component_observation_ids`, `origin_sample_id`) et les reducers le ramènent au niveau sample. Le promouvoir en `PredictionLevel` / `PredictionUnitId` public de première classe est une décision séparée, utile seulement si un meta-modèle ou un cache public doit consommer directement des lignes combo sans les traiter comme observations dérivées.

Terminologie refit: `refit_one` signifie un slot qui sélectionne un seul candidat; `refit_ensemble` signifie un slot qui sélectionne plusieurs candidats et une règle d'ensemble. Les mots `single` et `aggregated` ne doivent plus être utilisés seuls dans les nouveaux artifacts: ils ne sont valides que comme valeurs explicites de `selection_level` ou de `reduction_role` dans un `RefitSlotPlan`.

Cette séparation évite les ambiguïtés du vocabulaire "aggregation avant/après". Une réduction peut viser le score, le refit, l'entrée du meta, la sortie finale, le reporting ou la persistance d'un artifact. Ces rôles doivent être configurables séparément, même s'ils partagent la même implémentation de `ReductionPlan`.

### Deux domaines meta à supporter

Le moteur doit supporter deux plans meta, avec validation différente.

**Plan A - `meta_row_domain=sample`**

Le meta reçoit une ligne par sample biologique. Chaque branche expose des features de largeur fixe après réduction:

- branche cartésienne: prédictions combo réduites en statistiques sample-level;
- branche concat stricte: prédictions sample-level directes;
- branche source: prédictions observation/source réduites par source puis sample;
- topK/refit: chaque refit slot sélectionné devient une ou plusieurs colonnes selon le `ReductionPlan`.

Ce plan est robuste aux cardinalités train/test différentes tant que les reducers produisent une largeur fixe. C'est le défaut maintenable.

**Plan B - `meta_row_domain=combo`**

Le meta reçoit une ligne par combinaison cartésienne. Les adapters alignent toutes les branches sur ce domaine:

- branche cartésienne: prédiction directe du combo;
- branche concat stricte: prédiction sample-level broadcastée vers chaque combo;
- branche source: prédictions `A_i`, `B_j`, `C_k` mappées sur le combo correspondant; les réductions source/sample peuvent être broadcastées;
- sortie finale: `combo -> sample` via `ReductionPlan(role=final_output, axis=unit)`.

Ce plan est légitime pour apprendre une fonction sur les combinaisons, mais il doit porter explicitement le risque de pseudo-réplication. La validation impose `split_unit=sample`, `EvaluationScope=sample`, `ReductionPlan(role=final_output, axis=unit)`, et une `FitInfluencePolicy` déclarée. Il ne faut pas le bloquer à cause de PLS; il faut seulement rendre la politique d'influence et le score honnêtes.

### Relecture des trois branches du stress test

| Branche | Représentation moteur | Statut de design |
| --- | --- | --- |
| B1 cartésien complet | `RepresentationPlan(cartesian_full, unit=combo)` | mode de première classe, utile pour `E[f(combo)]`; nécessite manifest combo, split sample, reducers sample-level pour score/refit; poids seulement si politique d'influence l'exige ou cardinalités variables |
| B2 concat complet | `RepresentationPlan(stack_fixed, unit=sample)` | valide en mode strict cardinality/order; sinon doit devenir `stack_padded_masked` ou être refusé en prediction mismatch |
| B3 sources séparées | `RepresentationPlan(per_source_observation)` + late prediction fusion | meilleur défaut général, surtout sources manquantes; doit aligner OOF par `physical_sample_id` |

Le cartésien ne doit plus être présenté comme une option à différer nécessairement après `sample_weight`. Le vrai ordre d'implémentation dépend de ce qu'on veut livrer:

- pour un moteur propre, commencer par relations + reducers + scopes, puis implémenter `cartesian_full` proprement;
- pour un usage rapide legacy, `cartesian_augmentation` est plus facile, mais ce n'est pas le design canonique du moteur.

### Ownership nirs4all / DAG-ML

Décision proposée: nirs4all porte une couche de compatibilité et de migration, DAG-ML porte le modèle moteur canonique.

- Les noms, niveaux d'entité, `ReductionPlan`, `CombinationPlan`, `FitInfluencePolicy`, `StackingFitContract` et `FoldContract` doivent être définis comme contrats partagés ou au minimum comme ADR communs, pas réinventés deux fois.
- nirs4all implémente les adaptateurs nécessaires pour préserver les workflows existants et livrer rapidement `per_source_aggregate`, `cartesian_full` et la fusion tardive.
- DAG-ML doit devenir la source de vérité pour les ports typés, arêtes relationnelles, reducers, replay et validation graphe.
- Si un module partagé Python/Rust n'est pas réaliste tout de suite, chaque primitive doit avoir un mapping de conformance documenté et des fixtures golden identiques.
- Toute divergence volontaire doit être nommée comme `legacy_nirs4all`, pas laissée comme une sémantique concurrente.

### Blocages critiques à éviter

1. **Reducer bloqué au reporting.** Si `Predictions.aggregate()` reste seulement un mécanisme de rapport/export, on ne pourra pas sélectionner/refitter sur le score sample-level agrégé. Le reducer doit entrer dans `top()`, `get_best()`, `RefitCriterion`, `config_extractor` et `model_selector`.
2. **Identité sample dérivée plusieurs fois.** Si split, fusion/reshape et scoring recalculent chacun leurs groupes, un pipeline peut être leak-safe dans les folds mais scoré sur une autre notion de sample. Il faut une résolution unique de `sample_key`.
3. **`fold_id` surchargé.** Les valeurs `avg`, `w_avg`, `final` et `_agg` ne sont pas des folds. Tant qu'elles restent dans `fold_id`, les refit slots et le stacking multi-domaines resteront fragiles.
4. **Assembler OOF sample-level avec prédictions combo-level.** Si un contrôleur attend une prédiction par sample et reçoit 12 combos, le risque est un écrasement silencieux ou un alignement faux. Il faut réduire ou porter `combo_id` explicitement avant assemblage.
5. **Poids de fit confondus avec poids de reducer.** `FitInfluencePolicy` / `SampleInfluencePolicy` contrôle l'influence empirique au fit; `ReductionPlan.weight_source` contrôle une moyenne/agrégation de prédictions. Les fusionner rend le cartésien scientifiquement opaque.
6. **Fusion cartésienne comme jointure implicite.** Une expansion `A x B x C` doit être un `CombinationPlan` avec cap, seed, lineage et replay. Une concaténation par position ou un broadcast silencieux doit être refusé.

### Roadmap révisée nirs4all

**N0 - Audit et unification des reducers existants**

- Inventorier `Predictions.aggregate`, twins `_agg`, `by_repetition`, `TestAggregation`, rapports et visualisations.
- Extraire une API interne `ReductionPlan` compatible avec l'existant.
- Introduire `PredictionLevel` et `PredictionUnitId` comme métadonnées typées sur les entrées de prédictions, avant de changer les formats de stockage.
- Garantir que les reducers peuvent être appelés par scoring, reporting, stacking et final prediction sans duplication de logique.
- Désurcharger `fold_id` de manière additive: ajouter `prediction_scope` (`oof`, `refit`, `test`), `reduction_role`, `reduction_id` et un accessor typé qui distingue fold réel et pseudo-prédiction, sans modifier encore les valeurs legacy `avg`, `w_avg`, `final` et twins `_agg`. La suppression physique de ces valeurs dans `fold_id` relève d'un changement de format majeur.
- Tests: parity `Predictions.aggregate` vs nouveau reducer, reporting inchangé, `top(by_repetition=...)` inchangé.

**N1 - Identités relationnelles**

- Ajouter `physical_sample_id`, `internal_sample_id`, `source_id`, `observation_id`, `rep_id`, `origin_sample_id`, `derived_unit_id` dans le modèle metadata/indexer.
- Persister ces identités dans workspace et artifacts.
- Interdire le chargement multisource de longueurs incompatibles sans `link_by`/relation plan explicite.
- Résoudre `sample_key` une seule fois dans le dataset, puis le faire consommer par split, reshape/fusion, aggregation/scoring et export. Ne plus laisser split, reshape et `Predictions.aggregate` recalculer chacun leur propre notion de groupe.
- Tests: doublons, source absente, metadata contradictoire, reload workspace, ordre source permuté.

**N1.5 - Staging ragged avant matérialisation**

- Ajouter `RawMultiSourceDataset` / `NormalizedObservationTable` comme format interne transitoire pour les sources non alignées.
- Porter `X_by_source`, targets sample-level, metadata typées et relation table avant `Features`.
- Faire de `rep_fusion` le point de passage obligatoire vers `SpectroDataset` / `Features` aligné.
- Refuser les chemins qui essaient de charger `MIR=2N`, `RAMAN=3N`, `NIRS=2N` directement comme sources `Features` alignées.
- Tests: loader multisource hétérogène, `link_by` shuffled, source même longueur mais IDs divergents, reload workspace.

**N2 - `RepetitionSpec` et `SampleRelationPlan`**

- Décrire les répétitions par source et non par axe global unique.
- Valider cardinalités attendues, politiques missing source/rep, ordre de répétition.
- Compatibilité legacy: `repetition=` reste un raccourci vers une relation uniforme.
- Tests: `A=2/B=3/C=2`, cardinalités variables, erreurs explicites.

**N3 - `RepresentationPlan`**

- Implémenter `per_source_observation`, `sample_aggregate`, `cartesian_full`, `cartesian_mc`, `stack_fixed`, puis `stack_padded_masked`.
- Déclarer pour chaque représentation son `stage`: `raw_before_preprocessing`, `after_observation_preprocessing`, `fold_fitted_reducer`, `refit_fitted_reducer`.
- Chaque représentation produit un `LineagePlan` et un fingerprint.
- Le cartésien matérialise des combos dérivés; il n'est pas une jointure implicite.
- `CombinationPlan` devient l'objet partagé entre représentation cartésienne, sampling Monte-Carlo, appariement par clé et zip strict.
- Tests e2e: génération 12 combos déterministe, Monte-Carlo seedé, source manquante, replay predict.

**N4 - Split, OOF, scoring et refit par scope**

- Introduire `EvaluationScope` dans predictions, scores, ranking, refit, `top/export`.
- Auto-grouping par `physical_sample_id` dès qu'une représentation produit plusieurs lignes par sample.
- Passer les warnings leakage/fold-alignment critiques en erreurs pour les plans relationnels.
- Promouvoir `ReductionPlan` dans la sélection/refit: `Predictions.top()`, `get_best()`, `RefitCriterion`, `config_extractor`, `model_selector` doivent pouvoir ranker sur `score(row)`, `score(combo)`, ou `score(reduced sample)`.
- Créer des `RefitSlotPlan` distincts pour `refit_one` et `refit_ensemble`, avec `selection_level` explicite: un refit choisi par RMSECV row/combo-level ne doit pas écraser un refit choisi par RMSE/F1 sample-level réduit.
- Tests: aucun combo du même sample dans deux folds, score combo-level vs sample-level distincts, refit slot sample-level.

**N5 - Fit influence policy**

- Ajouter `FitInfluencePolicy` séparé des reducers de score.
- Support backend quand possible (`sample_weight`), fallback documenté (`uniform_rows`, `resample_equalized`, `scorer_only`).
- PLS ne doit pas bloquer le cartésien par défaut; il bloque uniquement `equal_sample_influence` si aucune stratégie de repli n'est déclarée.
- Tests: PLS cartésien accepté en `uniform_rows`, Ridge pondéré, refus contrôlé si `require_weight_support=true`.

**N6 - MetaFeaturePlan et stacking multi-domaines**

- Formaliser `meta_row_domain=sample|combo`.
- Construire des adapters de branche: direct, reduce, broadcast, source-to-combo map.
- Ajouter `StackingFitContract`: `meta_training_features=oof`, `inference_features=refit_base_predictions`, `selection_protocol=nested|reuse_oof|holdout`, `base_prediction_calibration=none|rank|calibrated_oof_to_refit`. `reuse_oof` est autorisé uniquement en exploration et doit être marqué optimiste; il est interdit dans les profils de validation finale. `nested` ou holdout est le protocole de décision robuste.
- Corriger les assembleurs OOF qui supposent une seule prédiction par sample et risquent un comportement last-write-wins en `combo-meta`; imposer un reduce ou une clé `combo_id` explicite avant assemblage.
- Sélection refit meta-aware via candidate pool topK multi-critères.
- Meta refit sur OOF quand demandé; in-sample refit uniquement opt-in explicite.
- Tests e2e: B1+B2+B3 en sample-meta, B1+B2+B3 en combo-meta, topK refit, fold alignment cross-branch.

**N7 - API, YAML et migration**

- Exposer une syntaxe lisible: `relations`, `representations`, `reducers`, `fit_influence`, `meta_features`, `refit_slots`.
- Migrer `aggregate`, `aggregate_method`, `repetition` vers ces primitives sans casser les configs existantes.
- Documenter les combinaisons interdites (`rep_to_sources` legacy + `rep_fusion` relationnel, etc.).

**N8 - E2E moteur**

- Fixtures multisource asymétriques petites et grandes.
- Tests `.n4a` export/predict avec cardinalités identiques et différentes.
- Tests visualisation/reporting sur prédictions réduites.
- Tests performance pour explosion cartésienne et caps.

### Roadmap révisée DAG-ML

**D0 - Modèle relationnel canonique**

- Faire de `SampleRelationSet/Table` la base obligatoire des graphes non triviaux.
- Séparer strictement `EntityUnitLevel` (`physical_sample`, `source_sample`, `observation`, `combo`) de `TargetGranularity`, `GroupConstraint`, `Partition` et `EnsembleAxis`.
- Les combos sont des observations/unités dérivées avec `origin_sample_id` et `component_observation_ids`.
- Décision recommandée pour le chemin bouclable sans régression: `combo` reste relation-backed / observation dérivée dans D0-D9. Le promouvoir en `PredictionLevel` public de première classe est une option breaking différée, à acter seulement pour exposer les combos comme domaine public autonome, par exemple un cache de prédictions combo ou une API de serving combo-level. Un `meta_row_domain=combo` interne/relation-backed peut rester dans le chemin principal si la sortie finale est réduite en sample-level.

**D1 - Ports et arêtes typés par unité**

- Chaque port déclare `unit_level`, `alignment_key`, `feature_schema`, `target_level`.
- Chaque edge déclare s'il transporte features, predictions, relations ou reducers.
- Validation stricte des joins: pas de concat positionnel si les alignment keys ne correspondent pas.

**D2 - Reducers comme contrats de graphe**

- Réconcilier `dag-ml-data::AggregationPolicy` et `dag-ml-core::AggregationPolicy`.
- Corriger la dérive de contrat: le set de reducers ADR côté `dag-ml-data` (`mean`, `weighted_mean`, `median`, `vote`, `robust_mean`, `exclude_outliers`, `custom`) doit être supporté ou explicitement refusé côté `dag-ml-core` avec le même vocabulaire.
- Étendre les reducers au-delà de observation->sample: combo->sample, source_observation->source_sample, prédictions de folds -> ensemble d'inférence, custom.
- Un reducer déclare `role`, `axis`, `input_unit_level`, `output_unit_level` si l'axe est `unit`, `method`, `weight_source`, `task_compatibility`.
- Ajouter `SampleInfluencePolicy` comme politique de fit distincte de `AggregationWeights`. Les poids de réduction ne doivent pas être détournés pour contrôler l'influence empirique des lignes au fit.

**D3 - Representation nodes**

- Ajouter des noeuds DAG natifs: `AggregateRepresentation`, `CartesianProductRepresentation`, `MonteCarloCartesianRepresentation`, `StackFixedRepresentation`, `StackPaddedMaskedRepresentation`.
- Différer `RaggedBagRepresentation`: c'est un refactor modèle/collation plus large, pas un prérequis pour livrer agrégation par source, fusion tardive, cartésien complet ou Monte-Carlo.
- Ces noeuds produisent features + relations + lineage, pas seulement une matrice.
- `CombinationPlan` / `FusionPolicy` doit couvrir `cartesian`, `zip`, `reference_broadcast`, `match_by`, `sample_k`, `budget`, `seed`, `missing_source_policy`, `component_ids`.
- `CombinationExpand` doit lever le refus actuel des fusions multi-répétitions ambiguës seulement quand aucune policy explicite n'est fournie; avec `cartesian` ou `sample_k`, il produit des observations dérivées avec lineage complet.

**D4 - OOF and selection contracts**

- Les noeuds modèle déclarent `training_unit`, `prediction_level`, `evaluation_scope`.
- Les noeuds stacking déclarent `meta_row_domain` et leurs adapters.
- Les selectors/refit slots opèrent sur des `EvaluationResult` typés par scope.
- Ajouter `RefitSlot{scope, selection_level, reduction}`: `single` et `aggregated` sont deux slots distincts, pas deux noms de fold.
- Faire de `selection_metric_level` et `required_metric_level` un contrat unique entre aggregation, selection et refit.

**D5 - Missingness and deployment**

- Policies explicites: `strict`, `drop_incomplete`, `partial_model`, `impute_declared`, `mask`.
- Validation train/serve: quelles cardinalités sont acceptées, quelles dimensions sont fixes, quels reducers stabilisent la largeur.

**D6 - Conformance et e2e**

- Golden DAGs pour `A=2/B=3/C=2`.
- Conformance Python/Rust sur reducers et relations.
- Tests de replay complet: fit DAG -> OOF -> refit -> predict with same reps -> predict with different reps.

### Principe final révisé

La bonne abstraction n'est ni "répétition" ni "output reducer" au singulier. C'est un graphe relationnel typé où chaque étape déclare son unité d'entrée, son unité de sortie, son reducer éventuel, son scope d'évaluation et son contrat OOF. nirs4all peut y arriver par évolution contrôlée; DAG-ML doit le prendre comme modèle natif.

## 1. Reformulation complète du problème

### Nature du problème

Dans un dataset spectroscopique classique, une ligne de `X` correspond à une observation et toutes les sources sont alignées par position: la ligne 17 de MIR, la ligne 17 de RAMAN et la ligne 17 de NIRS décrivent le même objet au même niveau d'observation.

Dans le cas `MIR=2`, `RAMAN=3`, `NIRS=2`, cette hypothèse est fausse si les répétitions sont stockées comme observations source-specific. Pour un échantillon physique `S1`, on a:

```text
MIR:   S1_MIR_1,   S1_MIR_2
RAMAN: S1_RAMAN_1, S1_RAMAN_2, S1_RAMAN_3
NIRS:  S1_NIRS_1,  S1_NIRS_2
```

Il n'existe pas de ligne naturelle unique qui aligne `MIR_1`, `RAMAN_1` et `NIRS_1` comme "la" première répétition multisource. La combinaison `MIR_1 + RAMAN_1 + NIRS_1` est un choix artificiel; la combinaison `MIR_1 + RAMAN_3 + NIRS_2` aussi. Les 12 combinaisons cartésiennes sont des vues augmentées du même échantillon, pas 12 échantillons indépendants.

### Problème d'appariement

La fusion feature-level impose toujours de résoudre un problème d'appariement: quel spectre MIR doit être concaténé avec quel spectre RAMAN et quel spectre NIRS? Il existe plusieurs réponses, chacune avec une sémantique ML différente:

| Résolution | Mécanisme | Ce que cela suppose |
| --- | --- | --- |
| Agréger d'abord | réduire chaque source à un vecteur par sample | les répétitions sont du bruit de mesure résumable |
| Ordonner | aligner `rep_1` avec `rep_1`, etc. | les répétitions ont un ordre sémantique stable |
| Cartésien complet | énumérer tous les appariements | toutes les répétitions sont échangeables et on marginalise sur le choix de mesure |
| Monte-Carlo | échantillonner `K` appariements seedés | on accepte une approximation contrôlée de cette marginalisation |
| Fusion tardive | ne jamais apparier les spectres bruts; agréger les prédictions par source | les interactions brutes inter-sources ne sont pas indispensables |

Parmi les représentations rectangulaires non ragged, la fusion tardive est la seule stratégie qui évite complètement le problème d'appariement. En contrepartie, elle ne peut pas apprendre une interaction directe entre une bande MIR brute et une bande RAMAN brute; elle apprend seulement des interactions entre prédictions ou résumés source-level. Si la chimie attend un signal dans ces interactions brutes, il faut conserver une fusion feature-level, donc accepter soit un empilement ordonné, soit un cartésien complet/Monte-Carlo. Un modèle ragged/multi-instance natif peut aussi éviter l'appariement brut, mais c'est un chantier moteur distinct.

### Unité statistique et pseudo-réplication

L'unité statistique correcte est normalement l'échantillon physique. Les répétitions spectrales sont corrélées entre elles parce qu'elles partagent le même objet, la même cible et souvent les mêmes conditions de prélèvement. Si on les traite comme des lignes indépendantes:

- les scores CV deviennent optimistes si une répétition d'un échantillon est en train et une autre en validation;
- les échantillons avec plus de répétitions pèsent plus lourd dans l'entraînement et dans les métriques;
- la sélection de modèle peut privilégier un pipeline qui explique bien les artefacts de répétition plutôt que la cible sample-level;
- un stacking peut recevoir des méta-features contaminées par des prédictions in-sample ou par des combinaisons quasi dupliquées.

Donc le split, le ranking et le refit doivent être pilotés par `physical_sample_id`, pas par `row_id`, `rep_id` ou `combo_id`.

Il faut séparer deux erreurs souvent confondues:

- Fuite de données: une répétition ou un combo de `S1` est en train et un autre en validation. C'est binaire et catastrophique. La réponse est le groupement obligatoire par `physical_sample_id`.
- Pseudo-réplication: même si tous les combos de `S1` restent dans le même fold, un score row-level ou une loss non pondérée donne plus de poids à `S1` qu'à un sample avec moins de combos. C'est un biais continu. La réponse obligatoire est le scoring sample-level. Pour le fit row-level, `1 / n_rows_derived(sample)` n'est la réponse que si `FitInfluencePolicy=equal_sample_influence`; `uniform_rows` reste valide lorsque les cardinalités sont constantes ou que l'utilisateur accepte scientifiquement cette influence.

On peut quantifier ce problème par le design effect `DEFF = 1 + (m - 1) rho`, où `m` est le nombre de lignes dérivées par sample et `rho` la corrélation intra-sample. Pour un cartésien à `m=12` avec des lignes très corrélées, `N_eff` est beaucoup plus proche du nombre de samples physiques que du nombre de combos. Les intervalles et comparaisons de modèles deviennent trop optimistes si on raisonne sur `N_rows`.

### Ce que signifie le cartésien `2 x 3 x 2 = 12`

Le cartésien peut être utile pour apprendre des interactions feature-level entre sources sans perdre de répétitions. Il crée 12 lignes par échantillon physique:

```text
S1_combo_001 = MIR_1 + RAMAN_1 + NIRS_1
S1_combo_002 = MIR_1 + RAMAN_1 + NIRS_2
...
S1_combo_012 = MIR_2 + RAMAN_3 + NIRS_2
```

Ces lignes doivent porter au minimum:

```text
physical_sample_id | combo_id | MIR_rep_id | RAMAN_rep_id | NIRS_rep_id | sample_influence_weight
```

Si `FitInfluencePolicy=equal_sample_influence`, le poids dérivé est `1 / n_combos(physical_sample_id)` pour que chaque échantillon physique contribue autant qu'un autre. Si `S1` a 12 combinaisons et `S2` en a 6 à cause d'une source manquante ou d'une répétition invalide, cette politique empêche `S1` de peser deux fois plus. Si `FitInfluencePolicy=uniform_rows`, ces lignes sont volontairement traitées comme des observations d'entraînement standards; ce choix doit être visible dans le run manifest.

Le score final doit être agrégé au niveau `sample_id`. Un RMSE calculé sur les 12 combos par échantillon n'est pas comparable à un RMSE sample-level.

Le point positif fort est le suivant: pour un modèle non linéaire `f`, la pré-agrégation entraîne souvent à prédire `f(E[x])`, alors que le cartésien complet puis l'agrégation des prédictions approxime `E[f(x)]`. Si le vrai protocole de prédiction veut marginaliser sur la mesure spectrale qui aurait été obtenue, `E[f(x)]` est souvent l'objectif le plus défendable. Le cartésien n'est donc pas à écarter; il est à encadrer.

Conditions minimales pour le rendre acceptable:

- split groupé par `physical_sample_id`;
- aucune prédiction OOF d'un combo ne provient d'un modèle entraîné sur le même sample physique;
- politique d'influence de fit déclarée: `uniform_rows` si toutes les cardinalités sont identiques et acceptées scientifiquement, `equal_sample_influence` ou stratégie équivalente si les cardinalités varient ou si l'on veut égaliser explicitement les samples;
- métrique principale après agrégation `combo -> sample`;
- sélection/refit sur cette métrique sample-level;
- replay identique en prédiction ou stratégie train-only explicitement déclarée;
- plafond global de lignes/mémoire et génération déterministe.

Cas où il faut justifier ou refuser le cartésien:

- backend modèle sans support de poids uniquement si la config exige `equal_sample_influence` sans politique de repli;
- grand nombre de samples physiques, où le gain d'augmentation ne compense plus le coût `produit des répétitions`;
- répétitions non échangeables sans ordre stable;
- source manquante fréquente en production si l'API finale doit rester feature-level;
- classification non calibrée: moyenner des probabilités de combos corrélés peut produire une confiance mal calibrée.

### Branching par source

Le branching par source pose une question différente du cartésien. Il y a deux sémantiques possibles:

1. Branches de features: chaque branche prétraite une source, puis on merge les features.
2. Branches de modèles: chaque branche entraîne un modèle source-specific, puis on merge les prédictions ou on stacke.

Avec répétitions hétérogènes:

- une branche MIR a 2 observations par sample;
- une branche RAMAN en a 3;
- une branche NIRS en a 2.

Un merge feature-level ne peut pas juste concaténer par position. Il faut soit réduire chaque source au niveau `sample_id`, soit créer un cartésien explicite, soit utiliser un modèle ragged/multi-instance.

Un merge prediction-level est plus naturel: chaque modèle source prédit ses répétitions, puis ses prédictions sont agrégées par `sample_id`; le stacker reçoit une ligne par échantillon avec, par exemple, `pred_MIR`, `pred_RAMAN`, `pred_NIRS`. Ce chemin limite l'explosion combinatoire et donne un stacking plus lisible.

Le critère de choix entre merge feature-level et merge prediction-level doit être explicite:

- si on veut apprendre des interactions entre features brutes de sources différentes, le merge feature-level reste nécessaire, donc il faut choisir une représentation alignée (`per_source_aggregate`, `stack_fixed`, cartésien complet ou Monte-Carlo);
- si chaque source peut fournir un signal prédictif autonome, la fusion tardive évite l'appariement, accepte mieux les répétitions hétérogènes et supporte plus naturellement les sources manquantes en prédiction.

### Choix des modèles à refit

Le refit doit répondre à deux questions qui sont confondues aujourd'hui dans le cas simple:

- Quel pipeline a le meilleur score de validation?
- À quel niveau ce score a-t-il été calculé?

Pour ce problème, le score de sélection doit être sample-level. Un pipeline ne doit pas être choisi parce qu'il a le meilleur RMSE combo-level ou repetition-level si le livrable attendu est une prédiction par échantillon physique.

Cas difficiles:

- Modèle source-specific: il peut être sélectionné sur un score source-level agrégé par `sample_id`.
- Modèle fusion feature-level: il doit être sélectionné sur le score sample-level après agrégation des combos, si cartésien.
- Stacking: le modèle meta doit être sélectionné sur des méta-features OOF sample-level. Les modèles de base peuvent être sélectionnés soit indépendamment par source, soit dans le contexte du stacker. Ces deux politiques donnent des résultats différents et doivent être déclarées.
- Refit final: le méta-modèle ne devrait pas être réentraîné sur des prédictions in-sample des bases sans opt-in explicite. Le contrat doit être explicite via `StackingFitContract`: `meta_training_features=oof`, `inference_features=refit_base_predictions`, `selection_protocol=nested|reuse_oof|holdout`, `base_prediction_calibration=none|rank|calibrated_oof_to_refit`. La variante saine est nested ou holdout; `reuse_oof` est un mode exploratoire optimiste et doit être interdit pour un profil de validation finale.

Il y a trois niveaux de sélection à ne pas confondre:

| Niveau | Ce qu'il choisit | Risque |
| --- | --- | --- |
| Branch-level | meilleur modèle par source | peut éliminer un modèle faible seul mais complémentaire en stack |
| Pipeline global | meilleure combinaison complète sur métrique sample-level | coûte cher si l'espace branches x meta est grand |
| Meta-aware / nested | bases et stacker choisis pour leur contribution OOF au meta-modèle | nécessite une CV imbriquée ou un protocole OOF strict |

Le chemin stacking le plus propre est nested: inner CV pour choisir/entraîner les bases et produire des OOF, outer CV pour estimer le stacker. Si on sélectionne les bases puis le stacker sur les mêmes OOF, on introduit une fuite de sélection, même si aucune ligne de validation brute n'est directement en train.

### Groupes, OOF et fuite de données

Les groupes peuvent venir de plusieurs contraintes:

- `sample_id`: répétitions du même échantillon;
- `batch_id`: même lot expérimental;
- `donor_id`, `parcel_id`, `site_id`, etc.;
- groupes métier explicitement fournis par l'utilisateur.

La bonne règle n'est pas de grouper par tuple strict `(sample_id, batch_id)`, mais de construire des composantes connexes: si deux lignes partagent `sample_id` ou `batch_id`, elles doivent rester dans le même fold. nirs4all possède déjà cette logique pour le cas plat via `compute_effective_groups()`.

Pour les OOF, les invariants à imposer sont:

- aucune prédiction OOF d'une ligne, combo ou sample ne vient d'un modèle entraîné sur le même `sample_id`;
- les prédictions de branches destinées au stacking ont le même domaine d'alignement, idéalement `sample_id`;
- les folds couvrent les mêmes `sample_id` entre branches lorsque le stacker les joint strictement;
- toute imputation de prédiction manquante doit être une politique explicite et visible dans le score.
- les avertissements actuels de leakage/alignment doivent devenir des erreurs dans ce profil; un warning n'est pas une barrière de validation suffisante.

### Cas limites à traiter explicitement

- Source absente pour un sample.
- Zéro répétition valide pour une source.
- Nombre de répétitions différent entre train et prediction set.
- Répétition dupliquée `(sample_id, source_id, rep_id)`.
- Répétitions désordonnées ou `rep_id` non numériques.
- Targets différentes entre répétitions du même sample.
- Metadata sample-level contradictoires entre sources.
- Metadata source-level ou repetition-level mélangées à du sample-level.
- Outlier removal par source versus global.
- Combinaison cartésienne trop grande.
- Augmentations spectrales confondues avec répétitions physiques.
- Multi-target ou classification avec probas/vote.
- Source order différent entre fichiers mais même nombre de lignes.
- Stacking avec OOF incomplets après branches source-specific.
- Refit automatisé basé sur `best_val` row-level au lieu de score agrégé.

## 2. Fonctionnement actuel du workflow nirs4all

### Modèle de données et multisource

Le modèle `Features` est construit autour de sources alignées. Sa docstring annonce qu'il gère des sources NumPy alignées (`nirs4all/data/features.py:9`), `num_samples` vient de la première source (`nirs4all/data/features.py:123`), et l'extraction `x()` applique les mêmes indices à chaque source avant concaténation (`nirs4all/data/features.py:267`).

Ce contrat est compatible avec:

- plusieurs sources ayant des nombres de features différents;
- plusieurs sources déjà réduites à une ligne par échantillon;
- plusieurs sources ayant le même nombre de répétitions alignées globalement.

Il n'est pas compatible avec des sources qui ont chacune leur propre table de mesures.

Point de vigilance majeur: les configs multisource exposent `link_by` dans le schéma (`nirs4all/data/schema/config.py:680`) et la documentation du parseur parle d'alignement par clé (`nirs4all/data/parsers/files_parser.py:354`), mais ce lien n'est pas une garantie. La conversion legacy conserve les sources dans un blob interne `_sources`, qui n'est pas utilisé par le loader et n'est pas sérialisé comme contrat stable. Le loader charge les targets/metadata depuis la première source puis les autres sources comme X seulement (`nirs4all/data/loaders/loader.py:399`). `Features.add_samples` vérifie le nombre de sources et les métadonnées de features, pas l'égalité des nombres de lignes inter-sources. Conséquence: un dataset `MIR=2N`, `RAMAN=3N`, `NIRS=2N` peut entrer dans un état incohérent au lieu de lever une erreur claire.

### `repetition` et `aggregate`

`DatasetConfigs` porte des listes de `repetition` et `aggregate` pour plusieurs datasets, mais chaque dataset n'a qu'un seul axe de répétition et un seul axe d'agrégation; il n'y a pas de répétition par source (`nirs4all/data/config.py:24`). Lorsque `repetition` est défini, l'agrégation de prédictions est activée automatiquement dans certains cas (`nirs4all/data/config.py:343`). Inversement, une colonne `aggregate` peut devenir une colonne `repetition` (`nirs4all/data/config.py:349`).

`SpectroDataset` stocke une seule colonne de répétition (`nirs4all/data/dataset.py:719`) et une seule configuration d'agrégation (`nirs4all/data/dataset.py:837`). `repetition_stats` compte les répétitions par groupe sur l'axe ligne global (`nirs4all/data/dataset.py:782`). Il ne sait pas produire une table "MIR a 2, RAMAN a 3, NIRS a 2".

### `rep_to_sources` et `rep_to_pp`

Les opérateurs de répétition actuels sont conçus pour un `n_reps` global:

- `RepetitionConfig` a un seul `column` et un seul `expected_reps` (`nirs4all/operators/data/repetition.py:61`).
- `reshape_reps_to_sources()` calcule ou valide un seul `n_reps`, puis crée `n_sources * n_reps` nouvelles sources (`nirs4all/data/dataset.py:1101`).
- `reshape_reps_to_preprocessings()` applique la même logique en dimension preprocessing (`nirs4all/data/dataset.py:1191`).
- `_validate_repetition_groups()` supporte `error`, `drop`, `truncate`, `pad`, mais sur les répétitions globales d'une même colonne (`nirs4all/data/dataset.py:1013`).

Ces opérateurs sont utiles pour "toutes les sources ont N répétitions alignées". Ils ne représentent pas "`MIR=2`, `RAMAN=3`, `NIRS=2`".

Un contournement consistant à forcer `expected_reps=3` est dangereux: il padde/tronque toutes les sources vers la même cardinalité et perd la distinction entre répétition source-specific et répétition multisource.

### Split et groupes

Le split est la partie la plus saine du workflow actuel pour le cas plat. `compute_effective_groups()` combine `dataset.repetition` et `group_by` par composantes connexes (`nirs4all/controllers/splitters/split.py:192`). Le contrôleur de CV stocke des sample IDs de train/validation et détecte les fuites de groupes (`nirs4all/controllers/splitters/split.py:602`, `nirs4all/controllers/splitters/split.py:741`), mais certains contrôles émettent aujourd'hui des warnings plutôt que des erreurs. Pour ce profil, ces warnings doivent devenir des refus explicites.

Mais cette protection suppose que l'axe ligne représente déjà toutes les répétitions à grouper. Si MIR, RAMAN et NIRS ont des tables séparées, il faut d'abord créer une identité canonique `sample_id` qui traverse toutes les sources.

### Branching et merge

`branch by_source` existe (`nirs4all/controllers/data/branch.py:720`), mais c'est une branche de traitement par source dans un dataset déjà aligné. Elle ne transforme pas des tables ragged en domaines compatibles.

Les merges de features supposent aussi un alignement par position:

- `_collect_features()` raisonne principalement sur des comptes de samples (`nirs4all/controllers/data/merge.py:3065`);
- `_extract_features_from_snapshot()` concatène par position (`nirs4all/controllers/data/merge.py:3193`);
- `_merge_sources_concat()` et `_merge_sources_stack()` vérifient les tailles puis font `np.concatenate` ou `np.stack` (`nirs4all/controllers/data/merge.py:4892`);
- `add_merged_features()` exige que le résultat ait `dataset.num_samples` lignes (`nirs4all/data/dataset.py:284`).

Donc `branch by_source -> merge_sources` ne résout pas le cas `2/3/2` si les sources n'ont pas déjà été réduites ou expansées vers un axe commun.

Le merge de prédictions OOF est moins primitif: certains chemins reconstruisent une matrice de prédictions en scatter par identifiants de samples internes (`nirs4all/controllers/data/merge.py:3993`). C'est un bon point de départ pour la fusion tardive, mais ces identifiants restent des IDs internes au dataset courant, pas une clé physique stable traversant loader, workspace, refit et prédiction.

### Agrégation de prédictions

L'agrégation actuelle est principalement une agrégation de reporting/prédictions, pas une transformation de `X_train`:

- `Predictions._apply_aggregation()` groupe par une colonne metadata ou par `y` (`nirs4all/data/predictions.py:1537`);
- `Predictions.aggregate()` agrège `y_pred`, `y_true` et éventuellement `y_proba` par une liste de groupes plate (`nirs4all/data/predictions.py:2178`);
- `top(by_repetition=True)` se résout vers une colonne de répétition unique (`nirs4all/data/predictions.py:1477`).

Cela ne décrit pas une agrégation hiérarchique `observation -> source -> sample`, ni une agrégation `combo -> sample`.

Autre écart à documenter: l'option outlier est décrite côté config comme Hotelling T² (`nirs4all/data/config.py:89`), mais l'implémentation actuelle fonctionne sur les prédictions avec un modified Z-score et ignore les groupes de taille <= 2 (`nirs4all/data/predictions.py:2310`). Ce n'est pas forcément mauvais, mais ce n'est pas l'outlier removal spectral source-level dont on aurait besoin ici.

### Entraînement, OOF, stacking

Les modèles consomment directement `dataset.x()` via `BaseModelController.get_xy()` (`nirs4all/controllers/models/base_model.py:248`). Le fait de définir `aggregate` ne transforme pas automatiquement l'entraînement en sample-level.

Le stacking est prudent dans le cas standard: `MetaModelController` utilise `TrainingSetReconstructor`, qui reconstruit des méta-features à partir des prédictions OOF (`nirs4all/controllers/models/meta_model.py:743`, `nirs4all/controllers/models/stacking/reconstructor.py:460`). Le reconstructor aligne par indices de samples/folds et évite qu'une prédiction voie un modèle entraîné sur le même sample.

Limites pour le cas hétérogène:

- les IDs sont des IDs internes du dataset courant, souvent auto-incrémentés via le gestionnaire de samples, pas des `sample_id` physiques first-class persistés;
- la validation de folds est insuffisante pour garantir une égalité stricte des ensembles de `sample_id` entre branches source-specific;
- certains chemins de merge de prédictions utilisent `validate_fold_alignment=False` et `IMPUTE_MEAN` (`nirs4all/controllers/data/merge.py:3933`);
- `merge.unsafe=True` est explicitement un chemin à risque de fuite et doit être interdit pour ce profil (`nirs4all/operators/data/merge.py:643`, `nirs4all/controllers/data/merge.py:4049`).

### Refit et sélection

Le refit simple remplace le splitter par un fold full-train et réinjecte les meilleurs paramètres (`nirs4all/pipeline/execution/refit/executor.py:54`). La sélection de config utilise `best_val` / `rmsecv` ou une moyenne des `val_score` (`nirs4all/pipeline/execution/refit/config_extractor.py:127`, `nirs4all/pipeline/execution/refit/config_extractor.py:246`).

Pour les répétitions hétérogènes, cela n'est correct que si `best_val` est déjà le score sample-level voulu. Si `best_val` est row-level, repetition-level ou combo-level, le refit consacre le mauvais critère.

Le refit stacking actuel refit les branches de base, collecte des prédictions in-sample, puis entraîne le méta-modèle final (`nirs4all/pipeline/execution/refit/stacking_refit.py:688`, `nirs4all/pipeline/execution/refit/stacking_refit.py:889`). Dans un contexte de répétitions hétérogènes, ce mode doit être considéré expérimental tant qu'il ne garantit pas une cohérence entre les méta-features OOF de sélection et les méta-features de refit.

## 3. Choix de design pour un support général

### Invariants non négociables

Un design correct doit imposer ces invariants:

- une clé sample externe et stable est l'unité de CV par défaut; l'id interne auto-incrémenté de nirs4all ne suffit pas.
- Les sources ne sont jamais jointes par position si une clé `sample_id` existe.
- Toute transformation qui crée ou détruit des lignes conserve une provenance stable, idéalement en réutilisant le mécanisme existant d'origine/augmentation plutôt qu'un second mécanisme parallèle.
- Les prédictions déclarent leur niveau: `observation`, `source`, `combo`, `sample`.
- Le ranking, le refit et les rapports finaux utilisent explicitement un `evaluation_scope`.
- Les cartésiens déclarent leur politique d'influence de fit; le score/refit reste sample-level pour ne pas surpondérer les samples riches en répétitions.
- Le groupement par sample physique est auto-activé dès qu'une représentation produit plusieurs lignes par sample; transformer des warnings en erreurs ne suffit pas si aucun groupe n'a été fourni.
- Les chemins OOF/stacking refusent les alignements incomplets par défaut.
- Les stratégies de missing source, missing rep et imputation sont explicites.
- `rep_fusion`, `rep_to_sources`, `rep_to_pp` et `repetition=` ont une matrice d'exclusivité/priorité explicite; ils ne doivent pas se composer implicitement.
- Les transformations cross-sample sont fit sur le train uniquement. Si `rep_fusion` change le nombre de lignes, les folds doivent être remappés par clé sample stable.
- Toute sélection/troncature de combinaisons cartésiennes est déterministe, seedée et incluse dans le fingerprint/run manifest.
- Les opérateurs de répétitions source-aware rejouables en production doivent déclarer `supports_prediction_mode=True`; reprendre le contrat actuel de `rep_to_sources`/`rep_to_pp`, qui sont skippés en prédiction, créerait un bundle impossible à servir correctement.
- Les scores et prédictions exposés déclarent séparément `training_unit`, `prediction_level` et `evaluation_scope`.

### Taxonomie opérationnelle des stratégies

Cette table est volontairement plus prescriptive que les options A-F historiques: elle force chaque famille de solution à déclarer ses unités, son comportement en refit et son coût.

| Stratégie | Fit | OOF / validation | Refit | Prédiction | Merge / stack | Pondération | Où c'est valide | Risques principaux |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 garde-fous | inchangé, mais erreurs sur configs impossibles | inchangé | inchangé | inchangé | inchangé | n/a | toujours | casse des configs qui étaient silencieusement fausses |
| S1 pré-agrégation par source | 1 vecteur par `(sample, source)` | row-level devient sample-level car 1 ligne/sample | standard | rejoue mean/median sur reps disponibles | feature merge classique | n/a ou poids sample | jalon technique simple, modèles peu non linéaires | biais `f(E[x])`, perte variance |
| S2 stack fixe | slots fixes par source | sample-level natif | standard | exige cardinalités et ordre attendus | feature merge classique | n/a | reps ordonnées/stables | fragile aux reps manquantes, dimension élevée |
| S3a cartésien complet | toutes les combinaisons comme lignes | OOF combo, score après `combo -> sample` | full train sur combos avec politique d'influence déclarée | énumère combos puis agrège | feature merge par combo | optionnel selon `FitInfluencePolicy` | interactions brutes, petit N, modèles non linéaires | explosion, calibration, cardinalités variables |
| S3b cartésien train-only via augmentation | base agrégée + combos comme augmentations `origin=sample` | validation sur base, combos seulement en train | bases agrégées + augmentations | base agrégée uniquement | feature merge classique | souhaitable mais moins bloquant | cartésien rapide, régularisation | mismatch train/serve assumé |
| S3c Monte-Carlo cartésien | `K` combos seedés par sample | idem S3a ou S3b | idem mode choisi | idem mode choisi | idem mode choisi | selon `FitInfluencePolicy` | produit trop grand | biais si sélection déterministe |
| S4 fusion tardive par source | modèles source-level sur observations | OOF observation -> sample par source, puis stack sample-level | meta entraîné sur OOF, bases full-train | agrège chaque source, stack sample-level | prediction merge par `sample_id` | optionnel par fiabilité/n_reps | défaut ML général, sources manquantes possibles | ne voit pas interactions brutes |
| S5 stacking combo-level | meta reçoit combos | OOF combo strictement groupé | très délicat | combos puis meta puis agrégation | prediction merge par combo | politique explicite obligatoire | rare, expérimental | mélange d'unités, pseudo-réplication meta |
| S6 ragged / multi-instance | modèle consomme bags par source | sample-level natif | modèle ragged full-train | bags variables + masks | merge côté modèle | appris ou loss-level | long terme, sources/reps variables | refactor data/model/export majeur |

Lecture pratique:

- S1 est le jalon technique le plus simple parce qu'il produit une matrice alignée et ne dépend pas d'une refonte complète du scoring: une ligne vaut déjà un sample.
- S4 doit être livré dans le même périmètre fonctionnel que S1, car la fusion tardive est nécessaire pour exploiter les sources séparées, les répétitions disponibles et les sources partiellement manquantes sans imposer d'appariement feature-level.
- S3a/S3c ne sont pas des hacks: ils sont la famille correcte si on veut approximer `E[f(x)]` ou apprendre des interactions brutes, mais ils exigent les contrats d'influence de fit, de groupes et de métriques.
- S3b est une option intermédiaire très intéressante pour aller vite: les combos deviennent des augmentations d'entraînement. nirs4all possède déjà une notion d'`origin` pour empêcher les augmentations de fuir en validation; il faut la réutiliser plutôt que créer une provenance parallèle.
- S5 doit être refusé par défaut. Agréger `combo -> sample` avant le stacker est presque toujours plus défendable.

### Option A - Pré-agrégation par source

Principe: pour chaque source et chaque `sample_id`, réduire les répétitions à un seul vecteur. Exemple: moyenne, médiane, trimmed mean, QC outlier puis moyenne.

Avantages:

- chemin le plus direct pour produire un dataset aligné compatible avec nirs4all actuel;
- compatible avec le modèle `Features` actuel après transformation;
- compatible avec `branch by_source`, `merge_sources`, modèles sklearn, OOF et refit existants;
- le score final est naturellement sample-level.

Coûts:

- perte d'information intra-source;
- l'incertitude des répétitions n'est pas directement donnée au modèle;
- biais potentiel `f(E[x])` pour les modèles non linéaires si la quantité scientifiquement voulue est `E[f(x)]`;
- les méthodes d'agrégation doivent être traçables et, si elles apprennent des paramètres, fit uniquement sur le train fold.

Recommandation: c'est le mode le plus simple pour fermer le trou de sécurité immédiat et produire un dataset aligné. Il ne dépend pas d'une refonte du scoring parce qu'il ramène le problème à une ligne par sample. Il faut toutefois le présenter comme un compromis rapide et robuste, pas comme la vérité scientifique universelle.

### Option B - Empilement fixe par source

Principe: transformer `MIR=2` en un bloc de features `[MIR_rep1, MIR_rep2]`, `RAMAN=3` en `[RAMAN_rep1, RAMAN_rep2, RAMAN_rep3]`, etc., puis une ligne par sample.

Avantages:

- conserve plus d'information que la moyenne;
- produit une matrice rectangulaire compatible avec le code actuel si les cardinalités sont fixes;
- utile lorsque l'ordre des répétitions a un sens stable.

Coûts:

- nécessite `expected_reps_by_source`;
- fragile avec répétitions manquantes;
- impose une politique `pad`, `drop`, `truncate` par source;
- augmente fortement la dimensionnalité.

Recommandation: mode utile après `per_source_aggregate`, mais ne doit pas être le défaut.

### Option C - Cartésien multisource

Principe: créer des combinaisons de répétitions entre sources et entraîner un modèle fusion feature-level sur ces combos. Cette option recouvre trois stratégies différentes qui ne doivent pas être confondues.

#### Option C1 - Cartésien complet

Le cartésien complet génère toutes les combinaisons `MIR_i x RAMAN_j x NIRS_k`, les utilise au fit/refit, les prédit en validation et en production, puis agrège les prédictions `combo -> sample`.

Avantages:

- permet au modèle d'apprendre des interactions directes entre sources;
- conserve toutes les répétitions;
- approxime `E[f(x)]` lorsque les prédictions des combos sont agrégées, ce qui peut être moins biaisé que `f(E[x])` pour des modèles non linéaires;
- reste compatible avec `Features` après expansion.

Coûts:

- explosion combinatoire;
- pseudo-réplication si métriques row-level;
- politique d'influence de fit obligatoire, pondération seulement si cette politique l'exige;
- score final à agréger `combo -> sample`;
- OOF et refit plus difficiles;
- risque de comparer des modèles sur des unités différentes;
- `sample_weight` n'est pas câblé aujourd'hui dans les contrôleurs modèles; il faut une plomberie transverse selon les backends pour les politiques qui exigent une influence égale des samples au fit.
- la génération/troncature des combos peut casser la reproductibilité si elle n'est pas déterministe.
- calibration classification plus délicate: moyenner des probabilités de combos très corrélés peut donner une confiance mal calibrée.

Recommandation: ce n'est pas le défaut général, mais ce n'est pas une option à écarter. C'est le bon choix si l'on veut des interactions feature-level brutes ou une approximation explicite de `E[f(x)]`. Il doit exiger `group_by=sample_id`, `combo_id`, `origin_sample_id`, `FitInfluencePolicy` explicite, reducers `combo -> sample`, et `evaluation_scope="sample"`.

#### Option C2 - Cartésien train-only via augmentation

Principe: créer une ligne de base agrégée par sample, puis ajouter des combos cartésiens comme augmentations d'entraînement avec `origin=physical_sample_id`.

Avantages:

- réutilise la mécanique d'augmentation existante: les augmentations sont ajoutées au train mais exclues des validations lorsque les folds sont remappés par `origin`;
- permet d'injecter la variabilité des répétitions au fit sans changer la représentation de validation, de test et de prédiction;
- évite de payer le cartésien au déploiement;
- ne dépend pas immédiatement d'un scoring combo-level, car la validation reste sur une ligne par sample.

Coûts:

- mismatch assumé entre train et serve: le modèle voit des combos bruts en train mais une ligne agrégée en prédiction;
- pondération souhaitable pour éviter que les samples riches en combos reçoivent trop d'augmentations;
- ne produit pas directement l'estimateur `E[f(x)]` au moment de la prédiction.

Recommandation: c'est probablement le cartésien le plus rapide à livrer après `per_source_aggregate`. Il doit réutiliser `origin` plutôt que créer une provenance parallèle, mais il faut documenter qu'il s'agit d'une régularisation par augmentation, pas d'une inférence cartésienne complète.

#### Option C3 - Monte-Carlo cartésien

Principe: échantillonner `K` combinaisons par sample au lieu de toutes les énumérer.

Avantages:

- contrôle le coût sans abandonner l'idée de marginaliser sur les répétitions;
- si `combo_selection=random_seeded`, donne un estimateur Monte-Carlo reproductible de `E[f(x)]`;
- permet de choisir `K` selon le budget mémoire/temps.

Coûts:

- `deterministic_first` est biaisé et doit servir au debug ou à la reproductibilité stricte, pas comme estimateur scientifique par défaut;
- la variance Monte-Carlo doit être contrôlée, surtout si peu de combos sont tirés;
- la graine, la méthode et `K` doivent être persistés dans le manifest.

Recommandation: dès que le produit des répétitions devient grand, préférer `random_seeded` avec `max_combos_per_sample=K` à une troncature arbitraire.

### Option D - Fusion tardive par prédictions source-level

Principe: entraîner un modèle par source sur ses répétitions, prédire au niveau observation, agréger observation -> sample par source, puis stacker ou merger les prédictions sample-level.

Avantages:

- très pertinent scientifiquement;
- évite entièrement le problème d'appariement;
- chaque source garde son propre nombre de répétitions;
- le stacker voit un domaine simple: une ligne par sample.
- le chemin de merge de prédictions OOF nirs4all contient déjà une reconstruction sample-indexed, à durcir plutôt qu'à inventer.
- meilleur comportement en prédiction avec source manquante: une feature-level concat casse en largeur, alors qu'un stacker peut avoir une politique explicite `strict`, `drop_incomplete`, `impute_declared` ou modèle partiel.

Coûts:

- demande des OOF source-level propres;
- `TrainingSetReconstructor` doit aligner par `sample_id` et pas seulement par indices de lignes;
- les branches avec source manquante doivent être gérées explicitement;
- le refit stacking doit être durci;
- ne capture pas les interactions directes entre features brutes de sources différentes.

Recommandation: jalon à inclure dans le même périmètre fonctionnel que `per_source_aggregate`. C'est probablement le meilleur compromis ML pour des datasets réels quand les interactions brutes inter-sources ne sont pas indispensables, mais il impose une clé sample physique stable, une couverture OOF stricte et un refit stacking corrigé.

### Option E - Modèle ragged / multi-instance / multi-view

Principe: représenter chaque sample comme un dictionnaire de bags:

```text
sample_id -> {
  MIR:   [spectre_1, spectre_2],
  RAMAN: [spectre_1, spectre_2, spectre_3],
  NIRS:  [spectre_1, spectre_2],
}
```

Le modèle ou un encodeur par source agrège les répétitions par pooling, attention, set transformer, etc.

Avantages:

- modèle conceptuel le plus fidèle;
- support naturel des répétitions variables et sources manquantes;
- compatible avec une approche DAG-ML relationnelle.

Coûts:

- refactoring majeur de `Features`, indexer, contrôleurs modèles, stockage prédictions et exports;
- moins compatible avec les modèles sklearn classiques;
- demande de nouvelles conventions de métriques et d'explicabilité.

Recommandation: cible long terme, pas chemin rapide.

### Option F - DAG-ML comme couche relationnelle

L'analyse DAG-ML indique que `../dag-ml` possède déjà plusieurs primitives alignées avec ce problème:

- `SampleRelation` / `SampleRelationSet` pour relier observations, samples, targets, groupes et origine (`dag-ml/crates/dag-ml-core/src/relation.rs:17`);
- `LeakageUnitPolicy` avec `split_unit=Sample` par défaut et refus des splits observationnels dangereux sans opt-in (`dag-ml/crates/dag-ml-core/src/policy.rs:17`);
- `PredictionLevel` incluant `Observation` et `Sample` (`dag-ml/crates/dag-ml-core/src/policy.rs:67`);
- pour le chemin principal, les combos peuvent rester des observations dérivées dans les relations; ajouter `PredictionLevel=Combo` est une extension publique différée, pas un prérequis;
- `ObservationPredictionBlock` et `AggregatedPredictionBlock` pour représenter observation-level puis sample-level (`dag-ml/crates/dag-ml-core/src/aggregation.rs:20`, `dag-ml/crates/dag-ml-core/src/aggregation.rs:63`);
- `aggregate_observation_predictions()` et `aggregate_sample_predictions_by_unit()` pour exécuter l'agrégation observation -> sample ou sample -> target/group (`dag-ml/crates/dag-ml-core/src/aggregation.rs:546`, `dag-ml/crates/dag-ml-core/src/aggregation.rs:680`);
- `DataModelShapePlan` pour porter `input_granularity`, `target_granularity` et les politiques d'agrégation/sélection au niveau du noeud (`dag-ml/crates/dag-ml-core/src/policy.rs:379`);
- `EdgeContract.requires_oof` et `requires_fold_alignment` (`dag-ml/crates/dag-ml-core/src/graph.rs:80`);
- `DataBinding.require_relations`, qui existe déjà sur le binding plutôt que sur l'arête (`dag-ml/crates/dag-ml-core/src/data.rs:568`);
- `BranchViewPlan` et `DataProviderViewSpec` pour matérialiser des vues par source/fold (`dag-ml/crates/dag-ml-core/src/data.rs:108`, `dag-ml/crates/dag-ml-core/src/runtime.rs:5936`).
- dans l'écosystème `../dag-ml-data`, `fuse_feature_blocks()` conserve les répétitions de la source de référence mais refuse de broadcaster plusieurs lignes répétées d'une source non référence sur le même sample (`dag-ml-data/crates/dag-ml-data-core/src/fusion.rs:49`, `dag-ml-data/crates/dag-ml-data-core/src/fusion.rs:182`). Ce refus est une information de design: une fusion feature-level ragged N-sources-à-répétitions-multiples n'est pas une simple jointure de données.

Le flux DAG-ML recommandé est:

1. `DataBinding(require_relations=true)` expose les relations sample/observation/source.
2. `Branch(mode=by_source)` matérialise une vue par source.
3. Les modèles source prédisent au niveau observation en validation.
4. Une `AggregationPolicy` agrège observation -> sample.
5. `MergeModel` ou stacking consomme seulement des prédictions OOF sample-level.

Manques DAG-ML à combler:

- les arêtes ne déclarent pas encore assez explicitement `unit_level` ou `alignment_key`; une partie du contrat vit aujourd'hui sur `DataBinding`, `LeakageUnitPolicy`, `AggregationPolicy` et `DataModelShapePlan`, pas sur `EdgeContract`;
- la missing-source policy n'est pas encore un contrat de port/arête explicite;
- `BranchViewPlan` est encore porté comme métadonnée DSL/runtime dans certains chemins, pas comme sémantique native partout;
- la façade Python manque de builders simples pour `SampleRelationSet`, `DataBinding` et enveloppes de données;
- les joins feature/source devraient refuser les domaines incompatibles sans réduction ou expansion explicite.
- la vraie solution ragged doit vivre côté modèle/collation avec masques de présence, pas comme une tentative de broadcaster arbitrairement plusieurs répétitions de plusieurs sources dans un bloc feature unique.

Recommandation: utiliser DAG-ML comme référence de design pour les invariants et préparer un export nirs4all -> DAG-ML plutôt que reproduire toute la sémantique relationnelle à la main dans les contrôleurs nirs4all. Pour la fusion de features hétérogènes, l'écosystème DAG-ML pousse déjà vers agrégation préalable, ports séparés ou modèle ragged, pas vers une jointure magique de toutes les répétitions.

### Config proposée

Ne pas surcharger le `repetition: "sample_id"` actuel. Ajouter une forme source-aware explicite:

```yaml
dataset:
  sample_id: sample_id
  repetitions:
    kind: per_source
    sample_id: sample_id
    target_level: sample
    on_missing_source: error
    on_missing_repetition: error
    per_source:
      MIR:
        rep_col: mir_rep
        expected: 2
      RAMAN:
        rep_col: raman_rep
        expected: 3
      NIRS:
        rep_col: nirs_rep
        expected: 2
```

Exemple `per_source_aggregate`:

```yaml
pipeline:
  - rep_fusion:
      mode: per_source_aggregate
      group_by: sample_id
      sources:
        MIR:   {method: mean}
        RAMAN: {method: median}
        NIRS:  {method: mean}
      output: sources
      keep_repetition_stats: true
  - branch:
      by_source: true
      steps: ...
  - merge:
      sources: concat
  - model: ...
```

Exemple cartésien:

```yaml
pipeline:
  - rep_fusion:
      mode: cartesian
      group_by: sample_id
      sources: [MIR, RAMAN, NIRS]
      combo_id: rep_combo
      fit_influence_policy: equal_sample_influence
      max_combos_per_sample: 64
      combo_selection: random_seeded
      random_state: 42
      output: sources
  - split:
      method: kfold
      group_by: sample_id
  - model: ...
  - evaluate:
      scope: sample
      reduction:
        role: score
        axis: unit
        from: combo
        to: sample
        method: mean
```

Exemple cartésien train-only via augmentation:

```yaml
pipeline:
  - rep_fusion:
      mode: cartesian_augmentation
      base: per_source_aggregate
      group_by: sample_id
      combo_id: rep_combo
      combo_selection: random_seeded
      max_combos_per_sample: 16
      use_origin_for_combos: true
      fit_influence_policy: equal_sample_influence
      predict_representation: base
  - split:
      method: kfold
      group_by: sample_id
      include_augmented: false
  - model: ...
```

Exemple fusion tardive:

```yaml
pipeline:
  - branch:
      by_source: true
      unit: observation
      group_by: sample_id
      steps:
        - preprocess: ...
        - model: ...
        - reduce_predictions:
            role: meta_feature
            axis: unit
            from: observation
            to: sample
            method: mean
  - merge:
      predictions: stack
      alignment_key: sample_id
      require_oof: true
      coverage: strict
  - meta_model: ...
```

### Schéma v1 expérimental

Le premier schéma utilisateur doit être volontairement minimal et derrière un feature flag, par exemple `experimental_relation_pipeline: true`, pour éviter de casser la syntaxe legacy.

Grammaire minimale proposée:

```yaml
relations:
  sample_id: sample_id        # alias utilisateur de physical_sample_id
  target_level: physical_sample
  sources:
    MIR:   {rep_col: mir_rep, expected: 2}
    RAMAN: {rep_col: raman_rep, expected: 3}
    NIRS:  {rep_col: nirs_rep, expected: 2}

representations:
  cartesian_train:
    kind: cartesian_full
    unit: combo
    combination:
      mode: cartesian
      max_per_sample: 64
      seed: 42
    fit_influence_policy: equal_sample_influence

reducers:
  combo_score:
    role: score
    axis: unit
    from: combo
    to: physical_sample
    method: mean

stacking:
  meta_row_domain: physical_sample
  fit_contract:
    meta_training_features: oof
    inference_features: refit_base_predictions
    selection_protocol: nested
    base_prediction_calibration: rank
```

Mapping v1:

- `relations` construit `NormalizedObservationTable`;
- `representations.*` appelle `rep_fusion` ou un futur contrôleur de représentation;
- `reducers.*` est traduit vers `ReductionPlan`;
- `fit_influence_policy` remplace toute option `combo_weighting`;
- `sample_id` en YAML est normalisé en `physical_sample_id`;
- erreur explicite si `experimental_relation_pipeline=false` et que les sources ont des axes incompatibles.

Les blocs déclaratifs `relations`, `representations`, `reducers` et `stacking` sont la syntaxe canonique à stabiliser. Les exemples inline plus haut (`rep_fusion`, `evaluate`, `reduce_predictions`) sont des projections utilisateur/legacy vers ces blocs; `evaluate` ne doit pas devenir un noeud de fit implicite, seulement une déclaration de scoring/réduction post-prédiction.

Tests de parser attendus: round-trip YAML, aliases `sample_id`/`physical_sample_id`, refus de `combo_weighting`, refus de `rep_to_sources` + `rep_fusion`, erreur si `cartesian_full` dépasse le cap global sans `cartesian_mc`, et migration legacy inchangée.

### Politique ML recommandée

Par défaut:

- split par `sample_id`, et par composantes connexes si `group_by` métier existe;
- métrique finale sample-level;
- ranking/refit basés sur métrique sample-level;
- `merge.unsafe=False` obligatoire;
- couverture stricte ou `drop_incomplete` explicite pour stacking;
- pas d'imputation silencieuse dans les scores de validation finaux;
- politique d'influence explicite pour tout entraînement row-level issu de répétitions ou combos.
- erreurs strictes, pas warnings, pour les fuites de groupes et les alignements incomplets dans les runs de validation finale.

Quand utiliser chaque mode:

| Situation | Mode recommandé | Raison |
| --- | --- | --- |
| Besoin de déployer vite et proprement | `per_source_aggregate` | Produit un dataset aligné sample-level |
| Répétitions fixes et ordre stable | `stack_fixed` par source | Conserve l'information sans cartésien |
| Besoin d'interactions feature-level entre sources | `cartesian` complet ou Monte-Carlo | Seule famille qui expose les features brutes croisées |
| Petit N, modèle non linéaire, besoin d'augmentation rapide | `cartesian_augmentation` | Réutilise `origin` et les rails d'augmentation |
| Sources très différentes, modèles spécialisés | late fusion / stacking sample-level | Meilleur compromis scientifique sans appariement |
| Source manquante possible en production | late fusion ou ragged | Dégrade mieux qu'une concat feature-level |
| Sources/répétitions très variables | modèle ragged / DAG-ML | Modèle de données plus fidèle |

## 4. Impact sur les contrats stables

Le support des répétitions hétérogènes ne peut pas rester une transformation locale de pipeline. Il touche des contrats stables du projet: workspace, bundle exporté, API de prédiction, ranking et exports. C'est bloquant pour passer de note de cadrage à blueprint d'implémentation.

### Bundle `.n4a` et prédiction

Un modèle exporté doit pouvoir rejouer exactement la même sémantique en prédiction:

- `RepetitionSpec` utilisée à l'entraînement;
- mapping clé sample physique -> observations par source;
- méthode d'agrégation par source ou de génération cartésienne;
- représentation de prédiction: base agrégée, combos complets, Monte-Carlo seedé, ou late fusion par source;
- politique de missing source/repetition;
- politique d'outlier/QC;
- `combo_selection`, `random_state`, `max_combos_per_sample`;
- niveau de sortie attendu: observation, source, combo ou sample.

Si ces éléments ne sont pas dans le bundle, `predict()` peut produire une matrice différente de celle vue en CV/refit. Le cas actuel de `_sources`/`link_by` montre le risque: une intention de lien peut être parsée puis disparaître parce qu'elle n'est pas persistée comme contrat runtime. Le nouveau `rep_fusion` doit donc être rejoué en prédiction, pas skippé comme les `rep_to_sources`/`rep_to_pp` actuels.

### Workspace et stockage

Le workspace doit stocker les identités et niveaux nécessaires sans casser les lecteurs existants:

- clé sample externe stable;
- `observation_id`, `source_id`, `rep_id`;
- `combo_id` si cartésien;
- provenance/origine, idéalement alignée avec la provenance d'augmentation existante;
- `prediction_level` ou `evaluation_scope`;
- score row-level diagnostique et score sample-level sélectionnable.

La question de compatibilité doit être tranchée: ajout de colonnes optionnelles compatible 0.9.x, migration de schéma, ou bump de format. Le défaut legacy doit continuer à produire les mêmes résultats pour les datasets sans `RepetitionSpec`.

### API `RunResult`, `PredictResult`, `top()` et exports

Changer le score sélectionnable est visible utilisateur et visible webapp:

- `top()` doit pouvoir choisir `evaluation_scope="sample"` sans casser le défaut actuel;
- les exports doivent indiquer si le score est row-level, combo-level ou sample-level;
- les prédictions agrégées doivent être identifiables comme telles, pas seulement comme un fold companion;
- le refit automatique doit refuser un pipeline qui n'a pas de score dans le scope demandé.

### Explainability

`per_source_aggregate` change la provenance des features dès la Phase 2. SHAP/explain doit dire si l'explication porte sur:

- la feature agrégée par source;
- une répétition particulière;
- une moyenne/dispersion de répétitions;
- une combinaison cartésienne.

Sans ce contrat, une explication peut être correcte numériquement mais trompeuse scientifiquement.

## 5. Backlog d'implémentation dans nirs4all

### Phase 0 - Verrouiller les limites actuelles

Objectif: éviter les mauvais usages pendant qu'on implémente le support natif.

- Documenter que `rep_to_sources` et `rep_to_pp` supposent une répétition globale uniforme.
- Lever une erreur au chargement si les sources legacy ont des nombres de lignes différents, sauf chemin explicitement `RepetitionSpec`.
- Diagnostiquer les configs `sources[*].link_by`: aujourd'hui le lien est parsé mais non exécuté comme jointure source-aware et non persisté comme contrat runtime.
- Ajouter des validations explicites lorsque `repetition` est global mais que les sources ont des axes incompatibles.
- Lever une erreur si des sources ont le même nombre de lignes mais des IDs de samples divergents lorsque des IDs sont disponibles.
- Interdire ou signaler fortement `merge.unsafe=True` dans les pipelines avec répétitions.
- Transformer les warnings de fuite de groupes/alignment en erreurs dans les profils avec répétitions source-aware.
- Auto-binder `physical_sample_id` comme groupe effectif dès qu'une expansion crée plusieurs lignes par sample; durcir les warnings ne protège pas les runs où aucun groupe n'a été fourni.
- Corriger la doc qui mentionne `merge_sources: "average"` alors que le code supporte `concat/stack/dict` dans les chemins inspectés.

Modules/doc touchés:

- `docs/source/user_guide/data/aggregation.md`
- `docs/source/user_guide/pipelines/multi_source.md`
- `docs/source/concepts/branching_and_merging.md`
- `docs/source/reference/pipeline_syntax.md`
- `nirs4all/operators/data/repetition.py`
- `nirs4all/controllers/data/repetition.py`
- `nirs4all/controllers/data/merge.py`
- `nirs4all/data/features.py`
- `nirs4all/data/loaders/loader.py`

### Phase 1 - `RepetitionSpec` source-aware

Objectif: représenter et valider le problème sans encore changer tous les contrôleurs.

Ajouter une structure conceptuelle:

```text
physical_sample_id | internal_sample_id | unit_level | unit_id | source_id | observation_id | rep_id | origin_sample_id | derived_unit_id | row_id | partition | target_id | sample_influence_weight | quality_flag
```

Invariants:

- `physical_sample_id` est une clé externe stable, persistée, distincte de l'id interne auto-incrémenté;
- `sample_influence_weight` est le poids numérique effectif dérivé du `FitInfluencePolicy` du noeud de fit; le nom de la policy reste une propriété d'artifact/config, pas une colonne de provenance ligne par ligne;
- `unit_level` vaut au minimum `observation`, `combo`, `sample`; `unit_id` est la clé primaire de la ligne courante à ce niveau;
- pour une observation brute, `unit_level=observation`, `unit_id=observation_id`, `derived_unit_id` est null;
- pour une ligne cartésienne, `unit_level=combo`, `unit_id=derived_unit_id`, `origin_sample_id=physical_sample_id`, et le lineage porte les `observation_id` composants;
- `observation_id` est soit stocké, soit dérivé de manière déterministe de `(physical_sample_id, source_id, rep_id)`;
- `(physical_sample_id, source_id, rep_id)` unique après normalisation;
- cardinalités attendues par source;
- target sample-level identique sur toutes les observations d'un sample, sauf politique explicite;
- metadata typées `sample`, `source`, `repetition`;
- `source_id` stables, pas seulement `source_0`, `source_1`.
- mapping clair entre cette table et les colonnes existantes de l'indexer (`sample`, `partition`, `processing`, `augmentation`), pour ne pas créer une provenance parallèle incompatible.

Modules touchés:

- `nirs4all/data/config.py`: accepter une spec dict en plus du `str` legacy.
- `nirs4all/data/schema/config.py`: ajouter schema et validation.
- `nirs4all/data/parsers/files_parser.py`: clarifier et implémenter réellement `link_by` source/target/metadata.
- `nirs4all/data/loaders/loader.py`: construire la table normalisée.
- `nirs4all/data/dataset.py`: `set_repetition_spec()`, `repetition_spec`, `source_repetition_counts()`, `normalized_repetition_table()`.
- `nirs4all/data/_indexer/sample_manager.py` et `nirs4all/data/indexer.py`: porter une clé sample externe stable et son mapping vers les IDs internes.
- `nirs4all/data/features.py`: au minimum stocker et vérifier les `source_id`; long terme, le support de tables source-specific relève de la Phase 7 ragged, pas du premier livrable.

### Phase 1.5 - `RawMultiSourceDataset` / `NormalizedObservationTable`

Objectif: créer le modèle de staging que les représentations consomment. Sans cette phase, `rep_fusion` n'a pas d'entrée bien définie pour `MIR=2N`, `RAMAN=3N`, `NIRS=2N`.

Contenu:

- `X_by_source: dict[source_id, ndarray]` ou wrapper équivalent;
- relation table avec le schéma Phase 1 complet: `physical_sample_id`, `internal_sample_id`, `unit_level`, `unit_id`, `source_id`, `observation_id`, `rep_id`, `origin_sample_id`, `derived_unit_id`, `row_id`, `partition`, `target_id`, `sample_influence_weight`, `quality_flag`;
- targets sample-level séparées et validées contre les duplications contradictoires;
- metadata typées par niveau: sample, source, observation, derived unit;
- fingerprint de relation table et ordre déterministe des observations;
- API de matérialisation vers `SpectroDataset/Features` uniquement via `RepresentationPlan`.

Règle: aucun contrôleur aval ne doit inventer une jointure positionnelle depuis des tables source-specific. Il consomme soit le staging relationnel, soit un `SpectroDataset/Features` déjà aligné.

### Phase 2 - `rep_fusion(mode="per_source_aggregate")`

Objectif: livrer vite un chemin scientifiquement sûr.

Point de séquencement: cette phase peut sortir avant la refonte complète `evaluation_scope`, car la sortie contient une seule ligne par sample physique. Dans ce mode, les métriques row-level legacy deviennent effectivement sample-level.

Comportement:

- entrée: sources avec répétitions source-specific identifiées par `RepetitionSpec`;
- sortie: dataset aligné, une ligne par `physical_sample_id`, sources conservées;
- méthode par source: `mean`, `median`, `trimmed_mean`, plus tard outlier/QC;
- provenance: `n_reps_by_source`, `rep_ids_used`, flags outlier/drop;
- prediction mode: accepter des cardinalités différentes en prédiction si la méthode est robuste (`mean`/`median`), lever une erreur pour les méthodes nécessitant `expected` strict.
- `supports_prediction_mode=True` obligatoire, avec replay de l'agrégation dans le bundle `.n4a`.
- ordre pipeline: si `rep_fusion` change le nombre de lignes, il doit soit s'exécuter avant la génération des folds, soit fournir un remapping folds par clé sample stable. Les preprocessings qui apprennent des paramètres cross-sample doivent être fit train-only après split, pas sur toutes les observations avant split.
- composition: refuser par défaut `rep_fusion` si le pipeline a déjà utilisé `rep_to_sources`, `rep_to_pp` ou une `repetition` legacy qui modifie le même axe.

Modules touchés:

- `nirs4all/operators/data/repetition.py`: nouveau config `RepFusionConfig`.
- `nirs4all/controllers/data/repetition.py`: nouveau contrôleur `RepFusionController`.
- `nirs4all/data/dataset.py`: méthode de reconstruction vers sources alignées.
- `nirs4all/data/predictions.py`: conserver contexte d'agrégation.
- `nirs4all/sklearn` / `NIRSPipeline` / explainability: exposer la provenance des features agrégées.
- tests unitaires `tests/unit/controllers/data/test_repetition.py`.

### Phase 3 - Évaluation, ranking et refit sample-level

Objectif: s'assurer que le modèle refitté correspond au bon critère.

À faire:

- introduire `evaluation_scope: row | observation | combo | sample`;
- stocker le score agrégé sample-level comme score sélectionnable, pas seulement comme companion report;
- faire dépendre `extract_top_configs()` du scope choisi;
- faire dépendre `Predictions.top()` / `get_best()` du `ReductionPlan` demandé pour que les exports et le refit lisent le même gagnant;
- étendre le registre de métriques pour supporter le flux "agréger puis scorer" et, si nécessaire, les poids;
- reporter clairement les scores row-level comme diagnostics seulement;
- pour refit, propager le contexte `evaluation_scope`, `aggregation_method`, `origin_sample_id`;
- refuser le refit automatique si le pipeline sélectionné n'a pas de score dans le scope demandé.
- garder les défauts API actuels pour les pipelines sans `RepetitionSpec`, et ajouter `evaluation_scope` de manière additive dans `top()`, exports et résultats.
- ne plus dépendre directement de `avg`, `w_avg`, `final` et `_agg` dans `fold_id` pour le stacking/refit multi-slots; utiliser les nouveaux champs dédiés et l'accessor de compatibilité. La migration qui nettoie physiquement `fold_id` est une étape majeure séparée.

Modules touchés:

- `nirs4all/data/predictions.py`
- `nirs4all/core/metrics` ou équivalent scoring central: agrégation avant métrique et pondération.
- `nirs4all/pipeline/execution/refit/config_extractor.py`
- `nirs4all/pipeline/execution/refit/model_selector.py`
- `nirs4all/pipeline/execution/refit/executor.py`
- reporting workspace, `RunResult`, `PredictResult`, `top()` et exports.
- stockage predictions/refit pour `prediction_scope`, `reduction_role`, `reduction_id`.

### Phase 4 - `rep_fusion(mode="cartesian")`

Objectif: permettre `2 x 3 x 2 = 12` de manière contrôlée, en distinguant cartésien complet, Monte-Carlo et cartésien train-only via augmentation.

Comportement obligatoire:

- créer `combo_id`;
- conserver `origin_sample_id`;
- générer une ligne alignée par combo;
- `FitInfluencePolicy` explicite; `1 / n_combos(physical_sample_id)` est le défaut des politiques qui égalisent l'influence sample au fit;
- limiter `max_combos_per_sample`;
- définir `combo_selection` (`random_seeded` par défaut scientifique, `deterministic_first` pour debug/repro stricte, `stratified` si pertinent) et persister la graine/politique dans le manifest;
- fournir un garde-fou global de nombre de lignes ou mémoire, pas seulement un plafond par sample;
- split groupé par `origin_sample_id`;
- agrégation de prédictions `combo -> sample`;
- score final sample-level.
- refuser l'entraînement cartésien seulement si la configuration exige une influence égale des samples au fit et que le modèle choisi ne supporte ni poids ni politique de repli déclarée.

Sous-modes:

- `cartesian_full`: train, OOF, refit et prédiction sur combos; agrégation `combo -> sample` pour scorer et servir.
- `cartesian_mc`: idem mais avec `K` combos seedés par sample.
- `cartesian_augmentation`: ligne de base agrégée + combos enregistrés comme augmentations `origin=sample`; validation/test/predict sur la base, combos seulement en train. C'est le premier cartésien à implémenter si l'objectif est de déployer vite sans toucher tout le scoring.

Refus par défaut:

- scoring final combo-level sans opt-in;
- cartésien sans `group_by=sample_id`;
- source manquante sans stratégie explicite;
- combinaison avec `merge.unsafe=True`.

Modules touchés:

- `RepFusionController`
- `controllers/splitters/split.py`
- `controllers/models/base_model.py` et contrôleurs modèles spécifiques pour `FitInfluencePolicy`;
- backends sklearn/keras/torch/jax pour stratégie de poids, resampling, uniform rows ou refus explicite;
- orchestrateur/parallélisme pour garde-fous mémoire/temps;
- `data/predictions.py` pour `combo_id` et agrégation pondérée;
- refit/ranking.

### Phase 5 - Fusion tardive et stacking sample-level

Objectif: supporter proprement `by_source -> modèle par source -> agrégation OOF -> stacker`.

À faire:

- permettre à une branche source de travailler au niveau observation;
- agréger les prédictions OOF observation -> sample avant le merge;
- durcir le merge de prédictions déjà partiellement sample-indexed pour utiliser une clé sample physique stable, pas seulement l'id interne du dataset;
- valider strictement les folds et la couverture des `sample_id`;
- remplacer les imputations silencieuses par `coverage: strict | drop_incomplete | impute_declared`;
- interdire les prédictions in-sample comme méta-features de validation;
- refuser ou réduire explicitement les prédictions multi-combo avant un assembleur OOF sample-level; aucun comportement last-write-wins ne doit être possible.
- revoir le refit stacking pour entraîner le méta-modèle sur OOF sample-level, puis refitter les bases full-train pour l'inférence.
- ajouter `StackingFitContract` dans les artifacts: `meta_training_features`, `inference_features`, `selection_protocol`, `base_prediction_calibration`, fingerprints des bases refittées et schéma des méta-features;
- documenter la sélection branch-level, globale et meta-aware; pour un stacking fiable, prévoir une CV imbriquée ou un protocole explicite de sélection OOF pour éviter la fuite de sélection.

Modules touchés:

- `nirs4all/controllers/data/branch.py`
- `nirs4all/controllers/data/merge.py`
- `nirs4all/controllers/models/meta_model.py`
- `nirs4all/controllers/models/stacking/reconstructor.py`
- `nirs4all/pipeline/execution/refit/stacking_refit.py`

### Phase 6 - Export / convergence DAG-ML

Objectif: aligner les invariants nirs4all avec DAG-ML au lieu de réinventer une sémantique parallèle.

À produire côté nirs4all:

- `SampleRelationSet` avec `sample_id`, `observation_id`, `source_id`, `target_id`, `group_id`, `origin_sample_id`;
- `FoldSet` au niveau `sample_id`;
- `DataBinding(require_relations=true)`;
- DSL `branch mode=by_source` + `merge_model`;
- `AggregationPolicy` observation -> sample, idéalement en réutilisant `aggregate_observation_predictions()`;
- contrats d'arêtes `requires_oof=true`, `requires_fold_alignment=true`.

À compléter côté DAG-ML:

- contrats de port/arête avec `unit_level` et `alignment_key`; `require_relations` existe déjà sur `DataBinding` et doit être utilisé plutôt que réimplémenté;
- politique de missing source;
- `BranchViewPlan` natif dans les plans;
- helpers Python pour relations/bindings/enveloppes;
- validation des joins feature/source par domaine d'alignement.
- cohérence avec `dag-ml-data`: la fusion de features y accepte une source de référence répétée avec broadcast singleton, mais refuse des répétitions ambiguës sur les sources non référence. Cela appuie un design nirs4all qui choisit explicitement agrégation, ports séparés ou modèle ragged.

### Phase 7 - Modèle ragged natif

Objectif long terme: ne plus forcer un dataset multisource hétérogène dans une matrice rectangulaire trop tôt.

Chantiers:

- nouvelle abstraction de données multi-table par source;
- indexer capable de projeter sample -> observations par source;
- opérateurs preprocessing par source sur tables ragged;
- modèles multi-instance ou adaptateurs sklearn;
- stockage de prédictions multi-niveaux;
- visualisation et explainability compatibles avec source/repetition/sample.

### Tests à ajouter

Unitaires:

- parsing `RepetitionSpec`;
- doublon `(sample_id, source_id, rep_id)`;
- mapping clé sample externe -> ID interne et persistance/rechargement;
- mapping `observation_id` -> indexer;
- `NormalizedObservationTable` avec `X_by_source` et metadata typées;
- `link_by` shuffled avec mêmes longueurs mais ordres différents;
- source absente;
- target contradictoire;
- target reducer explicite si une cible varie par observation;
- metadata sample-level contradictoire;
- ordre de répétitions non trié;
- `per_source_aggregate`;
- ordre preprocessing/réduction: `raw_before_preprocessing` vs `after_observation_preprocessing`;
- reducer fold-fitted entraîné seulement sur train fold, replayé sur validation;
- `cartesian`;
- `cartesian_augmentation` avec combos marqués par `origin`;
- `cartesian_mc` avec sélection seedée reproductible;
- `cartesian_full` refusé si produit > cap global sauf mode MC ou cap explicite;
- `FitInfluencePolicy=uniform_rows` accepté quand cardinalités constantes;
- backend sans `sample_weight` accepté en `uniform_rows`, refusé si `equal_sample_influence` est strict sans fallback;
- sélection/troncature déterministe des combos;
- refus de composition `rep_fusion` avec `rep_to_sources`/`rep_to_pp` incompatibles;
- validation `max_combos_per_sample`.

Intégration:

- chargement legacy `MIR=2N`, `RAMAN=3N`, `NIRS=2N` refuse au lieu de créer un dataset incohérent;
- `link_by` réellement exécuté ou erreur explicite si non supporté;
- fixture `MIR=2`, `RAMAN=3`, `NIRS=2`;
- `per_source_aggregate -> branch by_source -> merge_sources -> model -> refit`;
- `cartesian -> grouped CV -> aggregate combo->sample -> refit`;
- `cartesian_augmentation -> grouped CV -> validation sur base uniquement -> refit`;
- `by_source -> modèles source -> OOF sample-level -> stacking`;
- stacking avec sélection branch-level versus globale, et garde anti méta-features in-sample;
- fit influence policy propagée, ou refus explicite si la politique demandée est incompatible avec le backend;
- `top(evaluation_scope="sample")` et export sans casser le défaut legacy;
- `.n4a` export/predict rejoue la même agrégation;
- explainability sur features agrégées avec provenance;
- source order permuté entre fichiers;
- source manquante avec `error`, `drop_incomplete`, `impute_declared`;
- missing source avec imputer entraîné uniquement sur train;
- prédiction avec cardinalités différentes train/test;
- classification: proba moyenne, vote, calibration, seuil figé hors test;
- ranking/refit sur score sample-level;
- nested-vs-reuse OOF selection pour stacking;
- DAG-ML: join invalide entre domaines d'unité incompatibles refusé;
- non-régression des tests existants `test_repetition.py`, `test_aggregation_integration.py`, `test_multisource_branching_stacking.py`.

## Décisions issues des annotations utilisateur

Cette section remplace les questions ouvertes initiales. Elle fixe le cadrage d'implémentation nirs4all / DAG-ML et retire les considérations propres à un challenge ou à un dataset donné.

- **Cible de production.** La sortie principale est une prédiction par échantillon physique (`physical_sample_id`). Les prédictions par répétition/source/combo peuvent exister comme diagnostics, méta-features ou artifacts intermédiaires, mais le split, le score, le ranking, le refit et l'export principal restent sample-level.
- **But des répétitions.** Les répétitions existent pour exploiter toute l'information spectrale disponible pour un même échantillon, pas pour créer de nouveaux échantillons indépendants.
- **Ordre des répétitions.** Les répétitions sont échangeables par défaut. Un mode `stack_fixed` avec ordre stable peut exister, mais il doit déclarer explicitement que l'ordre est sémantique et que les cardinalités train/predict doivent être compatibles.
- **Cibles.** Le périmètre de cette feature suppose des cibles sample-level. Une cible par répétition/source devient un autre problème et doit être refusée ou passer par une extension future explicite.
- **Interactions feature-level.** Les interactions brutes entre sources restent dans le périmètre moteur parce qu'elles ont un impact sur les représentations à implémenter (`cartesian_full`, `cartesian_mc`, `stack_fixed`, et plus tard ragged/multi-instance). La roadmap ne doit pas décider scientifiquement à la place de l'utilisateur; elle doit rendre ces représentations configurables et leak-safe.
- **Premier livrable.** `per_source_aggregate` et late fusion sont tous les deux nécessaires. La roadmap peut les séquencer par dépendances techniques, mais elle ne doit pas supprimer l'un au profit de l'autre. Le cartésien complet reste aussi une représentation de première classe.
- **Source absente en prédiction.** La politique par défaut ne doit pas être un crash systématique: warning + fallback déclaré (`impute_declared`, padding/mask, modèle partiel, ou `strict` si demandé). Le choix exact est une policy de représentation/prediction et doit être visible dans les artifacts.
- **nirs4all / DAG-ML.** Les deux projets doivent avoir leur propre implémentation. DAG-ML viendra plus tard et doit être conçu comme futur moteur propre, mais nirs4all ne doit pas attendre DAG-ML pour livrer la feature.
- **Workspace et `.n4a`.** Le projet n'est pas en v1 publique stable. On garde la compatibilité et le legacy code utiles, mais on ne crée pas une longue phase de dépréciation artificielle. Les nettoyages peuvent être planifiés progressivement. Toute signature publique ou format d'export visible qui change doit toutefois être listé dans la roadmap comme décision utilisateur explicite.
- **Classification.** L'agrégation des probabilités/classes doit être configurable. Les défauts nirs4all existants à préserver sont la moyenne de probabilités et/ou le vote selon le contexte.
- **Provenance.** La provenance doit toujours être conservée. `origin_sample_id` doit réutiliser l'esprit de la provenance/augmentation existante nirs4all quand c'est cohérent, et tout combo ou ligne dérivée doit garder son lineage.

### Explication simple: `FitInfluencePolicy`

La question n'est pas "est-ce que PLS accepte des poids?". La vraie question est: si un sample produit plus de lignes qu'un autre, doit-il compter plus dans l'entraînement?

Exemple: `S1` a 12 combos, `S2` a 6 combos parce qu'une répétition manque.

- `uniform_rows`: chaque ligne compte pareil. Donc `S1` pèse deux fois plus que `S2`. C'est acceptable si toutes les cardinalités sont identiques, ou si l'utilisateur veut vraiment cette influence.
- `equal_sample_influence`: chaque sample physique compte pareil. Les lignes de `S1` reçoivent donc un poids plus petit que celles de `S2`.
- `resample_equalized`: si le modèle ne sait pas recevoir de poids, on peut tirer/sous-échantillonner des lignes pour approximer une influence égale.
- `scorer_only`: on ne corrige pas la loss d'entraînement, mais on garantit que le score/ranking/refit se fait au sample-level.
- `strict`: si la politique demandée ne peut pas être respectée par le backend, le run échoue.

Recommandation d'implémentation: ajouter une valeur `auto` en configuration expérimentale. `auto` choisit `uniform_rows` quand toutes les cardinalités dérivées sont égales; choisit `equal_sample_influence` quand elles varient et que le backend supporte un poids; sinon choisit `resample_equalized` avec warning, ou échoue si `strict=true`.

### Explication simple: refit final du stacking

Un stacker apprend à combiner des prédictions de modèles de base. Pour ne pas tricher, il doit apprendre sur des prédictions OOF: chaque ligne vue par le meta-modèle vient d'un modèle de base qui n'a pas entraîné sur ce sample.

Le flux recommandé est:

1. CV des modèles de base, avec prédictions OOF alignées par `physical_sample_id`.
2. Entraînement/sélection du méta-modèle sur cette table OOF.
3. Refit des modèles de base sur tout le train.
4. Prédiction des données à servir par les bases refittées.
5. Application du méta-modèle choisi aux prédictions des bases refittées.

Le piège à éviter est d'entraîner le méta-modèle final sur des prédictions in-sample produites par des bases refittées sur tout le train: le meta verrait des prédictions trop faciles et surestimerait la performance. Ce mode peut rester un opt-in expérimental, mais ne doit pas être le défaut.

Autre piège: les prédictions OOF des bases et les prédictions des bases refittées peuvent ne pas être exactement sur la même échelle. `base_prediction_calibration=rank` ou une calibration OOF->refit permet de stabiliser les méta-features.

Décision roadmap: `StackingFitContract` est obligatoire pour les nouveaux plans hétérogènes. `reuse_oof` reste autorisé pour exploration rapide, mais les profils de validation finale doivent utiliser `nested` ou `holdout`.



## Position finale

Le support natif des répétitions multisources hétérogènes ne doit pas être ajouté en modifiant légèrement `rep_to_sources`. Ce serait fragile et probablement trompeur. Il faut introduire une sémantique explicite de répétitions par source.

Pour déployer rapidement sans réduire le périmètre fonctionnel, la meilleure trajectoire est:

1. validation/documentation et erreurs défensives sur les limites actuelles, en particulier `link_by`, longueurs inter-sources et groupement sample auto-activé;
2. clé sample physique stable + `RepetitionSpec`;
3. `rep_fusion(per_source_aggregate)` comme premier jalon de représentation alignée, car il ne dépend pas encore d'une refonte complète du scoring;
4. contrat stable workspace / `.n4a` / replay predict;
5. score/ranking/refit sample-level;
6. fusion tardive OOF sample-level dans le même lot fonctionnel que la pré-agrégation, comme défaut ML général quand aucune interaction brute inter-source n'est exigée;
7. cartésien train-only via augmentation, puis cartésien complet/Monte-Carlo avec influence déclarée et bornes de coût;
8. convergence DAG-ML / modèle ragged.

Cette trajectoire garde nirs4all utilisable rapidement tout en évitant le piège principal: faire croire que 12 combinaisons cartésiennes sont 12 échantillons indépendants. Elle évite aussi l'autre piège, plus subtil: écarter le cartésien alors qu'il peut être l'estimateur le moins biaisé pour `E[f(x)]` et le seul moyen d'apprendre des interactions feature-level brutes. La position finale est donc: cartésien oui, mais comme mode explicite, groupé, scoré sample-level, borné, avec une politique d'influence déclarée; pas comme multiplication naïve de lignes.

Après revue fable et Codex indépendant, les prérequis non négociables avant implémentation sont: identité sample stable, contrats persistés, ordre pipeline anti-fuite, replay en prédiction, auto-grouping par sample physique, et passage des warnings critiques à des erreurs dans ce profil.
