# Plan de Reprise 2026-05 - `bench/` NIRS Modelling Workspace

**Date** : 2026-05-05  
**Version** : v2.0, revue et rebasee pour execution multi-agents  
**Périmètre actif** : `bench/AOM`, `bench/AOM_v0`, `bench/nicon_v2`, `bench/fck_pls`, `bench/tabpfn_paper`, `bench/Subset_analysis`, `bench/scenarios` à créer, `bench/harness` à créer.  
**Périmètre explicitement hors travail courant** : `bench/nirs_synthetic_pfn`, `bench/synthetic`, `bench/ViTnirs`, `bench/AOM_v0/Blup`, webapp, root `Roadmap.md`.

---

## 0. Résumé exécutif

Le plan précédent posait la bonne direction, mais il restait trop large pour lancer des agents sans collision. Cette v2 applique les réponses aux Open Questions, sort la synthèse/PFN/ViT du scope, clarifie les dépendances, et transforme le travail en briefs d'agents autonomes.

Objectif opérationnel : produire quatre presets benchmarkables (`fast_reliable`, `strong_practical`, `best_current`, `exhaustive_research`) à partir d'un master CSV verrouillé, avec registry runnable, harness reproductible, validation sur `fast12_transfer_core` -> `audit20_transfer_core` -> full-57, et rapports assez propres pour alimenter P1/P2/P3.

Stratégie retenue :

| Voie | Role | Livraison principale | Horizon |
|---|---|---|---|
| **A** | Classical Production Pack | AOM-PLS unifié, AOM-Ridge audité, multiview ASL complété, candidats classiques pour presets | 2-6 mois |
| **B** | Residual/FCK Pack | V2L-Residual-AOMPLS multiseed + investigation FCK complète, avec gates do-no-harm | 2-6 mois |
| **C** | MLOps Spine | master freeze, registry, exporter scénarios, harness, dashboard, sync multi-agents | 1-3 mois |

Règle de départ : **A1/P0 est bloquant**. On lance d'abord le gel/reconstruction du master CSV. Après ce statut `P0_DONE`, les agents A, B et C peuvent travailler en parallèle avec contextes séparés.

---

## 1. Review Appliquée - Changements Par Rapport à v1.1

1. Les Open Questions ne sont plus ouvertes. Leurs réponses sont intégrées comme décisions de scope et de séquencement.
2. `nirs_synthetic_pfn`, `synthetic`, `ViTnirs` et P4 sont sortis du travail courant. Ils restent des archives/pre-gates futurs, pas des tâches d'agents maintenant.
3. La voie C n'est plus "MLOps + long-horizon". Elle devient strictement la colonne vertébrale MLOps et scenario product.
4. Le calendrier 4 semaines a été relâché. B1 multiseed, FCK residual et les runs full-57 ne tiennent pas dans un sprint court sans sacrifier la qualité.
5. `r20_curated_oof` est considéré comme déjà présent dans le master courant ; l'action est audit/tag, pas ingestion aveugle.
6. Les subsets sont corrigés : `fast12_transfer_core` et `audit20_transfer_core` sont acceptés comme gates de triage, avec caveat sur les scopes AOM-PLS-only et multi-kernel-only où Spearman tombe à 0.922 et 0.900.
7. Le choix des ratios est stabilisé : pour l'ingénierie et les presets, `score_ratio_vs_source_run_pls` est la vue primaire ; pour les claims papier et le strict leaderboard, `score_ratio_vs_dataset_pls` est toujours rapporté à côté.
8. TabPFN reste dans les gros presets malgré son coût, avec budget et rationale explicites sur la sélection de preprocessing.
9. FCK n'est pas archivé. Il reçoit une investigation dédiée, seul et combiné à AOM, avant décision.
10. RandomForest avec feuilles composées de modèles rapides est ajouté en tâche C, comme diagnostic/selector expérimental.

---

## 2. État Local Vérifié

Vérification locale du 2026-05-05 sur `bench/benchmark_master_results.csv` :

| Élément | Valeur courante |
|---|---:|
| Lignes totales | 21 769 |
| `observed` | 20 570 |
| `reference_paper` | 335 |
| `oracle_by_model_class` | 719 |
| `oracle_global_dataset` | 86 |
| `source_oracle` | 59 |
| `source_family` distinctes | 8 |
| Colonne `protocol_maturity` | absente |

Important : `bench/benchmark_synthesis.md` annonce encore 20 964 lignes source et 83 paires dataset/task eligibles, tandis que le master courant contient 21 769 lignes. Ce n'est pas une erreur à corriger à la main : **P0 doit reconstruire le master et documenter la définition exacte des datasets eligibles** avant toute décision de ranking.

### 2.1 Oracle par classe, strict vs global PLS

Calculé depuis les lignes `oracle_by_model_class` du master courant :

| Classe | Datasets | Médiane | Wins |
|---|---:|---:|---:|
| TabPFN | 59 | 0.908 | 45/59 |
| AOM-PLS | 59 | 0.929 | 49/59 |
| AOM-Ridge | 53 | 0.942 | 45/53 |
| Ridge | 55 | 0.970 | 42/55 |
| Meta-selector/MoE | 59 | 0.972 | 39/59 |
| Multi-kernel ridge | 53 | 0.983 | 34/53 |
| PLS | 67 | 1.000 | 0/67 |
| Hybrid CNN+linear | 51 | 1.002 | 25/51 |
| FCK-PLS | 8 | 1.005 | 4/8 |
| Hybrid CNN+AOM | 42 | 1.007 | 20/42 |
| NICON/CNN | 56 | 1.018 | 26/56 |
| CatBoost | 57 | 1.038 | 23/57 |
| POP-PLS | 57 | 1.457 | 9/57 |

Interprétation : TabPFN reste la meilleure famille observée, AOM est la meilleure direction classique/spectroscopique, les ensembles/MoE montrent une complémentarité réelle mais optimiste tant que le nesting n'est pas prouvé, et les CNN purs restent hors champion.

### 2.2 Subsets de triage

Source : `bench/Subset_analysis/RETHOUGHT_SUBSETS.md`.

Ordre obligatoire :

1. `fast12_transfer_core` : smoke gate rapide.
2. `audit20_transfer_core` : audit avant full.
3. full-57 : validation de production et claims papier.

Caveat : ces subsets sont excellents pour all-candidates, no-TabPFN, linear-core, AOM-Ridge, TabPFN et nonlinear-core. Ils sont moins forts pour AOM-PLS-only et multi-kernel-only ; ne pas conclure sur ces deux scopes sans full-57.

---

## 3. Décisions de Scope Verrouillées

### 3.1 In scope

- P1 et P2 séparés : P1 = AOM-PLS unification ; P2 = Multi-view Adaptive Super Learner.
- Cohorte primaire = full-57, mais toute itération passe d'abord par `fast12_transfer_core` puis `audit20_transfer_core`.
- A/B/C en parallèle seulement après P0/A1.
- AOMRidge headline-spxy3 : nested audit requis avant promotion.
- FCK : test intensif dédié, seul et combiné à AOM, puis GO/NO-GO documenté.
- Multiview : package séparé court terme, objectif de merge avec AOM plus tard.
- TabPFN : inclus dans `strong_practical`, `best_current`, `exhaustive_research` avec budget preprocessing explicite.
- RF avec feuilles de modèles rapides : prévu en diagnostic C.
- Compute : RTX 4090 locale. Demander explicitement avant de planifier des jobs sur GPUs LAN via SSH.

### 3.2 Out of scope

- Synthèse/PFN/ViT : trop préliminaire pour ce cycle.
- BLUP/REML : laissé de côté.
- Pure CNN architecture search : arrêté, sauf rôle residual/feature extractor avec gate.
- Webapp : oui à terme, mais hors contexte actuel librairie/bench.
- Root `Roadmap.md` : roadmap librairie séparée.

---

## 4. Protocole Multi-Agents

### 4.1 Séquence

1. **Agent 0 / C-bootstrap** exécute P0 : master freeze + reconstruction + schema maturity.
2. Quand P0 publie `P0_DONE`, lancer les trois agents autonomes :
   - **Agent A** : classiques AOM/AOM-Ridge/multiview.
   - **Agent B** : residual NN + FCK.
   - **Agent C** : registry/exporter/harness/dashboard et coordination.

### 4.2 Ownership strict

| Zone | Owner | Autres agents |
|---|---|---|
| `bench/benchmark_master_results.csv` | C/P0 | lecture seule après P0, sauf accord explicite |
| `bench/build_benchmark_synthesis.py` | C | lecture, propositions via `bench/SYNC.md` |
| `bench/scenarios/` | C | A/B proposent des entrées registry |
| `bench/harness/` | C | A/B consomment |
| `bench/AOM/`, `bench/AOM_v0/aompls/`, `bench/AOM_v0/Ridge/`, `bench/AOM_v0/multiview/` | A | B/C lecture seule |
| `bench/nicon_v2/`, `bench/fck_pls/` | B | A/C lecture seule |
| `bench/Subset_analysis/` | C | lecture seule |
| `bench/SYNC.md` | append-only tous agents | jamais réécrire l'historique |

Si un agent a besoin de modifier une zone non possédée, il ajoute une proposition dans `bench/SYNC.md` avec patch attendu, raison et blocage.

### 4.3 Status et handoff

Chaque agent termine chaque session par :

- un statut dans son fichier `STATUS.md` ou `docs/STATUS.md` ;
- une entrée `bench/SYNC.md` si un autre agent doit consommer quelque chose ;
- une liste des artefacts produits, avec chemin et maturité.

Format minimal pour `bench/SYNC.md` :

```markdown
## 2026-05-05 HH:MM - Agent A - A5 nested audit

Status: BLOCKED|READY|DONE
Produced:
- path/to/artifact.csv
Needs:
- Agent C: add registry entry for ...
Risk:
- ...
Next:
- ...
```

### 4.4 Règles compute

- Aucun job GPU long sans entrée préalable dans `bench/SYNC.md`.
- La 4090 locale est prioritaire pour B pendant les runs NN.
- Les jobs CPU A/C peuvent tourner hors créneaux GPU.
- Tout usage GPU LAN via SSH doit être demandé à l'utilisateur avant planification.

---

## 5. Phase P0 - Master Freeze et Contrat Commun

**Owner recommandé** : Agent C-bootstrap.  
**Bloque** : agents A/B/C parallèles.

### Tâches

1. Snapshot `bench/benchmark_master_results.csv` et SHA256 dans `bench/MASTER_CSV_HASH.txt`.
2. Exécuter `python bench/build_benchmark_synthesis.py` et documenter le diff exact.
3. Réconcilier les compteurs 21 769 / 20 964 / 83 paires dataset-task, avec définition écrite des filtres.
4. Ajouter `protocol_maturity` au master via le builder, pas par édition manuelle.
5. Définir les valeurs autorisées : `locked`, `exploratory`, `legacy`, `oracle`, `local_not_master`.
6. Auditer/tagger :
   - `r20_curated_oof` : déjà présent, vérifier OOF cleanness puis tag.
   - `AdaptiveSuperLearner` Phase-11 : partiel 35/61, tag `exploratory`.
   - `AOMRidge-Blender-headline-spxy3` : tag provisoire tant que nested audit non fait.
7. Produire un rapport `bench/MASTER_CSV_FREEZE.md`.

### Acceptance

- `bench/MASTER_CSV_HASH.txt` existe avec date, commande et hash.
- `protocol_maturity` est présent et non vide pour toutes les lignes.
- `build_benchmark_synthesis.py` reconstruit le master ou explique tout delta restant.
- `bench/MASTER_CSV_FREEZE.md` conclut par `P0_DONE` ou liste les bloqueurs.

---

## 6. Agent A - Classical Production Pack

### Mission

Stabiliser les candidats classiques qui alimentent `fast_reliable`, `strong_practical` et `best_current` : AOM-PLS, AOM-Ridge, multiview/ASL, sentinelles PLS/Ridge/MKR.

### Sources à lire

- `bench/AOM/ROADMAP.md`
- `bench/AOM_v0/Summary.md`
- `bench/AOM_v0/README.md`
- `bench/AOM_v0/Ridge/`
- `bench/AOM_v0/multiview/docs/SUMMARY.md`
- `bench/Subset_analysis/RETHOUGHT_SUBSETS.md`
- `bench/MASTER_CSV_FREEZE.md` après P0

### Tâches prioritaires

**A1 - AOM-PLS unification roadmap.** Continuer `bench/AOM/ROADMAP.md` M0-M8 comme source de vérité P1. Ne pas dupliquer le harness si C le promeut au niveau `bench/harness`.

**A2 - AOM-Ridge nested audit.** Tracer `AOMRidge-Blender-headline-spxy3` et `AOMRidge-AutoSelect-headline-spxy3`. Si sélection globale non nested, re-run OOF strict avant toute promotion.

**A3 - Elite partial runs.** Compléter les runs elites avant d'en lancer de nouveaux :

| Run | Statut | Action |
|---|---|---|
| AdaptiveSuperLearner Phase-11 full-57 | 35/61, kill 1h30 | relancer avec timeout > 4h, logs wall-clock, exclusions documentées |
| AOM-Ridge full-57 | OOM LMA | diagnostiquer mémoire, relancer ou produire variante downsampled `exploratory` |

Les runs partiels non elites restent dans le master mais exclus des scénarios.

**A4 - Candidate cards.** Pour chaque candidat classique, fournir à C une fiche registry : canonical name, aliases, module, config, runtime tier, nesting, contraintes n/p, maturité.

**A5 - Preset runs.** Dès que C livre le harness, exécuter les candidats A sur `fast12_transfer_core`, puis `audit20_transfer_core`, puis full-57.

### Acceptance

- AOMRidge headline-spxy3 a un verdict nested clair.
- ASL full-57 est complet ou exclu avec raison documentée.
- Les candidats A entrant dans `strong_practical`/`best_current` ont OOF/nesting prouvé.
- Résultats A exportés au schema C, avec `score_ratio_vs_source_run_pls` et `score_ratio_vs_dataset_pls`.

### Papier

- **P1** : AOM-PLS unification, priorité.
- **P2** : Multi-view Adaptive Super Learner, séparé de P1 mais dépendant du nested validation complet.

---

## 7. Agent B - Residual NN et FCK

### Mission

Évaluer proprement les seuls axes non-classiques encore crédibles : V2L residualisé sur teacher chemométrique, et FCK comme opérateur statique/residual compatible AOM.

### Sources à lire

- `bench/nicon_v2/docs/STATUS.md`
- `bench/nicon_v2/benchmark_runs/r20_curated_oof/results.csv`
- `bench/fck_pls/README.md`
- `bench/fck_pls/fckpls_torch.py`
- `bench/model_exploration_review.md`
- `bench/MASTER_CSV_FREEZE.md` après P0

### Tâches prioritaires

**B1 - Audit r20.** Confirmer que `r20_curated_oof` est OOF clean, single seed, 39 datasets. Ne pas le réingérer si P0 confirme sa présence.

**B2 - r21 multiseed.** Lancer `r21_curated_oof_multiseed` seulement après P0 :

- 39 datasets x 5 seeds x `V2L-Residual-AOMPLS`.
- teacher = `ASLS-AOM-compact-cv5` ou `AOMRidge-AutoSelect`.
- shrinkage CV avec `0` autorisé.
- fallback teacher si gain residual OOF non significatif.
- diagnostics catastrophic-loss obligatoires.

**B3 - FCK static.** Implémenter/tester `FCKStaticTransformer` avec petit bank :

- `alpha in {0.5, 1.0, 1.5, 2.0}` ;
- scales `{1, 2}` ;
- kernel sizes `{15, 31}` ;
- filtres normalisés, fit train-only.

Configurations smoke :

- `FCKStatic + PLS`
- `FCKStatic + Ridge`
- `FCKStatic + AOMPLS`
- `ASLS + FCKStatic + PLS`
- `concat_transform [SNV, FCKStatic] + AOMPLS`

**B4 - FCK extended + residual.** Si B3 smoke n'est pas catastrophique, promouvoir vers `audit20_transfer_core`, puis full-57. Tester `FCKResidualRegressor` avec teacher AOM, residual OOF, shrinkage CV et fallback zéro.

### Stop gates

| Gate | Condition |
|---|---|
| Residual production | V2L residual bat `aom_ridge_curated_best` d'au moins 2 % median, p<0.05, >=50 % wins sur curated39 multiseed |
| Residual science | V2L residual bat paper NICON d'au moins 5 % sur >=75 % cohorte |
| FCK GO | FCK static ou residual améliore AOM/AOM-Ridge sans q90/worst-case toxique sur audit20 puis full-57 |

Si le gate residual production échoue, P3 devient memo negative-result, pas submission. Si FCK échoue, produire `bench/fck_pls/docs/FCK_EVALUATION.md` et ne garder FCK que dans `exhaustive_research` ou archive.

### Acceptance

- B ne promeut aucun NN/FCK sans fallback do-no-harm.
- Les résultats B incluent med, q75, q90, worst-case, wins, p-values, runtime.
- Toute ligne candidate a `protocol_maturity` cohérent et registry card proposée à C.

---

## 8. Agent C - MLOps Spine et Scenario Product

### Mission

Rendre le benchmark reproductible et consommable : master verrouillé, registry, exporter, harness, dashboard, protocoles de validation.

### Sources à lire

- `bench/build_benchmark_synthesis.py`
- `bench/benchmark_master_results.csv`
- `bench/benchmark_synthesis.md`
- `bench/Backlog`
- `bench/Subset_analysis/RETHOUGHT_SUBSETS.md`
- `bench/AOM/ROADMAP.md` M0 harness/stats

### Tâches prioritaires

**C1 - Registry.** Créer `bench/scenarios/model_registry.yaml`. C est seul owner. A/B proposent des entrées via `bench/SYNC.md`.

Shape minimal :

```yaml
- canonical_name: ASLS-AOM-compact-cv5-numpy
  aliases: [asls-aom-compact-cv5, ASLS_AOM_compact_cv5]
  model_class: AOMPLSRegressor
  module: nirs4all.operators.models.sklearn.aom_pls
  config_template: bench/scenarios/configs/asls_aom_compact_cv5.yaml
  task_types: [regression]
  input_constraints: {min_n: 30, min_features: 20}
  supports_predefined_test_split: true
  inner_cv_nested: true
  runtime_tier: fast
  maturity: locked
```

**C2 - Scenario exporter.** Créer `bench/export_benchmark_scenarios.py` qui produit :

- `bench/scenarios/fast_reliable.json`
- `bench/scenarios/strong_practical.json`
- `bench/scenarios/best_current.json`
- `bench/scenarios/exhaustive_research.json`
- `bench/scenarios/README.md`

Exporter logic :

1. Filtrer rows regression OK, source rows ou locked oracle selon usage.
2. Agréger par modèle : coverage, medians sur les deux ratios, IQR, q75/q90, worst-case, wins, runtime, source families, maturity.
3. Pénaliser low coverage, non-nested selectors, q90 toxique, missing registry, source-run-only sans support strict.
4. Émettre des manifests avec candidats, ordre, budget, fallbacks, expected stats.

**C3 - Harness.** Promouvoir le concept `bench/AOM/roadmap_runs/harness/run_benchmark.py` vers `bench/harness/run_benchmark.py`.

API cible :

```bash
python bench/harness/run_benchmark.py \
  --cohort fast12_transfer_core \
  --pipeline bench/scenarios/fast_reliable.json \
  --workspace bench/scenarios/runs/fast_reliable \
  --seeds 0,1,2,3,4 \
  --n-jobs -1
```

Requis :

- resumable par `(dataset, seed, model, selection)`;
- schema CSV unifié ;
- skip/failed rows explicites ;
- stats helpers Wilcoxon, bootstrap CI, Friedman-Nemenyi, Nadeau-Bengio ;
- hook external smoke futur, sans bloquer ce cycle.

**C4 - Preset validation and dashboard.** Créer `bench/build_dashboard.py` ou étendre l'existant pour lire `bench/scenarios/runs/` et générer leaderboards, heatmaps, runtime CDFs, tests stat, failure/fallback tables.

**C5 - RF model leaves.** Spécifier `bench/AOM_v0/rf_model_leaves/SPEC.md`, puis prototype `RFModelLeavesRegressor` uniquement comme diagnostic/selector dans `exhaustive_research`.

### Acceptance

- Toute entrée `locked` du master a soit une entrée registry runnable, soit un flag `not_runnable_in_production`.
- Les quatre JSON scenarios existent et passent un lint.
- Le harness lance au moins un dummy/smoke run end-to-end.
- Les sorties de runs sont consommables par dashboard et par master synthesis.

---

## 9. Presets Produit

### 9.1 `fast_reliable`

Budget : secondes à quelques minutes. Pas de TabPFN, pas de multiview, pas de NN.

Ordre cible :

1. PLS tuned.
2. Ridge tuned.
3. `ASLS-AOM-compact-cv5-numpy`.
4. AOM-Ridge compact/global si runtime tier `fast` confirmé.

### 9.2 `strong_practical`

Budget : minutes à moins d'une heure.

Ajoute :

- AOMRidge AutoSelect/Blender headline-spxy3 après nested audit.
- TabPFN raw avec gate `n <= 5000`, `p <= 1000`.
- TabPFN opt avec même gate.
- Un ensemble mean/trimmed nested si OOF export validé.

### 9.3 `best_current`

Budget : 1-3h.

Ajoute :

- TabPFN-HPO-preprocessing avec budget/rationale de preprocessing selection.
- Top ASLS-AOM variants.
- Top AOM-Ridge variants.
- `AdaptiveSuperLearner` seulement après full-57 complet et locked.
- `V2L-Residual-AOMPLS` seulement si gate B passe.
- `FCKStaticResidual` seulement si gate FCK passe.

### 9.4 `exhaustive_research`

Budget : multi-hour à overnight.

Inclut :

- multiseed/repeated-CV top candidates ;
- HPO preprocessing large TabPFN/AOM ;
- FCK learned kernels seulement après static GO ;
- residual NN sweeps ;
- RFModelLeavesRegressor ;
- rows `exploratory` explicitement flaggées.

---

## 10. Validation et Qualité

### 10.1 Rapport obligatoire

Chaque score-card doit rapporter :

- median, q75, q90, worst-case clipped ratio ;
- wins/N vs PLS, Ridge, AOM-Ridge, TabPFN raw, TabPFN opt, paper CNN ;
- `score_ratio_vs_source_run_pls` et `score_ratio_vs_dataset_pls` ;
- runtime median/q90 ;
- failures/fallbacks ;
- p-values et CI adaptés au niveau de validation.

### 10.2 Tests statistiques

| Niveau | Cohorte | Tests |
|---|---|---|
| Smoke | `fast12_transfer_core` | Wilcoxon paired + sanity q90 |
| Audit | `audit20_transfer_core` | Wilcoxon + bootstrap CI |
| Production | full-57 | Friedman-Nemenyi + Nadeau-Bengio + per-dataset deltas |

### 10.3 CI et hygiene

- `ruff`, `mypy` si configuré localement, et tests unitaires ciblés sur les modules modifiés.
- Parity tests pour refactors AOM/POP.
- Aucun selector/stacker dans `strong_practical` ou `best_current` sans OOF nesting prouvé.
- Pas de dead code : archive avec README ou suppression.
- Les nouveaux scripts doivent être resumable et idempotents.

---

## 11. Calendrier Réaliste

### Semaines 1-2

- P0 master freeze et schema maturity.
- C registry skeleton + exporter skeleton.
- A nested audit AOM-Ridge.
- B audit r20 et préparation r21.

### Semaines 3-4

- C harness smoke end-to-end sur `fast12_transfer_core`.
- A complete elite partial runs ou produit exclusion memo.
- B lance r21 multiseed si compute disponible.
- B FCKStatic smoke.

### Semaines 5-8

- Presets `fast_reliable` et `strong_practical` validés sur fast12/audit20.
- FCK extended audit20/full-57.
- Residual gate B3.
- Dashboard initial.
- `best_current` candidate set gelé.

### Mois 3-6

- Full-57 complet pour presets.
- P1 drafting prioritaire.
- P2 seulement si ASL nested full-57 est solide.
- P3 seulement si residual gate passe ; sinon memo negative result.

---

## 12. Briefs de Lancement Agents

### Agent 0 / C-bootstrap

```text
Tu es Agent C-bootstrap sur nirs4all. Lis bench/PLAN_REPRISE_2026-05.md, puis exécute uniquement la Phase P0. Ne modifie pas les zones A/B sauf lecture. Objectif: freeze benchmark_master_results.csv, ajouter protocol_maturity via build_benchmark_synthesis.py, auditer les compteurs, produire bench/MASTER_CSV_FREEZE.md avec P0_DONE ou bloqueurs, puis mettre à jour bench/SYNC.md.
```

### Agent A

```text
Tu es Agent A, owner Classical Production Pack. Attends que bench/MASTER_CSV_FREEZE.md contienne P0_DONE. Lis bench/PLAN_REPRISE_2026-05.md sections 6, 9, 10 et les sources AOM. Travaille uniquement sur bench/AOM, bench/AOM_v0/aompls, bench/AOM_v0/Ridge, bench/AOM_v0/multiview sauf propositions via bench/SYNC.md. Priorités: nested audit AOM-Ridge headline-spxy3, complete elite partial runs ASL/AOM-Ridge, candidate cards registry pour Agent C, puis runs via bench/harness.
```

### Agent B

```text
Tu es Agent B, owner Residual NN et FCK. Attends P0_DONE. Lis bench/PLAN_REPRISE_2026-05.md sections 7, 9, 10 et les sources nicon_v2/fck_pls. Travaille uniquement sur bench/nicon_v2 et bench/fck_pls sauf propositions via bench/SYNC.md. Priorités: audit r20, préparer/lancer r21 multiseed avec fallback do-no-harm, implémenter FCKStaticTransformer, puis FCK extended/residual avec GO/NO-GO documenté.
```

### Agent C

```text
Tu es Agent C, owner MLOps Spine et scenario product. Après P0_DONE, lis bench/PLAN_REPRISE_2026-05.md sections 8, 9, 10. Travaille sur benchmark synthesis, bench/scenarios, bench/harness, dashboard, Subset_analysis en lecture. Priorités: model_registry.yaml, export_benchmark_scenarios.py, run_benchmark.py resumable, validation outputs, dashboard. Tu es seul owner du registry ; A/B proposent des entrées via bench/SYNC.md.
```

---

## 13. Prochaines Actions

1. Lancer Agent 0 / C-bootstrap sur P0.
2. Ne pas lancer A/B tant que `bench/MASTER_CSV_FREEZE.md` ne conclut pas `P0_DONE`.
3. Après P0, lancer A/B/C avec les briefs ci-dessus.
4. Faire une review Codex round 2 après :
   - master freeze terminé ;
   - registry skeleton créé ;
   - AOM-Ridge nested audit terminé ;
   - r20 audit terminé.

---

## 14. Références Locales

- Plan central : `bench/PLAN_REPRISE_2026-05.md`
- Backlog opérationnel : `bench/Backlog`
- Master CSV : `bench/benchmark_master_results.csv`
- Synthesis actuelle : `bench/benchmark_synthesis.md`
- Review exploration : `bench/model_exploration_review.md`
- Subsets : `bench/Subset_analysis/RETHOUGHT_SUBSETS.md`
- AOM roadmap : `bench/AOM/ROADMAP.md`
- Multiview summary : `bench/AOM_v0/multiview/docs/SUMMARY.md`
- nicon status : `bench/nicon_v2/docs/STATUS.md`
- FCK : `bench/fck_pls/README.md`
- Library roadmap séparée : `Roadmap.md`
