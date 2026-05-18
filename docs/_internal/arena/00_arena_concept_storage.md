# Base de preuves NIRS4All + Arena — concept, produits et stockage minimal

> Document de cadrage. Il rend explicite la raison d'etre du projet : construire d'abord une **base de preuves interne pour cribler les methodes NIRS**, puis publier une partie de cette connaissance via le website **Arena**. Il fixe aussi une regle d'architecture : **ne pas transformer `nirs4all` en base de donnees de benchmark**. Le workspace `nirs4all` reste le journal d'execution ; la base annexe versionnee stocke uniquement ce qui manque pour l'analyse, la recommandation et la publication.

| Champ | Valeur |
|---|---|
| Version | v0.1 |
| Statut | Cadrage a relire avant Phase 0b |
| Liens | [cartographie datasets](01_dataset_cartography_tasks.md), [runtime](04_runtime_sketch.md), [web](05_web_minimal_sketch.md), [roadmap](07_nirs4all_arena_roadmap.md) |

## 1. Finalite

Le projet a trois produits, alimentes par les memes runs mais avec des horizons differents.

1. **Court terme : une base de reference pour le criblage interne.** L'equipe passe au minimum 10 datasets par mois, teste systematiquement des couples/triples `(preprocessing, modele, reglage)`, et construit une memoire collective : "cette combinaison marche rarement", "celle-ci est un default solide", "celle-ci marche seulement sur tel type de dataset". L'objectif immediat est de gagner du temps quand on crible des methodes.
2. **Moyen terme : des recommandations de trousse a outils.** La base doit permettre de dire, preuves a l'appui : "dans notre corpus, `ASLSBaseline + PLS` est un choix tres efficace dans tel contexte", ou au contraire "ce modele avec ces preprocessings echoue presque toujours". Ces recommandations doivent etre conditionnelles au domaine, au type de cible, au split, a la taille du dataset et au profil spectral.
3. **Long terme : Arena, le website public.** Arena est la couche de publication : leaderboard, matrices, fiches methodes/datasets, et comparaison communautaire. Elle ne doit pas masquer l'objectif principal : accumuler des preuves robustes et exploitables.

Si, apres 2 ans, la base couvre par exemple 300 datasets, 300 methodes/recettes et des contributions externes, elle peut devenir le coeur d'un papier methodologique ambitieux. La valeur scientifique ne viendrait pas d'un leaderboard seul, mais de l'analyse des regularites : quelles familles de preprocessings sont utiles, quand elles ne le sont pas, quelles methodes generalisent, quelles conclusions classiques tiennent ou tombent a grande echelle.

## 2. Questions auxquelles la base doit repondre

La base doit etre utile avant meme le website public. Les sorties attendues sont operationnelles :

- **Anti-patterns** : recettes qui echouent souvent, ou qui ne battent presque jamais les baselines simples.
- **Defaults solides** : recettes recommandables par defaut pour un contexte donne.
- **Frontieres de validite** : conditions sous lesquelles une methode marche ou cesse de marcher.
- **Robustesse** : sensibilite aux splits, seeds, taille d'entrainement, domaine, instrument, preprocessing.
- **Cout** : gain de score vs temps CPU/GPU, memoire, complexite de mise en oeuvre.
- **Evidence cards** : pour chaque recette importante, une fiche avec nombre de datasets testes, distribution des scores relatifs, echecs, domaines couverts, exceptions notables.

Une phrase comme "ce modele avec ces preprocessings ca marche quasi jamais" doit etre traçable vers une evidence card : `n_datasets`, `n_runs`, ratio vs baseline, fragilite, IC/quantiles et exemples de contre-cas.

## 3. Non-objectifs

- Ne pas remplacer le workspace `nirs4all`.
- Ne pas etendre le schema coeur de `WorkspaceStore` avec des tables specifiques a l'arene.
- Ne pas dupliquer dans une base "arena" les predictions, scores, durees et artefacts deja presents dans les workspaces.
- Ne pas rendre `nirs4all` dependant de `nirs4all_arena`.
- Ne pas faire du website public le centre du projet en v0.1. Le website vient apres la base de preuves.

La direction v0.1 est : **`nirs4all_arena` depend de `nirs4all`, jamais l'inverse**.

## 4. Ce que les workspaces contiennent deja

Le stockage courant de `nirs4all` est `WorkspaceStore` SQLite (`store.sqlite`) + sidecars Parquet (`arrays/`) + artefacts (`artifacts/`). Les concepts visibles dans le studio/webapp, notamment la decouverte de workspaces et les vues Runs/Inspector, confirment que beaucoup d'informations utiles sont deja disponibles sans changer la librairie.

| Besoin arena | Deja disponible dans workspace `nirs4all` |
|---|---|
| Run/session | `runs.run_id`, `name`, `status`, `created_at`, `completed_at`, `config`, `datasets`, `summary`, `error` |
| Pipeline execute | `pipelines.pipeline_id`, `expanded_config`, `original_template`, `generator_choices`, `dataset_name`, `dataset_hash`, `metric`, `duration_ms`, `best_val`, `best_test`, `status`, `error` |
| Modele et chaine | `chains.model_class`, `model_name`, `preprocessings`, `fold_strategy`, `best_params`, scores CV/final/agreges |
| Scores atomiques | `predictions.val_score`, `test_score`, `train_score`, `metric`, `scores`, `task_type`, `n_samples`, `n_features`, `fold_id`, `partition` |
| Predictions/residus | sidecars Parquet via `ArrayStore` |
| Artefacts | `artifacts.artifact_path`, `content_hash`, `artifact_type`, `format`, `size_bytes`, caches |
| Logs | `logs.event`, `duration_ms`, `message`, `details`, `level` |

Ce niveau suffit pour les scores, durees, et beaucoup de metadonnees d'execution. L'arene ne doit pas les recopier comme source primaire.

## 5. Ce qui manque vraiment

Les workspaces ne suffisent pas pour construire une base de preuves exploitable, car ils ne portent pas tout le contexte analytique, editorial et protocolaire :

- **Version du protocole** : `grid_version`, politique de scoring, aliasing warnings, version des baselines.
- **Version des datasets** : dataset registry release, `DatasetCard`, hash canonicalise, licence, citation, split masks et `split_hash`.
- **Soumission** : auteur, format A/B/C, manifest soumis, fichiers recus, hashes, lockfiles, bridge R, CITATION.cff, statut editorial.
- **RunSpec canonique** : hash du run atomique dans la grille, facteurs `(D, S, K, P, A, F, T, N, block)`, compatibilite/skip.
- **Recette normalisee** : identifiant stable pour un couple/triple `(preprocessing, model, hyperparams)` afin de comparer `ASLSBaseline + PLS` entre plusieurs runs et workspaces.
- **Environnement controlant** : EnvCard complete, SeedCard, container digest, package lockfile, `sessionInfo()` R.
- **Verification** : replays de reproductibilite, canaris, audits anti-leakage, decisions humaines.
- **Analyse** : snapshots de recommandations, evidence cards, anti-patterns, decisions d'integration dans la trousse a outils.
- **Publication** : snapshots figes des vues publiques et de leurs hashes quand Arena est exposee.

Ces informations sont specifiques a la base de preuves/Arena. Elles doivent vivre dans une **base annexe compatible SQLite**, pas dans le schema general de `nirs4all`.

## 6. Architecture de stockage v0.1

```
nirs4all_arena/
  arena.sqlite                 # index protocolaire et editorial
  submissions/<submission_id>/ # fichiers recus, manifests valides, lockfiles
  registry/                    # DatasetCards, split masks, citations
  evidence/                    # evidence cards, recommendations, anti-patterns
  public_json/                 # exports statiques pour le site

workspaces/
  <campaign_or_submission>/
    store.sqlite               # ecrit par nirs4all
    arrays/                    # predictions/residus Parquet
    artifacts/                 # bundles et modeles
```

`arena.sqlite` est un index relationnel. Il reference les workspaces par chemin/URI et identifiants stables ; il ne devient pas le stockage primaire des runs.

## 7. Schema minimal de `arena.sqlite`

### Tables essentielles

| Table | Role |
|---|---|
| `arena_meta` | version du schema arena, date de creation, hash de release |
| `protocol_versions` | versions grille/protocole/scoring/baselines et chemins des fichiers geles |
| `dataset_cards` | `DatasetCard` normalise, licence, citation, statut editorial |
| `dataset_versions` | hash de donnees, source, version registry, date de gel |
| `split_masks` | `dataset_version_id`, `scheme`, `seed`, `mask_path`, `mask_hash` |
| `recipe_definitions` | identifiant stable d'une recette `(preprocessing, model, hyperparams, task)` |
| `submissions` | identite, format A/B/C, statut, auteur, citation, contraintes, manifest hash |
| `submission_files` | role, chemin, taille, hash de chaque fichier soumis ou genere a l'installation |
| `run_specs` | `run_spec_hash` et facteurs canoniques de la grille |
| `run_bindings` | lien `(submission_id, run_spec_hash)` vers `workspace_uri`, `run_id`, `pipeline_id`, `chain_id`, `prediction_id(s)` |
| `env_cards` | environnement de la soumission ou du run atomique, stocke en JSON + hash |
| `seed_cards` | seed demandee, etats PRNG captures quand disponibles, hash |
| `verification_checks` | replay reproductibilite, sanity, canaris, audits et decisions |
| `evidence_snapshots` | fiches recettes/methodes, recommandations, anti-patterns, version d'analyse |
| `publication_snapshots` | hash et chemin des JSON publics publies pour une release |

### Ce qui reste hors de `arena.sqlite`

| Donnee | Source primaire |
|---|---|
| Scores, durees, status d'execution | `workspace/store.sqlite` |
| Predictions, residus, probabilites | `workspace/arrays/*.parquet` |
| Modeles, bundles `.n4a`, artefacts | `workspace/artifacts/` ou `submissions/` selon leur origine |
| Logs detailles d'execution | table `logs` dans `workspace/store.sqlite` |
| Fichiers publics du site | `public_json/` + snapshot dans `publication_snapshots` |

## 8. Jointure analyse/publication

L'indexer peut joindre la base annexe et un ou plusieurs workspaces SQLite sans migration de schema :

```sql
ATTACH DATABASE 'workspaces/campaign_001/store.sqlite' AS ws;

SELECT
  s.submission_id,
  rs.block,
  rs.dataset_alias,
  rs.split_scheme,
  rs.seed,
  p.metric,
  p.val_score,
  p.test_score,
  pl.duration_ms,
  pl.status
FROM run_bindings rb
JOIN submissions s ON s.submission_id = rb.submission_id
JOIN run_specs rs ON rs.run_spec_hash = rb.run_spec_hash
JOIN ws.pipelines pl ON pl.pipeline_id = rb.pipeline_id
LEFT JOIN ws.predictions p ON p.prediction_id = rb.primary_prediction_id;
```

Si plusieurs workspaces sont utilises, l'indexer les parcourt et materialise :

- des exports internes dans `evidence/` : recipe cards, anti-patterns, recommandations ;
- des exports publics dans `public_json/` quand le website Arena est publie.

Les vues peuvent contenir des valeurs denormalisees, mais la source d'audit reste la paire `arena.sqlite` + workspaces.

## 9. Impact attendu sur `nirs4all`

**Objectif v0.1 : aucun changement de schema dans `nirs4all`.**

Le runtime arena doit utiliser les API publiques existantes :

- execution via `nirs4all.run(...)` ou l'orchestrateur courant ;
- persistance dans un workspace standard ;
- lecture via `WorkspaceStore` et `ArrayStore` ;
- bundles via les outils existants.

Si une friction apparait, la preference est de l'encapsuler dans `nirs4all_arena` :

- adaptateur `WorkspaceRunResolver` pour retrouver `run_id/pipeline_id/prediction_id` apres un run ;
- conventions de nommage deterministes dans `run.name` et `pipeline.name` ;
- stockage du mapping exact dans `run_bindings`.

Une extension de `nirs4all` ne devient acceptable que si elle est generique hors arena, par exemple un champ/tags optionnels de contexte d'execution. Elle ne doit pas introduire de tables `arena_*` dans le coeur.

## 10. Idees reprises du studio/Inspector

Le studio existant fournit deja des concepts utiles pour l'operateur arena :

- **Linked workspaces** : scanner plusieurs workspaces et les traiter comme sources decouvrables.
- **Run detail** : groupement par dataset, pipeline, modele, preprocessing, split, statut.
- **Dataset versioning** : hash courant, hash stocke, statut `current/modified/missing`.
- **Resume et reprise** : `summary`, `checkpoints`, `completed_results`, `failed_results`.
- **Drill-down predictions** : acces aux predictions et scatter via `WorkspaceStore` + sidecars.

Ces idees sont utiles pour une UI interne d'operation et d'audit. Le site public v0.1 peut rester statique ; l'Inspector sert de modele pour le drill-down v0.2 ou pour l'outil mainteneur.

## 11. Invariants

1. Un resultat public doit etre traçable vers un `run_spec_hash`, une `submission_id`, un `workspace_uri` et un `prediction_id` ou une erreur typée.
2. Un `run_spec_hash` est calcule uniquement a partir du protocole, du dataset versionne, du split et des facteurs de grille, jamais a partir du score.
3. Une publication porte toujours les hashes de `arena.sqlite`, du registry datasets, de la grille et des JSON exports.
4. Un bump majeur de protocole ou de baseline produit un nouveau snapshot ; les anciens restent lisibles.
5. L'arene peut etre reconstruite a partir de `arena.sqlite`, des workspaces references et des fichiers soumis/versionnes.
