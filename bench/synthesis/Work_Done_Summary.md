# Travail realise - synthese haut niveau

Ce document resume les lignes de travail deja presentes dans le depot. Il ne
remplace pas les rapports techniques; il donne une lecture d'ensemble et pointe
vers les artefacts utiles.

## 1. Moteur de synthese dans la librairie

Le socle maintenu est dans `nirs4all/synthesis`.

Ce qui existe:

- `nirs4all/synthesis/generator.py`: generateur NIRS physique avec composants,
  bandes, baseline, scatter, bruit, instruments, modes de mesure et effets
  environnementaux.
- `nirs4all/synthesis/builder.py`: API fluent `SyntheticDatasetBuilder` pour
  construire des datasets synthetiques.
- `nirs4all/synthesis/targets.py`: generation de targets de regression et
  classification, puis extensions non lineaires, confondeurs et multi-regimes.
- `nirs4all/synthesis/components.py`, `_bands.py`, `instruments.py`,
  `detectors.py`, `measurement_modes.py`, `scattering.py`: briques physiques et
  instrumentales.
- `nirs4all/synthesis/reconstruction/`: pipeline d'inversion/reconstruction
  pour ajuster des parametres a partir de donnees reelles.

Lecture haut niveau: la librairie contient deja beaucoup de briques utiles. Le
probleme n'est pas l'absence de generateur, mais l'alignement entre objectifs,
donnees reelles, validation et usage downstream.

## 2. Prototype physique historique

Chemin canonique: `bench/synthesis/synthetic`.

Ce bloc contient:

- `generator.py`: premier generateur physique Beer-Lambert avec composants et
  bandes NIR.
- `comparator.py`: comparaison real/synthetic par statistiques, pentes,
  courbure, PCA, bruit et pics.
- `dataset_category_identifier.py`: detection de famille produit/matrice a
  partir du spectre, avec templates chimiques et fitting interpretable.
- notebooks d'exploration: `physical_forward_generator.ipynb`,
  `physical_reconstruction_workflow.ipynb`,
  `spectra_synthesis_explorer.ipynb`, `generator_improvement.ipynb`.
- `synthesis_summary.md`: inventaire detaille d'une pipeline de generation en
  25 etapes.

Lecture haut niveau: ce travail a formalise la vision mecaniste complete,
depuis la grille spectrale jusqu'a la target. Il a aussi montre que la liste des
parametres possibles est tres large, donc qu'il faut des gates experimentaux
plus stricts pour eviter de tout optimiser sans critere.

## 3. Programme bench NIRS synthetic PFN

Chemin canonique: `bench/synthesis/nirs_synthetic_pfn`.

Ce repertoire est le plus riche. Il organise les experiences autour des priors
synthetiques, du realisme spectral, de la generation `X/Y`, des gates et des
rapports.

Documents d'orientation importants:

- `README.md`: frontiere bench/librairie et regles de travail.
- `docs/00_CONTEXT_REVIEW.md` a `docs/05_INTEGRATION_GATE.md`: cadrage initial,
  roadmap, validation scientifique et gate d'integration.
- `docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`: resume d'etat au 2026-05-01.
- `docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`: objectif actif de realisme `X`
  par discriminant adversarial.
- `reports/xrealism_final_summary.md`: synthese des meilleurs resultats
  realisme `X`.

### 3.1 Priors et contrats de taches

Les premieres experiences ont rendu executable l'idee de prior NIRS:

- `experiments/exp00_smoke_prior_dataset.py`
- `experiments/exp01_prior_predictive_checks.py`
- `experiments/prior_coverage.py`
- tests dans `tests/test_prior_task.py`, `tests/test_prior_adapter.py`,
  `tests/test_canonical_latents.py`, `tests/test_task_sampling.py`.

Ce travail fournit des contrats et des garde-fous. Il ne demontre pas encore un
PFN utile, mais il clarifie ce qu'est une tache synthetique valide.

### 3.2 Scorecards real/synthetic et gates stricts

Les premiers scorecards ont compare des presets synthetiques aux datasets reels:

- `experiments/exp02_real_synthetic_scorecards.py`
- `experiments/exp03_transfer_validation.py`
- `experiments/exp04_adversarial_auc.py`
- `reports/real_synthetic_scorecards.md`
- `reports/transfer_validation.md`
- `reports/encoder_tabpfn_gate.md`

Conclusion haut niveau: les gates stricts ont majoritairement bloque. Le
realisme brut ne passait pas assez souvent; le transfert a ete bloque; les
experiences d'encodeur/TabPFN ont ete maintenues en attente. C'est un resultat
utile: les briques physiques seules ou les presets initiaux ne suffisaient pas.

### 3.3 Cas DIESEL et limite du pur mecaniste

Une longue sequence d'audits DIESEL a teste des mecanismes support-local,
pathlength/reference, damping, readout et geometrie:

- `experiments/exp08_mechanistic_sentinel_ablation.py`
- `experiments/exp09_sentinel_morphology_audit.py`
- `experiments/exp10_*` a `experiments/exp29_*`
- rapports `reports/r9*.md`, `reports/exp24*.md` a `reports/exp29*.md`
- decisions dans `docs/10_P2B_COUPLED_OPTICAL_DEPTH_DAMPING_DECISION.md` et
  `docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`.

Conclusion haut niveau: aucune variante mecaniste DIESEL n'a clairement
supplante le baseline accepte. Le blocage principal vient du support de donnees:
pas assez de metadata physiques liees aux lignes reelles, pas assez de support
hors bande pour tester certains mecanismes, et geometrie non auditable.

### 3.4 Realisme `X` hybride

Le travail le plus concluant cote `X` est le generateur hybride evalue par
discriminant adversarial:

- `experiments/exp32_hybrid_xrealism_discriminator.py`
- `experiments/exp33_panel_xrealism_discriminator.py`
- `experiments/exp34_mode_comparison.py`
- `docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`
- `reports/xrealism_final_summary.md`
- `reports/xrealism_panel_m2_winner.md`

Idee: squelette mecaniste sur le spectre moyen, residus PCA, sampling des scores
par `knn_mixup`, bruit gaussien par canal. Le generateur n'utilise ni labels, ni
targets, ni split officiel comme oracle.

Resultats a retenir:

- ECOSIS Chla+b: RF AUC descend d'environ 0.92 avec Gaussian PCA a environ 0.58
  avec `knn_mixup k=5 alpha=1`.
- MANURE MgO: RF AUC autour de 0.51 avec la meme famille de recette.
- Panel representatif de 10 datasets: 8/10 dans les 8 points autour de 0.5.
- Les echecs restants concernent surtout des regimes small-N/high-P.

Lecture haut niveau: l'hybride mecaniste-statistique est la piste `X` la plus
prometteuse. Elle doit encore etre securisee contre les quasi-copies et etendue
au panel complet.

### 3.5 Faisabilite `Y` et validation `(X, Y)`

La generation supervisee a ete abordee ensuite:

- `experiments/exp35_y_predictor_feasibility.py`
- `experiments/exp36_mixture_y_pipeline.py`
- `experiments/exp37_xy_validation.py`
- `reports/y_predictor_feasibility.md`
- `reports/xy_validation.md`

Resultats a retenir:

- Gate faisabilite `Y`: 9 datasets `go` sur 12.
- Validation `(X, Y)` sur 9 datasets: plusieurs TSTR sont proches du baseline
  reel, par exemple DIESEL, MANURE, COLZA et MILK; d'autres cas restent faibles,
  notamment ALPINE et grapevine_chloride.
- Les AUC jointes `(X, Y)` et les tests KS de `Y` montrent encore des ecarts.

Lecture haut niveau: la generation `(X, Y)` est possible sur certains datasets,
mais pas encore un outil d'augmentation general. Elle doit devenir un programme
experimental cible, pas un generateur global immediat.

## 4. Encodeur spectral universel

Chemin canonique: `bench/synthesis/ViTnirs`.

Ce bloc contient surtout du cadrage:

- `onboarding.md`: specification d'un encodeur spectral universel a longueur et
  grille variables.
- `roadmap_nirs4all.md`: adaptation a `nirs4all`, usage de `SpectroDataset`,
  augmentations, AsLS cache, MAE et evaluation LODO.

Lecture haut niveau: c'est une direction coherente pour exploiter beaucoup de
`X` sans resoudre immediatement toute la generation `Y`. Elle doit etre reprise
apres consolidation des donnees, augmentations et gates de realisme.

## 5. Ce qui est acquis

- Le projet dispose d'un moteur de generation riche dans `nirs4all/synthesis`.
- Les objectifs ont ete experimentes sous plusieurs angles: physique, statistique,
  adversarial, transfer, XY, latent/PFN.
- Les presets purement mecanistes ou trop generiques ne suffisent pas.
- Les gates stricts ont evite de promouvoir trop tot des resultats faibles.
- L'hybride `X` par PCA + `knn_mixup` est la meilleure piste actuelle pour
  l'indiscernabilite spectrale.
- La generation `(X, Y)` montre des signaux positifs sur certains datasets mais
  reste non concluante comme methode generale.

## 6. Ce qui reste non acquis

- Aucun protocole d'augmentation synthetique n'est encore demontre comme gagnant
  global et robuste.
- Aucun encodeur latent universel n'a encore ete entraine et valide.
- Aucun prior NIRS-PFN n'est encore pret pour entrainement serieux.
- La validation multi-dataset complete reste a refaire avec des seuils stables,
  anti-fuite et ablations.
- Les metadata physiques manquantes limitent fortement le pur mecaniste.
