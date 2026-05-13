# Programme experimental futur

Ce programme repart des objectifs reformules et du bilan actuel. Il privilegie
des phases courtes, testables, avec gates explicites. L'ordre est important:
ne pas passer a un objectif plus ambitieux si le niveau precedent n'est pas
stabilise.

## Phase 0 - Stabiliser le perimetre

Objectif: rendre le travail retrouvable et separer les niveaux `X`, `(X, Y)`,
embedding et PFN.

Travaux:

- Utiliser `bench/synthesis` comme repertoire canonique.
- Conserver `nirs4all/synthesis` comme moteur de production, pas comme carnet
  d'experimentation.
- Tenir a jour trois documents: objectifs, travail realise, programme futur.
- Marquer chaque experience par niveau: `augmentation`, `x_realism`, `xy`,
  `embedding`, `pfn_prior`.

Gate:

- Toute nouvelle experience doit indiquer son niveau, ses entrees autorisees,
  ses sorties, son seed, sa commande et son rapport.

Livrable:

- Index `bench/synthesis/README.md` maintenu.

## Phase 1 - Atlas des datasets reels

Objectif: connaitre precisement les distributions reelles avant d'ameliorer les
generateurs.

Experiences:

1. Construire un inventaire de `bench/tabpfn_paper/data` et des datasets X-bank:
   taille, nombre de longueurs d'onde, unite, range, type de signal, presence de
   `Y`, split, matrice, instrument si disponible.
2. Classer les datasets en regimes experimentaux: small-N/high-P, large-N,
   grilles courtes, grilles longues, reflectance, absorbance, derivees,
   matrices plante/sol/fuel/food/pharma.
3. Selectionner un panel de travail en trois tailles:
   - smoke: 3 datasets;
   - representative: 10-12 datasets;
   - full: tous les datasets exploitables.
4. Identifier les donnees qui bloquent le pur mecaniste: metadata de geometrie,
   pathlength, capteur, preparation echantillon, temperature/humidite.

Gate:

- Chaque dataset du panel representative doit avoir un `dataset_card` minimal:
  chemin, `X`, `Y`, grille, type de signal, n_train/n_test, risques connus.

Livrables:

- `bench/synthesis/dataset_atlas/`
- CSV d'inventaire.
- Rapport court listant les datasets utilisables par phase.

## Phase 2 - Augmentations `X -> X'`

Objectif: produire des transformations locales utiles pour la robustesse des
modeles existants.

Experiences:

1. Definir une banque minimale d'augmentations: baseline drift, scatter,
   multiplicative gain/offset, bruit heteroscedastique, wavelength shift,
   resolution/broadening, band dropout.
2. Calibrer les amplitudes par regime de dataset, pas globalement.
3. Tester chaque augmentation seule puis en composition sur PLS (+cartesian preprocessing)/AOMPLS/Ridge(+cartesian preprocessing)/AOMRidge/CNN/TabPFN (+2-3 preprocessings).
ndlr: Utiliser AOMPLS en fast smoke model avant de se lancer dans plus gros. Les AOMPLS et AOMRidge sont rapides et efficace. (Ou prendre les presets listés avec le dashboard aussi)
4. Mesurer les effets sur:
   - performance moyenne;
   - queues de target;
   - sensibilite aux splits;
   - degradation quand l'augmentation est trop forte.

Gate:

- Une augmentation est gardee seulement si elle ameliore ou stabilise au moins
  un groupe de datasets sans degradation mediane claire sur le panel.

Livrables:

- Presets d'augmentation documentes.
- Tableau ablation par augmentation.
- Recommandations "utiliser / eviter / seulement diagnostic".

## Phase 3 - Realisme `X` hybride

Objectif: transformer la piste PCA + `knn_mixup` en generateur `X` valide sur
panel complet.

Experiences:

1. Rejouer `exp32`, `exp33`, `exp34` depuis les chemins canonises.
2. Ajouter des garde-fous anti-fuite:
   - distance minimale aux voisins reels;
   - detection de quasi-duplicats;
   - AUC avec plusieurs discriminateurs;
   - evaluation train-real/test-synthetic et inverse sur splits internes.
3. Etendre le panel de 10 datasets au panel full.
4. Analyser les echecs small-N/high-P: TIC, grapevine_chloride et autres cas
   similaires.
5. Comparer trois familles:
   - hybride actuel PCA + `knn_mixup`;
   - GMM/covariance robuste;
   - squelette mecaniste enrichi + residus statistiques.

Gate:

- Sur le panel representative: RF AUC moyen proche de 0.5, aucun dataset avec
  evidence de quasi-copie, et diversite synthetique mesurable.
- Sur le panel full: rapport des echecs, pas seulement un score moyen.

Livrables:

- `x_realism_panel_full.md`
- CSV par dataset.
- Decision par famille de generateur.

## Phase 4 - Generation `(X, Y)` ciblee

Objectif: evaluer si le synthetique peut ameliorer la robustesse predictive,
notamment aux extremes de `Y`.

Experiences:

1. Limiter la premiere vague aux datasets `go` de `y_predictor_feasibility`.
2. Pour chaque dataset, entrainer le modele `Y` uniquement sur official-train.
3. Generer `X'` par le meilleur generateur `X`, puis `Y'` par un modele gele.
4. Evaluer:
   - KS et Wasserstein sur `Y`;
   - AUC joint `(X, Y)`;
   - TSTR et TRTS;
   - performance par quantile de `Y`;
   - gain quand on ajoute `synthetic` au train reel.
5. Comparer plusieurs politiques de generation de `Y`:
   - ridge/PCA;
   - random forest;
   - modeles mixtes par regimes;
   - noise heteroscedastique conditionnel.

Gate:

- Un dataset passe seulement si l'ajout synthetique ameliore une metrique
  downstream predeclaree ou reduit l'erreur aux extremes sans degrader le centre.
- Les bons TSTR seuls ne suffisent pas si la distribution de `Y` ou l'AUC jointe
  reste manifestement mauvaise.

Livrables:

- Rapport `xy_augmentation_value.md`.
- Liste des datasets ou la generation `(X, Y)` est utile, inutile ou dangereuse.

## Phase 5 - Encodeur latent universel

Objectif: apprendre une representation spectrale transferable avec des donnees
reelles non etiquetees et du synthetique controle.

Experiences:

1. Implementer le minimum viable du plan `bench/synthesis/ViTnirs`:
   loader multi-dataset, collate variable-length, masquage par blocs, encodeur
   Perceiver 1D, decodeur MAE.
2. Preentrainer en trois conditions:
   - real-only;
   - real + augmentations;
   - real + augmentations + synthetique `X`.
3. Evaluer avec Ridge/PLS/TabPFN sur embeddings geles.
4. Tester LODO strict pour distinguer memorisation de transferabilite.
5. Ajouter ensuite des vues synthetiques appariees pour loss contrastive.

Gate:

- L'encodeur doit battre ou egaler des features simples stables sur plusieurs
  datasets, pas seulement reduire la loss MAE.
- Le synthetique est conserve dans le pretraining seulement s'il apporte un gain
  downstream ou LODO.

Livrables:

- Prototype `bench/synthesis/ViTnirs` executable.
- Rapport downstream embeddings.
- Decision sur l'utilite du synthetique pour le latent space.

## Phase 6 - Priors NIRS-PFN / ICL

Objectif: passer de la generation d'echantillons a la generation de taches.

Experiences:

1. Formaliser un task sampler: taille de contexte, taille de query, distribution
   de grilles, bruit, regimes, targets, confondeurs.
2. Construire des episodes synthetiques avec difficulty labels.
3. Tester d'abord un modele ICL/PFN miniature ou un proxy simple, pas un gros
   entrainement final.
4. Comparer a TabPFN et aux meilleurs baselines `bench` sur les memes datasets.
5. Ablater les composants du prior: physique, statistiques, regimes, target
   nonlinearities, instrument variation.

Gate:

- Le prior est utile seulement s'il ameliore une evaluation leave-dataset-out ou
  low-shot par rapport aux baselines fortes.
- Si le gain vient uniquement d'un dataset ou d'un type de target, le prior reste
  specialise et ne doit pas etre presente comme fondation NIRS.

Livrables:

- Specification du task sampler.
- Smoke ICL/PFN.
- Decision go/no-go pour entrainement plus lourd.

## Phase 7 - Promotion vers `nirs4all`

Objectif: integrer seulement les briques prouvees.

Criteres de promotion:

- API minimale claire.
- Tests unitaires et tests de non-regression.
- Rapport bench reproductible.
- Gain downstream ou utilite de validation demontree.
- Pas de dependance implicite a un dataset local non declare.

Ce qui peut etre promu en premier:

- augmentations spectrales robustes;
- utilitaires d'atlas/validation;
- generateur `X` hybride si les gates anti-fuite passent;
- contracts de task sampling si stabilises.

Ce qui doit rester en bench:

- notebooks;
- modeles experimentaux;
- generateurs dataset-specifiques;
- seuils et gates non stabilises;
- essais PFN/ICL avant preuve de valeur.
