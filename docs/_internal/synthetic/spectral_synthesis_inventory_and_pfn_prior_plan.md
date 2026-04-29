# Synthese de spectres NIRS dans nirs4all

Etat des lieux, ecarts et plan de travail pour augmentation, donnees synthetiques, encodeur latent et priors NIRS-ICL/NIRS-PFN.

Date d'analyse: 2026-04-28  
Perimetre inspecte: `nirs4all/synthesis`, `nirs4all/operators/augmentation`, `nirs4all/api/generate.py`, integration dataset/pipeline, `bench/synthetic`, tests, documentation source et documentation interne.

## Resume executif

Le socle de generation synthetique est beaucoup plus avance qu'un simple generateur Beer-Lambert. Le code contient aujourd'hui:

- un generateur physique principal, `SyntheticNIRSGenerator`, avec melange Beer-Lambert, pics Voigt, concentrations, baseline, scatter, bruit, broadening, batch effects, instruments, detecteurs, multi-sensor, multi-scan, environnement, particle size, EMSC et edge artifacts;
- un builder de datasets, `SyntheticDatasetBuilder`, qui produit des `SpectroDataset` avec cibles regression/classification, metadata, partitions, agregats, multi-source et cibles non lineaires;
- une API publique `nirs4all.generate` avec raccourcis regression, classification, builder, multi-source, export, produits, categories et imitation de template;
- un catalogue substantiel: 126 composants, 20 domaines applicatifs, 69 agregats, 18 templates produits, 19 archetypes instrumentaux;
- une couche d'augmentation X-seul, integree dans les pipelines, avec operateurs spectraux, environnementaux, scattering, edge artifacts et operateurs extraits de la chaine de synthese;
- une chaine experimentale de reconstruction physique qui essaye d'inferer des variables latentes depuis des donnees reelles puis de regnerer des spectres a partir de distributions apprises;
- des outils de validation: scorecard de realisme, AUC adversariale, registry de benchmarks NIR, TSTR/TRTS partiel;
- des docs internes et notebooks qui explorent la direction NIRS-PFN / encodeur spectral.

La faiblesse principale n'est pas l'absence de briques. C'est l'integration. Plusieurs modules implementent des idees proches mais ne sont pas encore connectes en un workflow coherent de "prior -> task -> dataset -> validation -> pretraining". Les ecarts critiques sont:

- le `PriorSampler` echantillonne des configurations mais ne sait pas generer directement un dataset ou une tache ICL/PFN;
- les noms de domaines du prior ne correspondent pas aux cles de `APPLICATION_DOMAINS`, ce qui degrade le choix des composants;
- `measurement_mode` est expose mais pas reellement applique par `SyntheticNIRSGenerator.generate()`;
- `SyntheticDatasetBuilder.fit_to()` et `generate.from_template()` n'utilisent qu'une petite partie des parametres inferees par `RealDataFitter`;
- les effets environnementaux, scattering et edge artifacts sont disponibles dans le generateur mais peu exposes dans le builder/API;
- l'augmentation supervisee est surtout label-preserving: elle conserve `y` via l'origine de l'echantillon, ce qui est correct pour bruit/baseline/scatter mais incorrect pour un vrai mixup ou pour une modification de composition;
- il n'existe pas encore de generateur de paires multi-vues ou de workflow encodeur/denoiser latent;
- il n'existe pas encore d'objet "synthetic task" pour NIRS-ICL/NIRS-PFN avec contexte, query, metadonnees, domaine, instrument, analyte, grilles variables et parametres latents.

Conclusion courte: le projet est faisable et le socle est solide. La prochaine etape ne devrait pas etre "ajouter encore des effets" en premier, mais normaliser les interfaces: unifier prior, builder, generator, reconstruction et validation autour d'une API de taches synthetiques. Pour le court terme, l'approche la plus pragmatique reste encodeur spectral + TabPFN, puis seulement ensuite un vrai NIRS-PFN si les gains plafonnent.

## Inventaire detaille

### Surface de code

Le perimetre synthetique represente environ 55k lignes en comptant code, benchs et docs internes inspectees. Les blocs principaux sont:

- `nirs4all/synthesis`: environ 34.6k lignes de code Python.
- `nirs4all/synthesis/reconstruction`: environ 4.3k lignes.
- `nirs4all/operators/augmentation`: environ 3.9k lignes.
- `nirs4all/api/generate.py`: 963 lignes.
- `bench/synthetic`: anciens prototypes, notebooks, comparateurs et visualiseurs.
- `docs/_internal/synthetic` et `docs/_internal/nirsPFN`: roadmaps et analyses historiques.

Le contenu est largement teste cote unitaire pour la synthese, les composants, domaines, instruments, validation, prior, builder, produits, agregats, procedural, wavenumber, reconstruction partielle et augmentation. Il existe aussi des tests d'integration augmentation/dataset/splits qui couvrent le risque de fuite entre train/test.

### API publique

L'entree utilisateur principale est `nirs4all.generate`, exposee via `nirs4all/api/generate.py`.

Fonctions disponibles:

- `nirs4all.generate(...)`: generation simple, renvoie un `SpectroDataset` ou `(X, y)`.
- `nirs4all.generate.regression(...)`: dataset regression avec distribution des concentrations, range cible et composant cible.
- `nirs4all.generate.classification(...)`: classes discretes avec separation et poids.
- `nirs4all.generate.builder(...)`: acces au `SyntheticDatasetBuilder`.
- `nirs4all.generate.multi_source(...)`: plusieurs sources NIR/vis/aux/markers.
- `nirs4all.generate.to_folder(...)` et `.to_csv(...)`: export vers formats chargeables.
- `nirs4all.generate.product(...)`: dataset depuis un template produit.
- `nirs4all.generate.category(...)`: combinaison de plusieurs templates produits.
- `nirs4all.generate.from_template(...)`: analyse un dataset reel puis genere un dataset ressemblant.

Ce qui est bon:

- l'API est facile a utiliser;
- elle produit directement des `SpectroDataset`, donc elle s'integre au reste de nirs4all;
- elle expose les cas les plus courants: regression, classification, produit, categorie, multi-source;
- le builder donne un point d'entree avance sans surcharger la fonction principale.

Ce qui reste fragile:

- la fonction `generate()` de haut niveau ne forwarde presque pas les kwargs avances: elle ne traite explicitement que `distribution` et `batch_effects`;
- les options physiques fines sont accessibles via le builder ou le generateur, mais pas comme prior complet;
- `from_template()` repose sur `builder.fit_to()`, qui applique surtout range/step/complexity, pas toute la configuration inferee;
- l'API publique ne propose pas encore `generate.prior(...)`, `generate.task(...)`, `generate.encoder_pairs(...)` ou `generate.pfn_batch(...)`.

### Generateur coeur

`SyntheticNIRSGenerator` est le moteur principal. Il suit globalement cette chaine:

1. choix de la grille de longueurs d'onde: custom wavelengths, instrument grid, range instrument, puis range par defaut;
2. bibliotheque de composants;
3. calcul des spectres purs `E`;
4. sampling des concentrations `C`;
5. melange Beer-Lambert `A = C @ E`;
6. path length;
7. baseline polynomial;
8. slope globale;
9. scatter multiplicatif/additif;
10. batch effects optionnels;
11. shift/stretch en longueur d'onde;
12. broadening instrumental;
13. bruit ou detecteur instrument-specific;
14. multi-sensor stitching;
15. multi-scan averaging;
16. temperature/moisture;
17. particle size/EMSC;
18. edge artifacts;
19. artefacts rares type spike, dead band, saturation.

Les distributions de concentrations disponibles sont:

- `dirichlet`;
- `uniform`;
- `lognormal` normalise;
- `correlated` via matrice de correlation et decomposition de Cholesky.

Les niveaux de complexite sont:

- `simple`: bruit et deformation faibles;
- `realistic`: bruit, scatter, slope et artefacts moderees;
- `complex`: plus de baseline, shift, stretch, bruit et artefacts.

Forces:

- chaine comprehensible et reproductible;
- structure vectorisee;
- metadata de generation;
- `generate_from_concentrations()` pour reutiliser une composition externe, notamment les agregats;
- les effets majeurs de spectroscopie NIR chemometrique sont presents;
- les operateurs de synthese ont ete extraits et reutilises dans l'augmentation.

Limites importantes:

- le generateur reste centre sur l'espace absorbance/Beer-Lambert;
- `measurement_mode` est stocke, mais la simulation creee est toujours `create_transmittance_simulator(...)` et elle n'est pas appelee dans `generate()`;
- reflectance, Kubelka-Munk, ATR, transflectance existent dans `measurement_modes.py`, mais ne pilotent pas encore le generateur principal;
- `generate_from_concentrations()` suit une route legerement differente de `generate()` et n'applique pas toute la logique multi-sensor/multi-scan;
- les parametres inferees depuis un dataset reel ne sont pas tous remappes dans le generateur via une API stable;
- le generateur ne produit pas une representation latente canonique exploitable telle quelle pour un encoder/denoiser.

### Composants, bandes et chimie

Le catalogue actuel contient 126 composants:

- water_related: 2;
- proteins: 12;
- lipids: 15;
- carbohydrates: 18;
- alcohols: 9;
- organic_acids: 12;
- pigments: 18;
- pharmaceuticals: 10;
- polymers: 11;
- solvents: 6;
- petroleum: 5;
- minerals: 8.

Modules importants:

- `components.py`: `NIRBand`, `SpectralComponent`, `ComponentLibrary`, recherche, validation, normalisation.
- `_constants.py`: composants predefinis et presets.
- `_bands.py`: dictionnaire de bandes NIR et fonctions de recherche.
- `wavenumber.py`: conversion nm/cm-1, zones Vis-NIR, overtones, combinations, broadening.
- `procedural.py`: generation de composants chimiquement plausibles avec groupes fonctionnels, overtones, combinations, hydrogen bonding et calculs en wavenumber.

Point fort: le code a deja corrige une faiblesse signalee dans les anciens docs NIRS-PFN. Ceux-ci parlent parfois de 31 composants; l'etat courant du code est 126 composants plus un generateur procedural.

Point faible: le generateur principal n'utilise par defaut qu'un petit ensemble pour `realistic`/`complex` (`water`, `protein`, `lipid`, `starch`, `cellulose`). Le procedural existe, mais n'est pas encore une source de diversite active dans le `PriorSampler` ou dans une API de pretraining.

### Domaines, produits et agregats

`domains.py` contient 20 domaines applicatifs:

- agriculture_grain, agriculture_forage, agriculture_oilseeds, agriculture_fruit;
- food_dairy, food_meat, food_bakery, food_chocolate;
- pharma_tablets, pharma_powder_blends, pharma_raw_materials;
- petrochem_fuels, petrochem_polymers;
- textile_natural, textile_synthetic;
- environmental_soil, environmental_water;
- beverage_wine, beverage_juice;
- biomedical_tissue.

Chaque domaine decrit des composants typiques, des priors de concentration, une plage de longueurs d'onde, une complexite, un mode de mesure et des types d'echantillons.

`products.py` contient 18 templates produits, par exemple:

- wheat_variable_protein;
- milk_variable_fat;
- cheese_variable_moisture;
- meat_variable_fat;
- tablet_variable_api;
- tablet_moisture_stability;
- universal_protein_predictor;
- universal_fat_predictor.

`_aggregates.py` contient 69 agregats de composition. C'est une brique utile pour generer des datasets X/Y plus realistes qu'un Dirichlet generique.

Forces:

- les objets "produit" et "agregat" reintroduisent des proportions plausibles;
- les templates produits savent generer des targets utiles;
- ils sont exposes via `generate.product` et `generate.category`.

Limites:

- les domaines, produits, agregats et prior sampler ne forment pas encore une meme ontologie;
- le prior utilise des noms courts (`grain`, `dairy`, `tablets`, etc.) qui ne correspondent pas aux cles de `APPLICATION_DOMAINS` (`agriculture_grain`, `food_dairy`, `pharma_tablets`, etc.);
- quand `PriorSampler.sample_components(domain)` recoit ces noms courts, `get_domain_config(domain)` echoue et tombe sur un fallback generique;
- le fallback contient `carbohydrate`, qui n'est pas forcement un composant predefini valide comme tel;
- il manque un validateur qui garantit qu'un domaine echantillonne par le prior produit toujours une bibliotheque de composants executable.

### Instruments, detecteurs et modes de mesure

Le code contient 19 archetypes instrumentaux, dont:

- foss_xds;
- bruker_mpa;
- viavi_micronir;
- scio;
- tellspec;
- linksquare;
- asd_fieldspec;
- neospectra_micro;
- thermo_antaris;
- foss_infratec;
- buchi_nirmaster;
- metrohm_ds2500.

`instruments.py` gere:

- categorie instrumentale;
- detecteur;
- monochromateur;
- plage spectrale;
- resolution;
- precision wavelength;
- bruit photometrique;
- SNR;
- stray light;
- warm-up drift;
- sensibilite temperature;
- vitesse scan;
- integration time;
- multi-sensor;
- multi-scan.

`detectors.py` gere les reponses et bruits de detecteurs:

- Si;
- InGaAs;
- extended InGaAs;
- PbS;
- PbSe;
- MEMS;
- MCT.

`measurement_modes.py` modelise:

- transmittance;
- reflectance avec Kubelka-Munk apparent;
- transflectance;
- ATR;
- interactance/fiber optic partiel.

Forces:

- la granularite instrument/detecteur est deja suffisante pour creer une vraie diversite cross-machine;
- les grilles instrumentales permettent d'apprendre des representations robustes aux resolutions et plages variables;
- les detecteurs permettent d'aller au-dela du bruit gaussien simple.

Limite critique:

- le mode de mesure n'est pas encore cable dans la chaine `SyntheticNIRSGenerator.generate()`. Pour une prior NIRS-PFN, c'est critique: reflectance, transmittance et transflectance ne sont pas de simples variations de bruit, ce sont des geometries de mesure differentes.

### Operateurs d'augmentation

Le package `nirs4all/operators/augmentation` contient plusieurs familles.

Operateurs spectraux generiques:

- bruit additif gaussien;
- bruit multiplicatif;
- drift lineaire;
- drift polynomial;
- wavelength shift;
- wavelength stretch;
- local wavelength warp;
- magnitude warp;
- band perturbation;
- smoothing jitter;
- unsharp spectral mask;
- band masking;
- channel dropout;
- spike noise;
- local clipping;
- mixup;
- local mixup;
- scatter simulation MSC.

Operateurs splines:

- smoothing;
- perturbations X;
- perturbations Y;
- simplification de courbe.

Operateurs environnement/scattering/edge:

- temperature;
- moisture;
- particle size;
- EMSC distortion;
- detector roll-off;
- stray light;
- edge curvature;
- truncated peak;
- wrapper edge artifacts.

Operateurs extraits de la synthese:

- `PathLengthAugmenter`;
- `BatchEffectAugmenter`;
- `InstrumentalBroadeningAugmenter`;
- `HeteroscedasticNoiseAugmenter`;
- `DeadBandAugmenter`.

Ces operateurs sont importants car ils permettent exactement le premier objectif: augmenter `X` seul avec les memes effets physiques que la synthese. Plusieurs sont wavelength-aware via `SpectraTransformerMixin` et declarent s'ils requierent ou acceptent les longueurs d'onde.

Integration pipeline:

- `sample_augmentation` ajoute de nouveaux samples en entrainement uniquement;
- `feature_augmentation` ajoute ou remplace des processings/features;
- l'indexer garde un champ `origin` pour chaque sample;
- les augmentations restent attachees a l'origine afin d'eviter les fuites cross-validation;
- `y_indices()` mappe les samples augmentes vers le `y` de leur origine.

Point fort: le design anti-leakage est bon. Une augmentation d'un sample train reste train, une augmentation d'un sample test ne fuit pas dans train.

Point faible: cette architecture suppose implicitement que l'augmentation preserve la cible. C'est vrai pour bruit, baseline, scatter, broadening, shift raisonnable, edge artifact faible. Ce n'est pas vrai pour un mixup supervise si `y` doit etre mixe, ni pour une augmentation qui modifie la composition latente. `MixupAugmenter` documente "modifie X et y", mais `transform()` ne renvoie actuellement que `X`. Dans une pipeline supervisee, le `y` reste celui de l'origine, ce qui rend ce cas mathematiquement faux.

### Builder et generation X/Y

`SyntheticDatasetBuilder` ajoute une couche dataset autour du generateur:

- choix des features: range, step, complexite, composants, bibliotheque, custom params, instrument, mode de mesure;
- choix de la grille: wavelengths custom ou grille instrument;
- targets regression: distribution, range, component, transform;
- targets classification: nombre de classes, separation, poids, methode;
- metadata: ids, groupes, repetitions;
- multi-source;
- agregats;
- partitions;
- batch effects;
- targets non lineaires;
- confounders;
- multi-regime target landscape;
- sortie dataset ou arrays;
- exports.

Forces:

- c'est deja une API de generation X/Y exploitable;
- elle sait produire des targets plus riches que `C[:, k]`;
- elle sait creer des datasets compatibles pipeline;
- elle sait creer du multi-source.

Limites:

- elle ne permet pas encore de declarer directement environmental/scattering/edge configs;
- elle ne mappe pas toute la sortie de `RealDataFitter.to_full_config()`;
- `fit_to()` est trop shallow pour "imiter" un dataset reel: il applique essentiellement wavelength start/end/step et complexity;
- elle n'a pas encore de notion de "tache" au sens ICL/PFN: contexte/query, splits conditionnels, variable sample count, target function samplee, metadata de prior.

### Prior sampler

`prior.py` contient:

- `NIRSPriorConfig`;
- `PriorSampler`;
- `sample_prior`;
- `sample_prior_batch`;
- contraintes domaine -> instrument category -> wavelength/mode/noise -> matrix -> particle/scatter/water -> components -> target type.

Le DAG conceptuel est bon:

```
Domain -> Instrument Category -> Wavelength Range / Resolution / Mode / Noise
       -> Matrix Type -> Particle Size / Scattering / Water Activity
       -> Component Set -> Concentration Distributions
       -> Target Type
```

Mais ce module retourne seulement un dictionnaire de configuration. Il n'existe pas encore de conversion robuste:

```
prior sample -> SyntheticNIRSGenerator -> SyntheticDatasetBuilder -> SpectroDataset -> PFN task
```

Ecarts majeurs:

- mismatch des noms de domaines;
- pas de mapping de `target_config` vers `with_targets`, `with_classification`, `with_nonlinear_targets`, etc.;
- `temperature`, `particle_size`, `noise_level`, `matrix_type` ne sont pas convertis en configs environnement/scattering/generator;
- pas de sample de grilles variables sous forme standardisee;
- pas d'objet tache ICL/PFN.

### Reconstruction et fitter

La sous-arborescence `nirs4all/synthesis/reconstruction` est tres pertinente pour l'objectif "denoiser/compresseur latent".

Elle contient:

- `CanonicalForwardModel`: modele physique sur grille canonique haute resolution;
- `InstrumentModel`: warp, ILS, stray light, gain/offset, resampling vers grille cible;
- `DomainTransform`: absorbance, transmittance, reflectance/Kubelka-Munk;
- `PreprocessingOperator`: none, derivatives, SNV, MSC, detrend, mean center;
- `ForwardChain`: chaine forward complete;
- `ReconstructionPipeline`: detection dataset, calibration globale, inversion sample-wise, apprentissage des distributions, generation, validation;
- `ReconstructionGenerator`: genere depuis des distributions de parametres apprises;
- `GenerationResult`: contient `X`, concentrations, path lengths, baselines, wavelengths, noise, shifts, environmental params.

C'est la brique la plus proche d'un espace latent commun: concentrations, baselines, path length, instrument params, environnement et preprocessing sont explicites.

Mais elle reste separee du workflow public:

- pas exposee via `nirs4all.generate`;
- pas connectee au `PriorSampler`;
- pas utilisee pour entrainer un encodeur;
- pas encore validee comme source canonique de latents pour toutes les familles de donnees;
- les composants de domaine sont filtres mais restent heuristiques;
- l'inversion physique sur donnees reelles est difficile et doit etre evaluee domaine par domaine.

`RealDataFitter` est aussi important:

- calcule des proprietes spectrales;
- infere instrument, domaine, measurement mode, effets environnementaux, scattering, edge artifacts et preprocessing;
- retourne `FittedParameters`;
- fournit `to_generator_kwargs()` et `to_full_config()`;
- evalue similarite et recommandations.

Le probleme est que `to_generator_kwargs()` ne retourne que range/step/complexity, et les APIs courantes l'utilisent surtout a ce niveau. `to_full_config()` est plus riche mais n'est pas encore un contrat executable de bout en bout.

### Validation et benchmarks

`validation.py` contient:

- validation de matrices X/C/E/wavelengths;
- scorecard de realisme spectral;
- correlation length;
- statistiques de derivees;
- densite de pics;
- baseline curvature;
- SNR;
- overlap de distributions;
- AUC adversariale real vs synthetic;
- `validate_against_benchmark`;
- `quick_realism_check`.

`benchmarks.py` contient un registry de datasets NIR classiques:

- corn;
- tecator;
- shootout2002;
- wheat_kernels;
- diesel;
- tablet_api;
- milk;
- olive_oil.

Forces:

- les bons types de metriques existent;
- l'AUC adversariale donne un test simple et fort de gap synthetique/reel;
- TSTR/TRTS existent conceptuellement dans `validate_against_benchmark`;
- les benchmark metadata sont utiles pour definir des cibles de prior.

Limites:

- il manque une suite de validation obligatoire pour tout changement de prior;
- les thresholds ne sont pas calibres par domaine/instrument;
- il manque des rapports automatiques qui comparent realisme, TSTR/RTSR, ablations et transfert cross-instrument;
- les datasets reels ne sont pas integres automatiquement pour raisons de licence, donc le protocole doit etre explicite.

### Bench et notebooks

`bench/synthetic` est un espace historique et experimental.

Contenu principal:

- `bench/synthetic/__init__.py`: deprecate `bench.synthetic` et re-exporte `nirs4all.synthesis`;
- `bench/synthetic/generator.py`: ancienne implementation autonome, utile historiquement mais pas source de verite;
- `bench/synthetic/S1_synthetic_generation.py`: exemple complet de generation, visualisation, comparaison, pipeline;
- `bench/synthetic/comparator.py`: comparator real/synthetic avec proprietes statistiques, PCA, bruit, pics;
- `bench/synthetic/visualizer.py`: plots de spectres, composants, concentrations, batch effects, bruit;
- `dataset_category_identifier.py`: experiment de classification de domaine/categorie a partir de bandes/composants;
- notebooks: explorations de synthesis, improvements, physical forward generator, reconstruction workflow, fitting hierarchique;
- `synthesis_summary.md`: specification en 25 etapes d'une chaine ideale de generation.

Valeur:

- c'est utile pour comprendre les intentions et les experimentations;
- plusieurs idees ont ete migrees vers `nirs4all.synthesis`;
- `synthesis_summary.md` est une checklist utile pour designer une config complete.

Risque:

- plusieurs fichiers sont obsoletes ou divergents;
- les exemples utilisent parfois l'ancien import `bench.synthetic`;
- les notebooks contiennent des prototypes qui ne sont pas encore API;
- il faut eviter d'avoir deux sources de verite.

## Analyse par objectif

### 1. Augmentation de X seul avec les operateurs de synthese

#### Ce qu'il faudrait idealement

Pour augmenter `X` seul de facon fiable, il faut un catalogue explicite d'operateurs label-preserving. Chaque operateur devrait declarer:

- s'il preserve `y`;
- s'il requiert une grille wavelength;
- s'il est plausible en absorbance, reflectance ou les deux;
- ses domaines d'application;
- ses plages de parametres realistes;
- s'il est sample-level, batch-level, instrument-level ou dataset-level;
- les invariances qu'il doit enseigner au modele.

Il faudrait aussi des presets:

- "robustesse instrument";
- "robustesse baseline";
- "robustesse sample presentation";
- "handheld noisy";
- "benchtop low noise";
- "powder reflectance";
- "liquid transmittance";
- "edge artifacts".

Chaque preset devrait etre valide par scorecard sur au moins un domaine reel.

#### Ce qui existe

Le socle existe deja:

- operateurs spectraux classiques;
- operateurs synthesis-derived;
- operateurs environnement/scattering/edge;
- support wavelength via `SpectraTransformerMixin`;
- integration `sample_augmentation`;
- integration `feature_augmentation`;
- variation_scope sample/batch pour plusieurs operateurs;
- anti-leakage via `origin`;
- targets des samples augmentes mappees vers l'origine.

C'est deja suffisant pour une augmentation X-seul prudente:

- bruit additif/multiplicatif;
- baseline drift;
- shift/stretch modere;
- path length;
- batch effect;
- instrumental broadening;
- heteroscedastic noise;
- particle size/EMSC modere;
- edge artifacts faibles.

#### Ce qui reste a faire

Priorite haute:

- separer officiellement les augmentations label-preserving des augmentations target-changing;
- corriger ou isoler `MixupAugmenter` et `LocalMixupAugmenter`: soit ils gerent `y`, soit ils ne doivent pas etre recommandes en supervised sample augmentation;
- creer des presets physiques documentes;
- ajouter des tests de non-regression par operateur: shape, finite, wavelength constraints, intensite plausible, pas de NaN;
- ajouter un validateur de pipeline qui alerte si un operateur target-changing est utilise dans un contexte qui conserve `y`.

Priorite moyenne:

- calibrer les plages de parametres sur des datasets reels;
- ajouter des recettes par domaine/instrument;
- enregistrer les parametres d'augmentation dans les metadata pour audit;
- ajouter un rapport "augmentation realism" avant/apres.

### 2. Generation de datasets X/Y synthetiques

#### Ce qu'il faudrait idealement

Un dataset X/Y synthetique devrait etre genere par une hierarchie claire:

```
domain
  -> product/matrix
  -> instrument/mode
  -> latent composition C
  -> nuisance params Z
  -> spectra X
  -> analyte/target function f(C, Z, metadata)
  -> reference method noise / censoring / missingness
  -> splits/groups/batches
```

Il ne faut pas seulement generer `X` et prendre une colonne de `C`. Pour un pretraining utile, il faut varier:

- regression mono-target;
- regression multi-target;
- classification;
- thresholds;
- nonlinearites;
- confounders;
- reference method error;
- analytes visibles vs peu visibles spectralement;
- group splits;
- batches;
- instruments;
- sample size;
- wavelength count;
- signal type;
- preprocessing deja applique ou non.

#### Ce qui existe

Le builder couvre deja une grande partie:

- concentrations;
- targets par composant;
- scaling de target;
- transforms log/sqrt;
- classification;
- non-linear targets;
- hidden factors;
- confounders;
- regimes;
- heteroscedasticite;
- metadata;
- partitions;
- batch effects;
- products et categories;
- agregats;
- multi-source;
- export.

Les produits et agregats sont les briques les plus importantes pour depasser le Dirichlet generique.

#### Ce qui reste a faire

Priorite haute:

- creer un adaptateur `prior_to_builder_config()` ou equivalent;
- aligner `PriorSampler` avec les noms de domaines existants;
- faire echouer explicitement tout prior qui choisit un composant absent;
- exposer environmental/scattering/edge configs dans le builder;
- faire utiliser `FittedParameters.to_full_config()` dans `fit_to()` et `from_template()`;
- ajouter un format de metadata de generation standard qui inclut latents, domain, product, instrument, mode, matrix, target function.

Priorite moyenne:

- ajouter reference method noise, LOD/censoring et missing targets;
- ajouter des splits par groupe, batch, instrument et domaine;
- ajouter des taches "calibration transfer": train instrument A, test instrument B;
- ajouter des priors de taille dataset et ratio train/test.

### 3. Denoiser/compresseur vers un espace latent commun

#### Ce qu'il faudrait idealement

L'objectif "latent commun et uniforme quel que soit machine, type de spectre, analyte" exige plus qu'un generateur de spectres. Il faut un protocole d'apprentissage.

Le schema ideal:

```
latent canonique L
  = composition, bands, matrix, path length, baseline, scatter, env, target

views V_j
  = instrument_j + mode_j + preprocessing_j + noise_j + wavelength_grid_j

X_j = forward(L, V_j)

encoder e(X_j, wavelength_grid_j, metadata_j) -> z
decoder/head optional

loss:
  - z same latent close across views
  - z different latent separated
  - reconstruction/canonical spectrum loss
  - analyte prediction auxiliary loss
  - nuisance invariance loss
```

Il faut generer des paires ou groupes multi-vues: meme composition et meme cible, mais instruments, bruits, baselines, scatter et grilles differents. C'est le coeur d'un denoiser/compresseur.

#### Ce qui existe

Les briques existent:

- `SyntheticNIRSGenerator.generate_from_concentrations()` permet de reutiliser un `C`;
- les instruments/detecteurs/grilles existent;
- les augmentations peuvent creer des vues differentes;
- la reconstruction forward chain a une notion de modele canonique;
- `ReconstructionPipeline` apprend des distributions de parametres depuis du reel;
- les docs NIRS-PFN recommandent deja une approche encodeur spectral avant full PFN.

#### Ce qui manque

Il manque le workflow encodeur:

- pas d'API pour generer `k` vues d'un meme latent;
- pas de `LatentSample` ou `CanonicalSample`;
- pas de dataloader contrastif;
- pas d'operateur `NIRSEncoder` implemente dans le package principal;
- pas de losses ni de training loop;
- pas de benchmark d'invariance cross-instrument;
- pas de validation "same latent different machine -> embedding proche";
- pas de protocole pour savoir si l'espace latent garde l'information analyte tout en retirant les nuisances.

Le module reconstruction est prometteur mais doit etre stabilise avant de servir de verite latente. Une inversion physique mal posee peut apprendre des latents jolis mais non identifiables.

### 4. Donnees synthetiques realistes pour preentrainer des modeles

#### Ce qu'il faudrait idealement

Pour du pretraining, il faut une distribution synthetique large mais controlee:

- nombreux domaines;
- nombreux instruments;
- nombreux modes de mesure;
- composants predefinis et proceduraux;
- produits/agregats realistes;
- matrices liquides, poudres, solides, tissus, emulsions;
- signal brut et preprocessings deja appliques;
- tailles variables;
- grilles variables;
- bruit et artefacts calibres;
- labels varies;
- metadata riches;
- ablations.

Il faut aussi un protocole de validation:

- scorecard real/synthetic;
- AUC adversariale;
- TSTR;
- RTSR ou pretrain puis finetune;
- evaluation few-shot;
- transfert cross-instrument;
- ablation par famille d'effets;
- comparaison aux pipelines standards.

#### Ce qui existe

Les pieces sont nombreuses:

- 126 composants;
- procedural component generation;
- 20 domaines;
- 69 agregats;
- 18 produits;
- 19 instruments;
- detector models;
- environmental/scattering/edge effects;
- multi-source;
- targets complexes;
- benchmark registry;
- scorecard validation;
- accelerated generator.

#### Ce qui manque

Le pretraining demande surtout une orchestration:

- un sampler de batches de pretraining qui combine toutes les briques;
- une API qui peut produire un flux infini ou de gros shards;
- des seeds et metadata pour tracer chaque sample;
- une validation automatique de la distribution synthetique;
- une separation claire entre prior large pour pretraining et prior cible pour un domaine;
- une integration GPU/accelerated dans le workflow public;
- un monitoring de diversite: PCA, derivative stats, component coverage, domain coverage, instrument coverage.

Le risque principal n'est pas de ne pas avoir assez de spectres. Le risque est de preentrainer sur une distribution large mais mal calibree, qui apprend des invariances fausses ou des artefacts du generateur.

### 5. Priors pour NIRS-ICL ou NIRS-PFN

#### Ce qu'il faudrait idealement

Un prior NIRS-ICL/PFN ne doit pas seulement produire des spectres. Il doit produire des taches.

Objet minimal propose:

```python
@dataclass
class NIRSPriorTask:
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray
    wavelengths: np.ndarray | list[np.ndarray]
    domain: str
    instrument: str | list[str]
    measurement_mode: str
    target_name: str
    target_type: str
    metadata_context: dict
    metadata_query: dict
    latent_params: dict
    prior_config: dict
```

Le sampler de taches devrait varier:

- nombre de samples contexte/query;
- nombre de wavelengths;
- range spectral;
- instrument;
- domaine;
- analyte;
- relation X->y;
- bruit de reference;
- batches/groupes;
- preprocessing;
- covariate shift entre context et query;
- distribution shift instrument ou domaine;
- proportion d'outliers;
- labels manquants ou cibles censurees.

Pour NIRS-PFN, le prior doit etre hierarchique. Le modele apprend a resoudre une tache conditionne sur un petit contexte, donc les variations entre taches sont aussi importantes que les variations entre samples.

#### Ce qui existe

Il existe une premiere version conceptuelle:

- `NIRSPriorConfig`;
- `PriorSampler`;
- domaine -> instrument -> mode -> matrix -> components -> target type;
- docs internes NIRS-PFN;
- plan encodeur spectral;
- benchmarks et validation.

#### Ce qui manque

Priorite tres haute:

- corriger les noms de domaines;
- creer `sample_dataset()` et `sample_task()`;
- convertir les configs en vrais appels builder/generator;
- inclure les latents et metadata;
- standardiser context/query;
- ajouter un mode "encoder contrastive pairs";
- ajouter un mode "PFN supervised tasks";
- ajouter des tests de prior predictive checks.

Priorite haute:

- supporter plusieurs grilles wavelength dans une meme meta-distribution;
- ajouter positional encoding ou resampling policy pour les modeles;
- representer explicitement les nuisances: instrument, batch, temperature, scatter;
- generer des taches de transfert: contexte sur instrument A, query sur instrument B;
- generer des taches "same analyte across domains" et "same domain different analyte".

Priorite moyenne:

- calibrer les poids de prior par frequence reelle;
- ajouter des priors de reference method noise par analyte;
- ajouter une API de curriculum: simple -> realistic -> complex -> real-fitted.

## Ecarts techniques critiques

### Measurement mode non cable

Le code annonce transmittance, reflectance, transflectance et ATR. Le module existe. Mais le generateur principal n'applique pas le simulateur de mode de mesure dans `generate()`. Pour un prior credible, c'est un blocker scientifique.

Action:

- remplacer la creation hard-codee `create_transmittance_simulator()` par une factory selon `measurement_mode`;
- appliquer le transform au bon endroit de la chaine;
- definir clairement l'espace: absorption coefficient, absorbance apparente, reflectance brute, transmittance brute;
- tester transmittance vs reflectance vs ATR sur les memes latents.

### PriorSampler deconnecte

Le prior sample une config, pas une tache. Il faut un adaptateur executable.

Action:

- normaliser les noms domaines;
- valider composants;
- mapper instrument/mode/matrix/noise/env/scatter vers generator/builder;
- generer `SpectroDataset`;
- generer `NIRSPriorTask`.

### Builder incomplet pour effets avances

Le generateur accepte environmental/scattering/edge configs; le builder ne les expose pas vraiment.

Action:

- ajouter `with_environmental_effects(...)`;
- ajouter `with_scattering_effects(...)`;
- ajouter `with_edge_artifacts(...)`;
- ajouter `with_measurement_mode(...)` separe si necessaire;
- s'assurer que `fit_to()` peut les remplir.

### from_template trop superficiel

`RealDataFitter` calcule beaucoup d'information. Le builder en utilise peu.

Action:

- faire de `FittedParameters.to_full_config()` un contrat executable;
- ajouter tests "fit real-like synthetic -> regenerate -> scorecard proche";
- exposer une version explicite `generate.from_real_fit(...)` avec rapport de fit.

### Augmentation supervisee target-changing

`sample_augmentation` preserve `y` par design. C'est bon pour les nuisances. C'est faux pour mixup supervise.

Action:

- tagger les operateurs `label_preserving=True/False`;
- interdire ou avertir si `False` dans sample augmentation supervisee;
- si mixup doit rester, creer une interface qui retourne aussi les nouveaux `y`.

### Latent commun absent

La reconstruction contient les concepts, mais pas l'apprentissage du latent.

Action:

- definir `CanonicalLatent`;
- definir `generate_views(latent, view_configs)`;
- definir dataloader contrastif;
- definir metriques d'invariance et preservation analyte.

## Proposition d'architecture cible

### Couche 1: prior de configuration

`PriorSampler` doit produire une configuration valide, mais pas directement `X`.

Sortie:

- domaine canonical;
- produit/agregat ou procedural;
- instrument;
- mode;
- matrix;
- component set;
- concentration prior;
- nuisance prior;
- target prior;
- split/task prior.

### Couche 2: latent sampler

Nouveau composant:

```python
LatentBatch = sample_latents(prior_config, n_samples)
```

Contenu:

- concentrations;
- component spectra/bands;
- path length;
- baseline coefficients;
- scatter params;
- temperature/moisture;
- particle params;
- batch ids;
- sample groups;
- target clean;
- target noisy;
- metadata.

### Couche 3: forward views

Nouveau composant:

```python
X_view = render_view(LatentBatch, instrument, mode, preprocessing, wavelength_grid)
```

Cela rend possible:

- plusieurs instruments pour le meme latent;
- augmentation controllable;
- generation de paires contrastives;
- generation de context/query avec shift controle.

### Couche 4: dataset/task adapters

Sorties:

- `SpectroDataset` classique;
- `(X, y)` arrays;
- `NIRSPriorTask`;
- shards pretraining;
- paires encodeur.

### Couche 5: validation

Chaque prior ou preset doit avoir:

- `quick_realism_check`;
- scorecard vs real si donnees disponibles;
- AUC adversariale;
- TSTR/RTSR;
- ablation report;
- coverage report.

## Plan de travail propose

### Phase 0: corriger les incoherences bloquantes

Objectif: rendre les sorties scientifiquement coherentes avant d'empiler du pretraining.

Travaux:

- cabler `measurement_mode`;
- aligner noms de domaines prior/domain configs;
- valider composants echantillonnes;
- exposer environmental/scattering/edge dans builder;
- tagger augmentations label-preserving;
- corriger ou isoler mixup;
- enrichir `fit_to()` avec `to_full_config()`.

Livrable:

- tests unitaires et integration;
- mini rapport "same prior sample builds dataset".

### Phase 1: stabiliser generation X/Y

Objectif: produire des datasets synthetiques credibles pour regression/classification.

Travaux:

- `prior_to_builder_config`;
- `generate.prior(...)`;
- reference method noise;
- splits par groupes/batches;
- metadata standard;
- scorecard automatique.

Livrable:

- 10 presets: grain, dairy, meat, tablet, fuel, soil, fruit, textile, handheld, benchtop.

### Phase 2: generateur multi-vues pour encodeur

Objectif: entrainer un encodeur/denoiser spectral.

Travaux:

- `CanonicalLatent`;
- `generate_views`;
- paires positives/negatives;
- dataloader contrastif;
- evaluation invariance.

Livrable:

- benchmark encodeur vs ASLSBaseline+PCA+TabPFN sur datasets reels.

### Phase 3: prior de taches NIRS-ICL/PFN

Objectif: produire des episodes de meta-learning.

Travaux:

- `NIRSPriorTask`;
- context/query sampler;
- variable sample sizes;
- variable wavelength grids;
- shifts instrument/domaine;
- target functions variees.

Livrable:

- baseline ICL avec encodeur + TabPFN;
- prototype NIRS-PFN uniquement si l'encodeur montre un plafond.

### Phase 4: validation continue

Objectif: eviter que le generateur devienne plausible visuellement mais inutile en transfert reel.

Travaux:

- suite benchmark;
- AUC adversariale par domaine;
- TSTR/RTSR;
- pretrain/fine-tune curves;
- ablations;
- regression reports.

Livrable:

- tableau de bord realisme et transfert.

## Tableau recapitulatif

| Objectif | Ideal | Existant | Ecart principal | Priorite |
|---|---|---|---|---|
| Augmenter X seul | Operateurs label-preserving calibres, presets par domaine/instrument, validation avant/apres | Nombreux operateurs, pipeline `sample_augmentation`, `feature_augmentation`, anti-leakage par `origin` | Pas de contrat label-preserving, mixup ne gere pas `y`, presets physiques a formaliser | Haute |
| Generer X/Y dataset | DAG domaine -> latent -> X -> y avec reference noise, metadata, splits, tasks | `SyntheticDatasetBuilder`, products, aggregates, targets complexes, multi-source, export | Prior non connecte au builder, `from_template` shallow, effets avances peu exposes | Haute |
| Denoiser/latent commun | Latents canoniques, multi-vues meme sample, encodeur invariant, losses contrastives/reconstruction | Reconstruction forward chain, generate_from_concentrations, instruments, augmentations | Pas de `CanonicalLatent`, pas de generateur de vues, pas de training encodeur | Haute |
| Pretraining synthetique realiste | Distribution large calibree, shards/batches, validation real/synth, ablations | 126 composants, procedural, 20 domaines, 69 agregats, 18 produits, 19 instruments, scorecard | Orchestration manquante, calibration insuffisante, accelerated pas integre a l'API de prior | Moyenne-haute |
| Priors NIRS-ICL/PFN | Sampler de taches context/query avec hierarchie domaine/instrument/analyte/split | `PriorSampler`, docs NIRS-PFN, builder/generator sous-jacents | Le prior retourne une config, pas une tache; noms domaines incoherents; pas de task object | Tres haute |
| Realisme scientifique | Scorecard, AUC adversariale, TSTR/RTSR, thresholds par domaine | Validation module, benchmark registry, comparator historique | Pas encore gate obligatoire ni thresholds calibres | Haute |
| API publique | `generate.prior`, `generate.task`, `generate.encoder_pairs`, rapports | `generate`, `.regression`, `.classification`, `.product`, `.category`, `.from_template` | API simple utile mais pas orientee prior/pretraining | Moyenne |

## Discussion

### Points forts du projet

Le plus gros point fort est que nirs4all possede deja les briques qui manquent souvent dans les generateurs synthetiques spectraux:

- composants chimiques et bandes;
- logique en wavenumber;
- produits/agregats plausibles;
- instruments et detecteurs;
- effets physiques et chemometriques;
- integration dataset/pipeline;
- validation adversariale et scorecard;
- reconstruction latente experimentale.

Le second point fort est l'integration avec `SpectroDataset`. Cela signifie que les donnees synthetiques peuvent immediatement passer dans les pipelines existants, avec partitions, processings, augmentation, predictions et evaluation.

Le troisieme point fort est la presence d'une pensee "foundation model" deja documentee. Les docs NIRS-PFN identifient correctement que TabPFN fonctionne deja bien sur NIRS apres features, et que le goulot court terme est probablement l'extraction/representation spectrale, pas forcement un PFN complet.

### Points faibles et risques

Le premier risque est l'incoherence entre modules. Beaucoup de choses sont implementees, mais certaines ne sont pas connectees:

- prior non executable;
- mode de mesure non applique;
- fitter riche mais builder shallow;
- reconstruction separee;
- accelerated separe;
- bench historique partiellement obsolete.

Le deuxieme risque est scientifique: un generateur peut produire des courbes qui ressemblent a des spectres mais apprennent au modele de mauvaises invariances. Par exemple, si reflectance est traitee comme une absorbance avec un peu de bruit, le modele peut echouer sur de vrais poudres ou solides.

Le troisieme risque concerne les cibles. Pour NIRS-ICL/PFN, `y` doit varier de facon realiste: bruit de methode de reference, analytes difficiles, confounders, regimes, censure, multi-output. Si les cibles restent trop directement liees a une colonne de concentration, le modele apprendra un monde trop facile.

Le quatrieme risque est l'identifiabilite du latent. Un espace latent commun n'est utile que si les variables latentes sont stables et verifiables. L'inversion physique est prometteuse, mais les spectres NIR sont souvent mal poses: plusieurs compositions et nuisances peuvent expliquer une forme similaire.

### Faisabilite

Faisabilite par objectif:

- augmentation X seul: tres faisable maintenant. Il faut surtout formaliser les contrats et presets.
- generation X/Y: faisable a court terme. Les briques existent; il faut connecter prior, builder et validation.
- pretraining synthetique: faisable a moyen terme. Le risque principal est la calibration real/synthetic, pas la generation de volume.
- denoiser/compresseur latent: faisable mais plus recherche. Il faut generer des vues multi-instruments et definir des pertes.
- NIRS-ICL/PFN complet: faisable a long terme, mais probablement pas la premiere etape optimale. L'approche encodeur spectral + TabPFN est plus pragmatique pour valider la valeur de la synthese.

Mon avis technique: il ne faut pas commencer par entrainer un gros NIRS-PFN. Il faut d'abord prouver que le generateur synthetique ameliore une representation spectrale. Si un encodeur entraine sur les priors synthetiques bat ASLSBaseline+PCA+TabPFN ou ameliore le few-shot, alors la distribution synthetique a une valeur reelle. Ensuite seulement, un PFN spectral devient justifiable.

### Comment valider tout ca

Validation minimale pour augmentation X seul:

- tests shape/finite/reproductibilite;
- labels conserves uniquement pour operateurs declares label-preserving;
- verification anti-leakage train/test;
- comparaison des distributions avant/apres: SNR, derivatives, baseline curvature.

Validation minimale pour generation X/Y:

- prior predictive checks: composants valides, concentrations dans ranges, targets coherentes;
- scorecard real/synthetic par domaine;
- AUC adversariale ciblee sous 0.7 au debut, puis plus stricte;
- TSTR: train sur synthetique, test sur reel;
- RTSR/pretrain: pretrain synthetique, finetune reel, comparer a real-only;
- ablations: sans instrument, sans scattering, sans env, sans products, sans procedural.

Validation minimale pour latent/denoiser:

- meme latent rendu par plusieurs instruments doit avoir embeddings proches;
- latents differents doivent etre separes;
- l'embedding doit conserver l'information analyte;
- l'embedding doit reduire instrument/batch predictability;
- evaluation downstream avec TabPFN, PLS, RF sur donnees reelles;
- courbes few-shot.

Validation minimale pour NIRS-ICL/PFN:

- episodes synthetiques avec context/query;
- generalisation a domaines/instruments non vus;
- comparaison a TabPFN sur PCA et encodeur;
- evaluation sur benchmarks reels;
- mesure de calibration/incertitude si le PFN produit des distributions;
- tests de robustesse aux grilles wavelength variables.

Le critere final n'est pas "le synthetique ressemble visuellement au reel". Le critere final est:

1. les metriques real/synthetic ne detectent pas un gap grossier;
2. un modele entraine ou preentraine sur synthetique s'ameliore sur reel;
3. les gains tiennent en few-shot et cross-instrument;
4. les ablations montrent quelles familles de prior apportent le gain.

## Recommandation finale

La direction est bonne et l'existant est suffisant pour construire une vraie ligne de recherche. Le projet a une forte chance de produire quelque chose d'utile si l'effort se concentre maintenant sur l'unification et la validation.

Ordre recommande:

1. corriger les incoherences de prior, measurement mode, builder et augmentation supervisee;
2. livrer `generate.prior(...)` et une validation real/synthetic reproductible;
3. construire le generateur multi-vues et l'encodeur spectral;
4. valider encodeur + TabPFN contre ASLSBaseline + PCA + TabPFN;
5. seulement ensuite, formaliser une tache NIRS-PFN complete.

Le meilleur usage immediat de la synthese dans nirs4all n'est pas encore "entrainer un modele universel de bout en bout". C'est d'abord apprendre une representation spectrale robuste, puis utiliser cette representation dans des modeles qui marchent deja bien. Cette strategie reduit le risque, donne des validations plus rapides et force le prior synthetique a prouver sa valeur sur des donnees reelles.
