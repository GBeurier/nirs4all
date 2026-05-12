# Synthese de spectres NIRS synthetiques - objectifs de recherche

## Positionnement

Le projet vise a etablir quand, comment et jusqu'ou des spectres NIRS
synthetiques peuvent ameliorer des modeles de prediction, des representations
latentes et des architectures de type prior-data fitted networks, sans creer une
illusion de performance due a un generateur trop pauvre, trop proche des donnees
reelles, ou mal valide.

La question centrale n'est donc pas seulement de "generer des spectres
realistes". La question de recherche est:

> Peut-on construire des generateurs de spectres NIRS et de taches synthetiques
> qui capturent les invariances instrumentales, physiques, statistiques et
> target-dependantes utiles aux modeles, tout en restant suffisamment controles
> pour que les gains mesures sur donnees reelles soient interpretables?

Cette formulation impose de distinguer quatre niveaux:

1. Generer seulement `X`: spectres plausibles ou indiscernables de spectres
   reels.
2. Generer des vues de `X`: memes latents observes sous plusieurs instruments,
   modes de mesure, baselines, bruits et grilles spectrales.
3. Generer `(X, Y)`: relations cible-spectre credibles, utiles pour entrainer ou
   augmenter des modeles supervises.
4. Generer des taches completes: distributions de datasets, context/query sets,
   targets et regimes assez divers pour entrainer une architecture ICL/PFN.

## Objectifs scientifiques

### O1 - Data augmentation spectrale `X -> X'`

Objectif: transformer des spectres reels en variantes plausibles gardant le meme
`Y`, afin d'ameliorer la robustesse des modeles supervises.

La priorite n'est pas d'inventer des signaux ex nihilo mais de simuler les
facteurs de nuisance reels: bruit instrumental, baseline, scatter, humidite,
temperature, broyage, resolution, shift de longueur d'onde, capteurs et effets
de lot. Les augmentations doivent rester petites et mesurables: elles doivent
ameliorer la generalisation sans detruire l'information cible.

### O2 - Generation `X` ex nihilo pour espace latent

Objectif: produire des spectres non etiquetes en quantite et diversite
suffisantes pour entrainer:

- des auto-encodeurs, VAE, VQ-VAE ou encodeurs masques;
- des embeddings universels compatibles avec des datasets, grilles et
  instruments heterogenes;
- des denoisers, imputers et correcteurs de spectres reels.

Ici, la cible `Y` est exclue. Le critere de succes est la qualite de la
representation apprise et sa transferabilite, pas la beaute visuelle des
spectres.

### O3 - Generation `(X, Y)` pour robustesse predictive

Objectif: generer des paires spectre-cible credibles pour etendre la plage de
prediction, en particulier dans les zones ou les modeles produisent des
sigmoides et predisent mal les faibles ou fortes valeurs.

Ce niveau est plus difficile que O1/O2: une bonne distribution de `X` ne suffit
pas. Il faut modeliser la relation cible-composition-spectre, les confondeurs,
les regimes, l'heteroscedasticite et les zones d'information spectrale
partiellement masquees. La validation doit inclure les marges de `Y`, la
distribution jointe `(X, Y)`, TSTR/TRTS et les performances dans les queues de
distribution.

### O4 - Priors NIRS-PFN / ICL

Objectif: generer des taches completes, pas seulement des echantillons, pour
entrainer ou tester des architectures capables d'apprendre en contexte sur des
spectres NIRS.

Le generateur doit produire des familles de datasets avec:

- des axes spectraux variables;
- des tailles d'echantillons petites a moyennes;
- des targets de regression et classification;
- des regimes de bruit et de confounding;
- des distributions de complexite comparables aux datasets reels.

Le succes se mesure sur des taches reelles tenues a l'ecart, avec comparaison a
TabPFN, PLS/Ridge, AOM et autres baselines deja presentes dans `bench`.

## Etat de l'art operationnel

### Augmentations chemometriques

La pratique courante en NIRS repose sur les transformations de spectres: SNV,
MSC/EMSC, Savitzky-Golay, derivees, detrending, correction de baseline, filtres
de bruit et selection de bandes. Leur version generative consiste a echantillonner
les nuisances correspondantes au lieu de les retirer deterministiquement.

Avantage: robuste, interpretable, peu couteux. Limite: ces augmentations
modelisent surtout des nuisances locales; elles ne creent pas une distribution
de spectres ou de taches complete.

### Modeles mecanistes

Les generateurs physiques utilisent des melanges de composants, des bandes
NIR, Beer-Lambert, Kubelka-Munk, des profils Gauss/Voigt/Lorentz, des modes de
mesure, des detecteurs et des effets instrumentaux.

Avantage: controle scientifique et extrapolation conceptuelle. Limite: les
spectres reels dependent souvent de metadata manquantes: geometrie optique,
pathlength, granulometrie, temperature, humidite, protocole d'acquisition,
pretraitements deja appliques.

### Modeles statistiques

PCA, modeles de covariance, GMM, copules, bootstrap controle, kNN/mixup et
bruits par canal permettent de capturer la distribution empirique de `X`.

Avantage: forte efficacite pour l'indiscernabilite real/synthetic. Limite: risque
de quasi-copie, de fuite de donnees, et d'apprentissage d'un manifold local qui
ne generalise pas a de nouveaux instruments ou matrices.

### Modeles generatifs appris

VAE, GAN, diffusion, transformers ou auto-encodeurs masques peuvent apprendre
des distributions complexes de spectres. Ils sont attractifs pour l'espace
latent et la generation globale.

Avantage: flexibilite. Limite: besoin de donnees, risque d'artefacts
detectables, faible interpretabilite, et validation plus difficile que pour un
hybride mecaniste-statistique.

### Self-supervised learning et fondations tabulaires

Les encodeurs masques, contrastifs ou Perceiver-like peuvent apprendre une
representation spectrale sans labels. TabPFN/ICL suggere qu'un bon prior de
taches peut remplacer une partie de l'optimisation supervisee, mais seulement si
la distribution de taches synthetiques couvre les problemes reels.

## Choix rationnels

### Bench avant librairie

Tout ce qui est incertain reste dans `bench/synthesis`. `nirs4all/synthesis`
sert de moteur stable et de reference d'integration, mais les generateurs,
gates et rapports exploratoires ne doivent etre portes dans la librairie qu'une
fois leur valeur demontree.

### Separateur strict entre `X`, `(X, Y)` et taches

Les echecs passes viennent en partie du melange des objectifs. Un generateur qui
passe un test de realisme `X` ne valide pas automatiquement une generation
`(X, Y)`. Un generateur `(X, Y)` utile a un dataset ne suffit pas pour entrainer
un PFN. Chaque niveau a ses gates.

### Hybride mecaniste-statistique comme trajectoire principale

Un generateur purement mecaniste est scientifiquement propre, mais il bloque vite
quand les donnees reelles ne fournissent pas la geometrie et les conditions
d'acquisition. Un generateur purement statistique peut etre performant mais trop
local. La voie principale doit donc combiner:

- squelette mecaniste explicite;
- residus statistiques controles;
- validations adversariales;
- tests downstream independants.

### L'adversarial AUC est necessaire mais insuffisant

Pour `X`, un discriminant real/synthetic proche de 0.5 est un bon test de
distribution. Mais il ne doit pas devenir le seul oracle: il faut detecter les
quasi-duplicats, tester plusieurs discriminateurs, mesurer la diversite, et
verifier que les modeles entraines avec le synthetique progressent sur du reel.

### Priorite aux cas d'usage mesurables

Les objectifs doivent rester lies a des decisions experimentales:

- augmentation utile ou non;
- encodeur latent utile ou non;
- generation `(X, Y)` utile ou non;
- prior PFN plausible ou non.

## Verrous scientifiques

1. Heterogeneite des grilles spectrales: longueurs, pas, unite, zones
   disponibles et instruments varient fortement.
2. Ambiguite du type de signal: absorbance, reflectance, derivees et signaux
   pretraites ne sont pas toujours identifies.
3. Metadata physiques manquantes: pathlength, geometrie, capteur, temperature,
   humidite, preparation echantillon.
4. Small-N high-P: beaucoup de datasets ont peu d'echantillons et beaucoup de
   longueurs d'onde, ce qui rend les discriminateurs et generateurs instables.
5. Fuite de donnees: les methodes bootstrap/kNN/mixup peuvent devenir des copies
   deguisees si elles ne sont pas surveillees.
6. Relation `Y` non identifiable: la cible peut dependre de variables non vues,
   de confondeurs, de regimes ou de choix experimentaux non spectraux.
7. Validation multi-niveau: realisme visuel, AUC adversarial, TSTR, TRTS,
   ablations et gain predictif ne disent pas exactement la meme chose.
8. Generalisation hors domaine: un generateur adapte a un dataset peut echouer
   sur une autre matrice ou un autre instrument.

## Hypotheses de travail

- H1: Les augmentations `X -> X'` ameliorent surtout la robustesse locale et les
  petits datasets, a condition d'etre calibrees par matrice et instrument.
- H2: Un generateur hybride mecaniste-statistique peut atteindre un realisme `X`
  suffisant sur un panel large plus rapidement qu'un pur generateur physique.
- H3: Les gains downstream de la generation `(X, Y)` apparaitront d'abord sur
  les datasets ou la relation `X -> Y` est deja bien capturee par un modele
  simple et ou la distribution de `Y` est mal couverte aux extremes.
- H4: Un encodeur universel est plus realiste a court terme qu'un PFN complet:
  il peut exploiter beaucoup de `X` sans resoudre immediatement la generation de
  taches `Y`.
- H5: Un PFN NIRS n'est pertinent qu'apres avoir etabli des task samplers dont
  la difficulte, la taille, la structure spectrale et les regimes de target
  ressemblent aux benchmarks reels.

## Livrables attendus

1. Un atlas des datasets reels: grilles, type de signal, taille, matrice,
   target, metadata disponibles et qualite.
2. Une bibliotheque d'augmentations spectrales testees et classees par usage.
3. Un generateur `X` hybride avec gates anti-fuite et rapports par dataset.
4. Un generateur `(X, Y)` avec validation jointe et stress tests sur les
   extremes de target.
5. Un protocole d'entrainement d'encodeur latent avec evaluation downstream.
6. Un prototype de task sampler pour NIRS-ICL/PFN.
7. Des criteres explicites de promotion vers `nirs4all/synthesis`.
