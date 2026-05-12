## Datasets

Les datasets (57 regression et 17 classification de mémoire) du papier Tabpfn. (bench/tabpfn_paper/data)

## Data Augmentation

Posséder des functions de bruits réalistes pour pouvoir faire de la data augmentation.
Objectif: Générer des spectres (X) à partir de spectres existants pour améliorer le fit des modèles. On crée des spectres augmentés avec le même Y que les originaux. L'idée est de pouvoir combiner des effets réalistes (machines, humidité, broyage, capteurs, etc.) et donc d'avoir une collection de fonctions, de combinaisons de fonctions et de presets pour correctement ajouter de la diversité spectrale aux datasets.

## Latent Space
Être capable de générer des spectres réalistes ex-nihilo suffisament réalistes et en suffisament grande quantité que je puisse:
- entrainer soit un Auto-Encoder (et variantes VAE, VQ-VAE, etc..) efficace pour les datasets réels,
- entrainer un embedding compatible avec tous les datasets
- entrainer un Denoiser ou faire de l'imputation pour corriger des spectres réels

L'idée ici et que le générateur soit capable de facon réaliste de couvrir tout l'espace des spectres possibles pour permettre d'entrainer des modèles self-supervised dans le but d'améliorer les modèles de prédictions, de corriger les données et d'avoir un espace latent intéressant. Bref, tout le panel possible avec données synthétiques. Encore une fois, ici on ne génère que les X.

## Models Robustness
Pouvoir générer pour un dataset donnée des paires (X, Y) en data augmentation afin d'augmenter la plage de prédictions pour les targets. En gros, j'ai souvent des sigmoides en prédictions avec les valeurs basses et hautes qui sont mal prédites. Un modèle capable de générer des paires X, Y crédibles pour un dataset donné, pourraient permettre d'améliorer la robustesse des modèles de prédiction. Ici on génère X et Y.

## NIRS-PFN priors
Enfin l'idée finale est d'être capable de générer des sets de priors (spectres X et targets Y) complètement synthétique, suffisament réalistes et en quantité et diversité suffisante pour pouvoir entrainer une architecture ICL/PFN. On introduit une génération de Y complexe à partir de spectres complètement synthétiques. L'idée ici et d'avoir des sets de paramètres que l'on peut sampler intelligemment pour générer des priors.


### Techniques:

** Mecaniste **: Générer toutes les transformations depuis le substrat jusqu'au spectres avec des fonctions "mécanistes" paramétrables. Ca signifie aussi pouvoir déterminer les paramètres ou les intervalles de paramètres et la stack de méthodes qui correspondent à des spectres réalistes.

** Statistique **: Extraire les propriétés statistiques des signaux réels pour les reproduire.

** Meca-Stat **: Commencer par une série d'opérateurs mécanistes et finir avec l'ajout de "bruit" statistique pour obtenir des spectres réalistes.

** Apprentissage (ML/DL) **: Entrainer des modèles à reproduire des spectres réalistes, depuis les VAE et GAN aux transformers ou autres.

** Hybride **: Mélanger mécaniste (ou mécaniste surrogation/distillation), statistique et apprentissage pour générer des spectres réalistes.