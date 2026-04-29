J’ai lu les quatre documents : [report.md](sandbox:/mnt/data/report.md), [ROADMAP.md](sandbox:/mnt/data/ROADMAP.md), [PUBLICATION_BACKLOG.md](sandbox:/mnt/data/PUBLICATION_BACKLOG.md), [PUBLICATION_PLAN.md](sandbox:/mnt/data/PUBLICATION_PLAN.md).

Mon diagnostic : le plan actuel est solide pour publier une **AOM-PLS NIPALS/adjoint** propre, mais notre échange sur **SIMPLS** ouvre une version scientifiquement plus ambitieuse : ne plus présenter seulement AOM-PLS comme une méthode, mais comme une **famille “Operator-Adaptive PLS”**, où AOM, POP, hard selection, soft mixture, NIPALS et SIMPLS sont des cas particuliers.

Le point clé : **SIMPLS n’est pas seulement une optimisation de vitesse**. C’est une reformulation conceptuelle qui permet de faire la sélection d’opérateurs directement dans l’espace des covariances (X^\top Y). Pour un projet type “PLS Mania”, c’est très intéressant parce que ça crée une architecture propre, rapide, GPU-friendly, et scientifiquement testable.

---

# 1. Ce que disent les documents, en synthèse

## 1.1 L’état actuel : AOM-PLS NIPALS est déjà une bonne base

Le cœur actuel est :

[
X_b = X A_b^\top
]

où (A_b) est un opérateur linéaire : identité, Savitzky–Golay, dérivée, detrend, Norris–Williams, compositions, etc.

La méthode actuelle évite de matérialiser (X_b) pour chaque opérateur. Elle utilise un **truc d’adjoint** dans NIPALS :

[
c = X^\top y
]

[
g_b = A_b^\top c
]

puis une direction de poids est construite à partir de (A_b) et (A_b^\top). C’est astucieux, rapide et cohérent avec des opérateurs linéaires.

Le document `PUBLICATION_PLAN.md` identifie déjà les forces :

* sélection intégrée du preprocessing ;
* identité toujours dans la banque ;
* garantie de dominance sur le critère de sélection ;
* coût linéaire en nombre d’opérateurs ;
* interprétabilité : on sait quel opérateur a été choisi.

La faiblesse principale est aussi claire : la sélection actuelle repose sur un **holdout interne 20 %** avec graine fixe. C’est reproductible, mais arbitraire et fragile sur petits (n). Le plan prévoit de remplacer ça par PRESS, CV ou hybride.

---

## 1.2 Le résultat empirique actuel : les prototypes sophistiqués ne tuent pas le baseline

Le fichier `report.md` compare :

* Baseline AOM-PLS ;
* Bandit AOM-PLS ;
* DARTS PLS ;
* Zero-Shot Router ;
* MoE PLS.

Sur les cinq datasets testés, le résultat important n’est pas “le baseline gagne tout”. Ce n’est pas le cas. Le résultat important est plutôt :

> Les méthodes plus complexes gagnent parfois, mais elles sont moins stables, plus coûteuses, ou moins propres scientifiquement.

D’après les chiffres du rapport :

| Modèle           | Lecture rapide                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------- |
| Baseline AOM-PLS | robuste, rapide, jamais catastrophique                                                      |
| Bandit AOM-PLS   | utile si banque énorme, mais misranking possible                                            |
| DARTS PLS        | très intéressant, gagne parfois, mais plus lent et biaisé par PLS différentiable simplifiée |
| Zero-Shot Router | coûteux, souvent revient au brut                                                            |
| MoE PLS          | gagne sur certains cas SNV/MSC, mais extrêmement lent et instable                           |

Le point le plus intéressant : **DARTS est théoriquement séduisant**, mais l’amélioration empirique ne justifie pas encore sa complexité. MoE montre que les preprocessings non linéaires peuvent aider, mais c’est un autre régime.

Donc le plan actuel a raison de dire : pour la publication, il faut d’abord solidifier le cadre linéaire, pas partir directement dans du neural / MoE / DARTS.

---

## 1.3 Le backlog actuel a une excellente intuition : unifier AOM et POP

Les documents veulent déjà fusionner :

* AOM = un opérateur pour tout le modèle ;
* POP = un opérateur potentiellement différent par composante.

Dans l’API prévue :

```python
AOMPLSRegressor(
    selection="global",         # AOM classique
    selection="per_component",  # POP classique
    selection_criterion="press",
    max_components=25,
    n_components="auto",
)
```

C’est une très bonne direction.

Mais avec notre discussion SIMPLS, cette unification doit être élargie. Il ne faut plus seulement avoir :

[
\text{AOM vs POP}
]

Il faut avoir :

[
\text{AOM/POP} \times \text{NIPALS/SIMPLS}
]

Autrement dit : le plan actuel est bon, mais il lui manque maintenant un axe fondamental.

---

# 2. Ce que change SIMPLS

## 2.1 Reformulation centrale

Soit :

[
X \in \mathbb{R}^{n \times p}
]

[
Y \in \mathbb{R}^{n \times q}
]

[
S = X^\top Y
]

Soit un opérateur spectral linéaire :

[
A_b \in \mathbb{R}^{p \times p}
]

appliqué aux spectres ligne par ligne :

[
X_b = X A_b^\top
]

Alors la covariance croisée du jeu transformé est :

[
S_b = X_b^\top Y
]

[
S_b = (X A_b^\top)^\top Y
]

[
S_b = A_b X^\top Y
]

Donc :

[
\boxed{
S_b = A_b S
}
]

C’est énorme.

Cela veut dire que pour SIMPLS, tu n’as pas besoin de recalculer une PLS complète sur chaque (X_b). Tu peux appliquer les opérateurs directement à la covariance (S).

Pour PLS1 :

[
S = X^\top y \in \mathbb{R}^{p}
]

Donc chaque opérateur agit sur un simple vecteur :

[
s_b = A_b s
]

Pour PLS2 :

[
S = X^\top Y \in \mathbb{R}^{p \times q}
]

et chaque opérateur agit sur les colonnes de cette matrice :

[
S_b = A_b S
]

C’est parfaitement adapté au GPU : une banque de convolutions / projections appliquées à (S), puis SVD ou extraction de directions.

---

## 2.2 Pourquoi c’est particulièrement fort pour POP-PLS

POP-PLS, dans ta définition, choisit une transformation par composante :

[
b_a \in {1,\dots,B}
]

[
X^{(a)} = X A_{b_a}^\top
]

En NIPALS, cela impose une logique séquentielle assez lourde : à chaque composante, tu testes plusieurs opérateurs dans un processus itératif.

En SIMPLS, la sélection devient :

[
S_{b,a} = A_b C_{a-1} S
]

où (C_{a-1}) est le projecteur qui enlève les directions déjà utilisées.

Ensuite tu scores chaque opérateur (b) pour la composante (a), tu choisis le meilleur, puis tu passes à la composante suivante.

Donc POP-SIMPLS devient une forme de :

> **greedy operator pursuit in covariance space**

C’est élégant, rapide, et publiable.

---

## 2.3 Le point technique critique : l’orthogonalisation doit se faire dans l’espace original

Il y a un piège.

Si tu changes d’opérateur à chaque composante, les composantes vivent dans des espaces transformés différents :

[
X A_{b_1}^\top,\quad X A_{b_2}^\top,\quad \dots
]

Donc il ne faut pas naïvement orthogonaliser les loadings SIMPLS dans “l’espace de l’opérateur courant”, sinon tu risques d’avoir une orthogonalité mal définie.

La bonne formulation consiste à revenir à des **poids effectifs dans l’espace original**.

Si SIMPLS donne une direction (r_{b,a}) dans l’espace transformé, alors le poids effectif sur (X) est :

[
z_{b,a} = C_{a-1} A_b^\top r_{b,a}
]

et le score est :

[
t_{b,a} = X z_{b,a}
]

Ensuite tu construis les loadings dans l’espace original :

[
p_a = \frac{X^\top t_a}{t_a^\top t_a}
]

et tu mets à jour le projecteur (C_a) à partir de ces loadings.

C’est probablement la formulation propre à écrire dans l’article.

---

# 3. Nouvelle formulation générale : Operator-Adaptive PLS

Je pense qu’il faut maintenant poser le problème comme une famille générale.

## 3.1 Cadre commun

On a une banque d’opérateurs :

[
\mathcal{A} = {A_1,\dots,A_B}
]

Chaque composante (a) utilise soit :

### Cas 1 — PLS classique

[
A^{(a)} = I
]

### Cas 2 — AOM global hard

Un seul opérateur pour tout le modèle :

[
A^{(1)} = A^{(2)} = \dots = A^{(K)} = A_b
]

### Cas 3 — POP hard per-component

Un opérateur par composante :

[
A^{(a)} = A_{b_a}
]

### Cas 4 — AOM soft mixture

Une combinaison d’opérateurs par composante :

[
A^{(a)} = \sum_{b=1}^B \alpha_{a,b} A_b
]

avec :

[
\alpha_{a,b} \ge 0,\quad \sum_b \alpha_{a,b}=1
]

### Cas 5 — Super-block / multi-view PLS

On concatène toutes les vues :

[
X_\text{super}
==============

[
XA_1^\top,\dots,XA_B^\top
]
]

puis on fait une PLS avec régularisation par groupe.

C’est l’alternative “je donne tout à PLS, mais je contrôle la redondance”.

---

## 3.2 Les axes expérimentaux à explorer

Tu veux explorer la diversité. Je pense qu’il faut formaliser ça comme une matrice de combinaisons.

| Axe                 | Modalités prioritaires                                               |
| ------------------- | -------------------------------------------------------------------- |
| Moteur PLS          | NIPALS-adjoint, SIMPLS-covariance, SIMPLS matérialisé référence      |
| Sélection opérateur | global, per-component, auto                                          |
| Gate                | hard, softmax, sparsemax/entmax, group-sparse                        |
| Critère             | covariance proxy, PRESS, CV5, hybrid PRESS+CV, holdout historique    |
| Banque              | identity, compact, default, extended, OSC/Whittaker, nonlinear-stack |
| Réponse             | PLS1, PLS2                                                           |
| Backend             | NumPy référence, Torch/JAX GPU, C/BLAS                               |
| Objectif            | RMSE, stabilité, temps, interprétabilité, sélection opérateur        |

Mais il ne faut pas tout lancer naïvement en full-factorial sur 60 datasets. Il faut **implémenter toutes les combinaisons**, mais les benchmarker en deux temps :

1. **screening large sur petit panel diversifié** ;
2. **validation statistique sur corpus complet**.

Sinon tu vas exploser le budget expérimental.

---

# 4. Point important : le soft AOM-SIMPLS peut dégénérer en hard selection

C’est une observation scientifique importante.

Si tu prends :

[
A_\alpha = \sum_b \alpha_b A_b
]

et que tu optimises uniquement la norme de covariance :

[
\max_{\alpha \in \Delta_B} |A_\alpha S|^2
]

alors l’objectif est essentiellement convexe en (\alpha). Maximiser une fonction convexe sur un simplexe donne généralement une solution à un sommet du simplexe.

Donc :

[
\alpha = e_b
]

Autrement dit, la soft mixture retombe souvent sur une sélection hard.

Conséquence :

> En SIMPLS pur, une mixture soft d’opérateurs n’a pas forcément d’intérêt si l’objectif est seulement la covariance.

Elle devient intéressante si :

* on optimise un critère prédictif type PRESS/CV ;
* on ajoute une régularisation qui récompense explicitement les mélanges ;
* on passe à une formulation super-block ;
* on fait du DARTS / différentiable PLS sur une validation loss ;
* on utilise des opérateurs complémentaires non redondants.

Donc je ne mettrais pas “soft AOM-SIMPLS” comme hypothèse principale. Je la garderais comme axe exploratoire.

La formulation la plus sérieuse pour mélanger vraiment les opérateurs est plutôt :

[
X_\text{super}
==============

[
XA_1^\top,\dots,XA_B^\top
]
]

avec pénalité group-sparse.

---

# 5. Ce qu’il faut faire maintenant

## Priorité 1 — Ajouter un nouvel axe au roadmap : `engine = "nipals" | "simpls"`

Le roadmap actuel doit être modifié. Aujourd’hui il suppose implicitement NIPALS/adjoint.

Je rajouterais un jalon avant la sélection PRESS/CV :

## M1-S — SIMPLS operator engine

Objectif : implémenter un moteur SIMPLS opérateur-compatible avant de figer les critères, banques et defaults.

Nouvelle signature interne :

```python
fit_operator_pls(
    X,
    Y,
    bank,
    engine="nipals_adjoint" | "simpls_covariance" | "simpls_materialized",
    selection="global" | "per_component" | "auto",
    gate="hard" | "softmax" | "sparsemax" | "group_sparse",
    selection_criterion="press" | "cv5" | "hybrid" | "holdout20",
    max_components=25,
    n_components="auto",
    operator_normalization="none" | "frobenius" | "data_variance",
)
```

L’objectif n’est pas d’avoir une belle API utilisateur tout de suite. L’objectif est d’avoir un **moteur expérimental factoriel**.

---

## Priorité 2 — Construire trois implémentations de référence

Il faut trois versions, dans cet ordre.

### 1. Référence matérialisée

Pour chaque opérateur :

[
X_b = X A_b^\top
]

puis tu appelles une PLS standard.

C’est lent, mais c’est le sol de vérité.

Elle sert à vérifier que les versions rapides donnent bien les mêmes directions ou les mêmes prédictions, au moins dans les cas où elles doivent coïncider.

### 2. NIPALS-adjoint

C’est ton moteur actuel, mais nettoyé :

* AOM global ;
* POP per-component ;
* PRESS/CV ;
* `max_components` / `n_components="auto"` ;
* mêmes objets de sortie que SIMPLS.

### 3. SIMPLS-covariance

Nouveau moteur :

[
S = X^\top Y
]

[
S_{b,a} = A_b C_{a-1} S
]

[
r_{b,a} = u_1(S_{b,a})
]

[
z_{b,a} = C_{a-1} A_b^\top r_{b,a}
]

[
t_{b,a} = X z_{b,a}
]

Puis mise à jour des loadings et du projecteur.

C’est cette version qui deviendra probablement la base rapide GPU/C.

---

## Priorité 3 — Écrire les tests mathématiques avant les benchmarks

Tests indispensables :

### Test 1 — Identité

Avec banque :

[
\mathcal{A} = {I}
]

AOM/POP doivent retomber sur PLS standard.

### Test 2 — Singleton operator

Avec banque :

[
\mathcal{A} = {A_b}
]

AOM-SIMPLS covariance doit matcher SIMPLS sur (X A_b^\top).

### Test 3 — Global hard

AOM global hard doit choisir le même opérateur que la version matérialisée, à tolérance numérique près, si le critère est identique.

### Test 4 — POP fixed sequence

Si tu imposes une séquence :

[
(b_1,b_2,\dots,b_K)
]

la version rapide doit produire les mêmes prédictions que la version de référence séquentielle.

### Test 5 — Coefficients originaux

Pour toute version opérateur, la prédiction doit pouvoir s’écrire :

[
\hat{Y} = X B + \bar{Y}
]

avec :

[
B = Z(P^\top Z)^{-1}Q^\top
]

où (Z) contient les poids effectifs dans l’espace original.

C’est crucial pour avoir une API propre, pour R/Python, et pour la reproductibilité.

---

# 6. Nouvelle roadmap scientifique recommandée

Je réorganiserais les documents comme suit.

## Phase A — Fondations

### A0. Benchmark et stats

Garder ce qui est déjà prévu :

* cohorte 55–60 datasets ;
* résultats référence ;
* RMSE / R² ;
* wall-clock ;
* Friedman–Nemenyi ;
* Nadeau–Bengio ;
* bootstrap CI ;
* sélection opérateur ;
* stabilité.

C’est indispensable.

### A1. Moteur factoriel

Implémenter :

| Moteur             | Statut                     |
| ------------------ | -------------------------- |
| NIPALS matérialisé | référence lente            |
| NIPALS-adjoint     | moteur actuel              |
| SIMPLS matérialisé | référence lente            |
| SIMPLS-covariance  | nouveau moteur rapide      |
| Superblock-SIMPLS  | baseline “tout concaténer” |

Cette phase passe avant la décision PRESS/CV.

---

## Phase B — Screening expérimental

Sur 8 à 12 datasets représentatifs :

* Rice Amylose ;
* Beer ;
* Firmness ;
* Milk Lactose ;
* Leaf P ;
* un dataset très petit (n < 80) ;
* un dataset grand (p) ;
* un dataset scatter/baseline fort ;
* un dataset multi-response si disponible ;
* un dataset où SNV/MSC est connu utile.

Tester :

| Famille            | Configurations                            |
| ------------------ | ----------------------------------------- |
| Baselines          | PLS, SG+PLS grid, SNV+PLS, MSC+PLS        |
| AOM NIPALS         | global-hard, per-component-hard           |
| AOM SIMPLS         | global-hard, per-component-hard           |
| Soft               | sparsemax/softmax, seulement exploratoire |
| Superblock         | raw superblock, group-sparse              |
| Nonlinear external | SNV/AOM, MSC/AOM, branch+merge            |

Objectif : réduire la matrice avant corpus complet.

---

## Phase C — Corpus complet

Sur 55–60 datasets :

Ne garder que 6 à 10 modèles.

Je proposerais :

1. PLS standard ;
2. grid SG/SNV/MSC + PLS ;
3. AOM-NIPALS-global ;
4. POP-NIPALS-per-component ;
5. AOM-SIMPLS-global ;
6. POP-SIMPLS-per-component ;
7. Superblock-SIMPLS group-sparse ;
8. branch+merge SNV/MSC/AOM ;
9. FCK-PLS ;
10. meilleur modèle de référence externe : Ridge/CatBoost/TabPFN-opt si disponible.

Ce serait une comparaison très forte.

---

## Phase D — Optimisation bas niveau

Seulement après avoir identifié les variants scientifiquement valables.

À optimiser :

| Partie                     | Candidat backend            |
| -------------------------- | --------------------------- |
| SIMPLS covariance          | C/BLAS, CUDA, Torch, JAX    |
| opérateurs convolutionnels | CUDA conv1d / FFT           |
| batch opérateurs           | GPU naturel                 |
| PRESS score-space          | C/BLAS                      |
| superblock                 | BLAS / sparse group penalty |
| wrappers                   | Python pybind11, R Rcpp     |

Ne pas commencer par le C/GPU. D’abord la vérité mathématique.

---

# 7. Classement des variantes à implémenter

## À implémenter absolument

### 1. `NIPALS + global + hard`

C’est le baseline AOM actuel.

### 2. `NIPALS + per_component + hard`

C’est POP dans le cadre unifié.

### 3. `SIMPLS + global + hard`

C’est le nouveau concurrent direct d’AOM-NIPALS.

### 4. `SIMPLS + per_component + hard`

C’est probablement la variante la plus prometteuse pour POP.

### 5. `SIMPLS materialized reference`

Pas pour performance, mais pour validation.

### 6. `PRESS / CV5 / hybrid`

Indispensable, car le holdout 20 % doit disparaître.

---

## À implémenter fortement

### 7. `Superblock-SIMPLS`

C’est le baseline scientifique naturel :

> “Et si on concatène toutes les transformations et qu’on laisse PLS choisir ?”

Cela permet de comparer AOM/POP à une stratégie multi-view.

### 8. `Group-sparse Superblock-SIMPLS`

Encore plus important. Sans sparsité, le superblock risque de diluer les poids entre opérateurs redondants.

Avec sparsité par groupe, tu as un concurrent sérieux et publiable.

### 9. OSC / Whittaker / wavelet / FFT

Les documents ont raison : OSC et Whittaker sont des candidats linéaires-at-apply très importants.

---

## À garder exploratoire

### 10. Softmax / sparsemax AOM

Intéressant, mais je ne le mettrais pas au centre. Le risque de dégénérescence vers hard selection est réel.

### 11. DARTS PLS

Scientifiquement intéressant, mais probablement comme baseline “differentiable search”, pas comme modèle principal.

### 12. MoE PLS

Très utile comme témoin : il montre que les méthodes non linéaires gagnent parfois, mais à coût et instabilité élevés.

---

## À éviter dans le cœur

### Pseudo-linear SNV dans la banque

Je garderais la position des documents : ne pas l’inclure dans la banque stricte. SNV/MSC doivent rester :

* upstream ;
* ou branch+merge ;
* ou stacking.

Ne mélange pas “strict linear operator bank” et “pseudo-linéaire approximatif”. Ça affaiblirait l’argument théorique.

---

# 8. Les hypothèses scientifiques fortes à tester

Je formulerais les hypothèses comme ça.

## H1 — SIMPLS-covariance est équivalent ou quasi équivalent à NIPALS en performance prédictive, mais plus rapide

[
\text{RMSE(SIMPLS-AOM)} \approx \text{RMSE(NIPALS-AOM)}
]

avec :

[
\text{time(SIMPLS-AOM)} < \text{time(NIPALS-AOM)}
]

surtout quand (B), (p) et (q) augmentent.

---

## H2 — POP/per-component est utile sur les spectres où plusieurs phénomènes structuraux coexistent

Exemple :

* composante 1 : baseline / scatter ;
* composante 2 : dérivée chimique ;
* composante 3 : haute fréquence / bruit ;
* composante 4 : correction orthogonale.

Dans ces cas :

[
\text{selection="per_component"}
]

devrait battre :

[
\text{selection="global"}
]

Mais sur petits (n), per-component peut sursélectionner.

---

## H3 — La banque compacte gagne en petit (n), la banque étendue gagne en moyen/grand (n)

Ça rejoint le backlog.

Tu peux tester :

[
n < 80,\quad 80 \le n < 300,\quad n \ge 300
]

et comparer :

* compact ;
* default ;
* extended ;
* OSC/Whittaker.

---

## H4 — Soft mixture n’aide que si elle est évaluée par un critère prédictif, pas par covariance brute

C’est une hypothèse importante. Elle peut expliquer pourquoi sparsemax n’a pas encore démontré d’avantage.

---

## H5 — Superblock group-sparse est le vrai concurrent “multi-opérateur”

Si superblock group-sparse bat AOM/POP, cela veut dire que la sélection hard est trop restrictive.

S’il ne bat pas AOM/POP, ton argument devient très fort :

> Il n’est pas nécessaire de concaténer toutes les transformations ; une sélection opérateur structurée suffit.

---

# 9. Nouvelle structure possible pour l’article

Le papier pourrait devenir plus fort que le plan actuel.

## Titre conceptuel possible

**Operator-Adaptive Partial Least Squares: unifying global, component-wise and covariance-space preprocessing selection for spectroscopy**

Ou, plus court :

**Operator-Adaptive PLS for Spectroscopy**

Le nom AOM-PLS peut rester pour l’implémentation historique, mais la contribution scientifique devient plus générale.

---

## Plan d’article recommandé

### 1. Motivation

Le choix du preprocessing en NIRS est combinatoire, instable et coûteux.

### 2. Cadre général

Définir la banque d’opérateurs :

[
\mathcal{A} = {A_b}
]

Définir :

* global selection ;
* per-component selection ;
* soft mixture ;
* superblock.

### 3. Deux moteurs

#### NIPALS-adjoint

Version historique, séquentielle, proche PLS classique.

#### SIMPLS-covariance

Nouvelle formulation :

[
S_b = A_b X^\top Y
]

### 4. Sélection

Comparer :

* holdout ;
* PRESS ;
* CV ;
* hybrid.

### 5. Théorèmes / propositions

À formaliser :

#### Proposition 1 — Identity dominance

Si (I\in\mathcal{A}), alors sur le critère de sélection :

[
\mathrm{Err}(\text{Operator-PLS}) \le \mathrm{Err}(\text{PLS})
]

#### Proposition 2 — Covariance equivalence

Pour tout opérateur linéaire (A_b) :

[
(XA_b^\top)^\top Y = A_b X^\top Y
]

Donc SIMPLS peut opérer dans l’espace des covariances.

#### Proposition 3 — Effective-weight representation

Même avec changement d’opérateur par composante, la prédiction peut s’écrire :

[
\hat{Y} = X B + \bar{Y}
]

avec des poids effectifs (z_a) dans l’espace original.

#### Proposition 4 — Linear-at-apply principle

Un opérateur peut entrer dans la banque si son application à prediction-time est linéaire, même s’il est fitted à training-time, à condition que l’évaluation évite les fuites.

Cela justifie OSC/EPO, mais exclut SNV/MSC.

### 6. Expériences

* corpus complet ;
* NIPALS vs SIMPLS ;
* global vs per-component ;
* banques ;
* PRESS/CV ;
* superblock ;
* SNV/MSC stacking ;
* temps CPU/GPU.

### 7. Discussion

* quand hard suffit ;
* quand POP aide ;
* quand non-linéaire externe est nécessaire ;
* limites PLS2 ;
* limites petits (n).

---

# 10. Architecture logicielle recommandée

Je séparerais strictement trois couches.

## 10.1 Couche opérateurs

```python
class LinearOperator:
    def fit(self, X, y=None):
        return self

    def apply_x(self, X):
        ...

    def apply_cov(self, S):
        # returns A @ S
        ...

    def adjoint_vec(self, v):
        # returns A.T @ v
        ...

    def family(self):
        ...
```

`apply_cov` est fondamental pour SIMPLS.

Pour les opérateurs convolutionnels, `apply_cov(S)` est juste la convolution du vecteur ou des colonnes de (S).

---

## 10.2 Couche moteurs

```python
class PLSCore:
    def extract(self, X, Y, bank, config):
        ...
```

Moteurs :

```python
NIPALSAdjointCore
SIMPLSCovarianceCore
SIMPLSMaterializedReference
SuperblockSIMPLSCore
```

---

## 10.3 Couche politiques

```python
GlobalHardSelection
PerComponentHardSelection
AutoPolicySelection
SoftMixtureSelection
GroupSparseSelection
```

Ces politiques ne doivent pas connaître les détails de NIPALS ou SIMPLS. Elles demandent au moteur des candidats, puis les scorent.

---

## 10.4 Couche critères

```python
CovarianceScore
PRESSScore
CV5Score
HybridPressCVScore
ExternalValidationScore
```

C’est important pour éviter que PRESS soit câblé dans une méthode mais pas dans l’autre.

---

# 11. Attention : “implémenter toutes les combinaisons” oui, mais pas toutes dans l’API finale

Pour l’exploration, oui :

[
\text{engine} \times \text{selection} \times \text{gate} \times \text{criterion} \times \text{bank}
]

Mais pour la librairie utilisateur, il faut éviter une API illisible.

Je recommande :

```python
AOMPLSRegressor(
    engine="auto",              # "nipals" | "simpls" | "auto"
    selection="global",         # "global" | "per_component"
    selection_criterion="press",
    operator_bank="default",
    max_components=25,
    n_components="auto",
)
```

Et garder les variantes avancées dans un module expérimental :

```python
plsmania.experimental.OperatorPLSResearchGrid
```

ou :

```python
AOMPLSExperimentalRegressor
```

Le moteur de recherche peut explorer tout, mais le modèle public doit rester lisible.

---

# 12. Verdict stratégique

## Est-ce intéressant ?

Oui, très.

Avec SIMPLS, le projet devient plus fort. Tu passes de :

> “J’ai une variante AOM-PLS NIPALS qui choisit un preprocessing.”

à :

> “Je propose un cadre général de PLS adaptative par opérateurs, avec deux solveurs, plusieurs granularités de sélection, une formulation covariance-space rapide, et une validation statistique large.”

C’est beaucoup plus ambitieux scientifiquement.

---

## Est-ce faisable ?

Oui, mais il faut procéder dans le bon ordre :

1. référence matérialisée ;
2. NIPALS unifié ;
3. SIMPLS covariance ;
4. tests d’équivalence ;
5. screening factoriel ;
6. corpus complet ;
7. optimisation bas niveau.

Ne pas commencer par C/GPU. Le danger principal n’est pas la vitesse. Le danger principal est une orthogonalisation mal définie dans POP-SIMPLS.

---

## Est-ce crédible pour publication ?

Oui, si tu évites de raconter “on a testé plein de trucs”.

La narration doit être :

> Les méthodes existantes choisissent le preprocessing hors du modèle. Nous proposons un cadre où le preprocessing est représenté par une banque d’opérateurs linéaires et sélectionné pendant l’extraction PLS. Nous montrons que ce cadre se décline en sélection globale, par composante, et en formulation SIMPLS dans l’espace des covariances. Nous validons empiriquement les choix de critère, de banque et de moteur sur un large corpus NIRS.

C’est solide.

---

# 13. Ma recommandation finale

Je modifierais immédiatement le plan actuel comme suit :

## À garder du roadmap actuel

* benchmark 60 datasets ;
* stats sérieuses ;
* suppression du holdout 20 % ;
* PRESS/CV/hybrid ;
* split `max_components` / `n_components`;
* unification AOM/POP ;
* operator-bank API ;
* OSC/Whittaker/wavelet/FFT ;
* exclusion de SNV/MSC de la banque stricte ;
* branch+merge pour non-linéaire.

## À ajouter avant tout ça

Un jalon nouveau :

> **M1-S — Operator-PLS engine layer: NIPALS vs SIMPLS**

Avec :

* `engine="nipals_adjoint"`;
* `engine="simpls_covariance"`;
* `engine="simpls_materialized_reference"`;
* `selection="global"`;
* `selection="per_component"`;
* tests d’équivalence ;
* microbenchmarks ;
* pilot sur 8–12 datasets.

## À ne pas faire maintenant

* ne pas optimiser en C/GPU avant d’avoir validé la formulation ;
* ne pas mettre sparsemax au centre ;
* ne pas mettre pseudo-SNV dans la banque ;
* ne pas publier tous les variants comme s’ils étaient également importants ;
* ne pas laisser le roadmap actuel figer les defaults uniquement sur NIPALS.

---

# 14. Formule courte du nouveau projet

La phrase scientifique centrale pourrait être :

> **Operator-Adaptive PLS learns the preprocessing structure of a PLS model by selecting, per model or per component, from a bank of linear spectral operators; the same framework admits both an adjoint-NIPALS implementation and a covariance-space SIMPLS implementation, enabling interpretable preprocessing selection with scalable CPU/GPU execution.**

En français :

> **Operator-Adaptive PLS apprend la structure de prétraitement d’un modèle PLS en sélectionnant, globalement ou composante par composante, des opérateurs spectraux linéaires ; le même cadre admet une implémentation NIPALS par adjoint et une implémentation SIMPLS dans l’espace des covariances, ce qui combine interprétabilité, stabilité statistique et scalabilité.**

