# Recapitulatif des modeles benchmarkes

Date de synthese : 2026-05-12.

Ce document regroupe les modeles disperses dans `bench/` : socle PLS/Ridge,
AOM, AOM-Ridge, multiview, multi-kernel, FCK, TabPFN, NICON/CNN et hybrides.
Les aliases historiques du CSV sont regroupes par famille pour rester lisible :
le but est d'etre exhaustif au niveau des principes, variantes et statuts, pas
de recopier les centaines de labels de runs un par un.

Sources locales principales :

- `bench/benchmark_master_results.csv`
- `bench/benchmark_synthesis.md`
- `bench/model_exploration_review.md`
- `bench/scenarios/model_registry.yaml`
- `bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md`
- `bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md`
- `bench/AOM_v0/Multi-kernel/*/docs/*MATH_SPEC.md`
- `bench/AOM_v0/multiview/docs/SUMMARY.md`
- `bench/fck_pls/docs/FCK_EVALUATION.md`
- `bench/nicon_v2/docs/STATUS.md`

## 1. Vue d'ensemble

### 1.1 Familles presentes dans le master

| Famille | Role | Etat empirique court |
|---|---|---|
| PLS | Denominateur chemometrique principal | Reference rapide, robuste, interpretable. |
| Ridge | Denominateur lineaire regulier | Tres fort apres preprocessing/HPO ; base de plusieurs hybrides. |
| AOM-PLS | Selection adaptative d'operateurs spectraux avant PLS | Meilleure direction PLS classique ; compact + ASLS + CV est le regime stable. |
| POP-PLS | Selection d'un operateur par composante PLS | Utile conceptuellement, mais plus instable que AOM compact. |
| AOM-Ridge | AOM transpose a Ridge/kernel Ridge | Meilleur challenger non-TabPFN ; tres bon compromis perf/cout. |
| Multi-kernel ridge | Moyenne ou apprentissage de poids entre noyaux AOM | Bon diagnostic de complementarite ; secondaire face a AOM-Ridge. |
| Multiview / MoE / ASL | Experts par bloc spectral, preprocessing ou candidat | Fort potentiel mais risque de selection/leakage ; nesting obligatoire. |
| FCK | Filtres convolutionnels fractionnaires + PLS/Ridge/AOM | Interessant comme diversite, pas promu en preset fort. |
| TabPFN | Prior tabulaire foundation model + preprocessing search | Meilleure famille observee, mais couteuse et moins interpretable. |
| NICON/CNN | CNN spectraux purs et hybrides | CNN pur no-go ; le role utile est residuel ou feature extractor. |
| Hybrides CNN/lineaire | Stacking ou residualisation autour de Ridge/PLS/AOM | Ameliore les CNN purs, mais reste derriere AOM-Ridge/TabPFN. |
| CatBoost / paper CNN | References papier externes | Comparateurs, pas des candidats principaux du harness courant. |

### 1.2 Lecture des scores

Deux ratios coexistent :

- `score_ratio_vs_source_run_pls` : comparaison dans le meme protocole. C'est
  le bon ratio pour juger si une methode ameliore son propre PLS.
- `score_ratio_vs_dataset_pls` : comparaison stricte contre le meilleur PLS
  trouve pour le dataset, tous protocoles confondus. C'est le ratio leaderboard.

Les lignes `oracle_by_model_class` donnent le potentiel optimiste d'une famille
si elle pouvait choisir sa meilleure variante par dataset. Ce n'est pas un
protocole deployable.

| Classe oracle | Datasets | Mediane rel. RMSEP vs PLS | Wins vs PLS |
|---|---:|---:|---:|
| TabPFN | 59 | 0.908 | 45/59 |
| AOM-PLS | 59 | 0.923 | 49/59 |
| AOM-Ridge | 58 | 0.942 | 49/58 |
| Ridge | 58 | 0.972 | 43/58 |
| Meta-selector/MoE | 59 | 0.975 | 38/59 |
| Hybrid CNN+AOM | 42 | 0.978 | 27/42 |
| Multi-kernel ridge | 53 | 0.983 | 34/53 |
| PLS | 67 | 1.000 | 0/67 |
| FCK-PLS | 8 | 1.005 | 4/8 |
| Hybrid CNN+linear | 51 | 1.005 | 24/51 |
| NICON/CNN | 56 | 1.018 | 26/56 |
| CatBoost | 57 | 1.038 | 23/57 |
| POP-PLS | 59 | 1.459 | 9/59 |

## 2. Notation commune

On note :

```text
X in R^(n x p)    n spectres, p longueurs d'onde / variables
Y in R^(n x q)    cible, q=1 en regression simple
A_b in R^(p x p)  operateur spectral lineaire strict
X_b = X A_b^T     spectres transformes par l'operateur b
```

Les modeles centrent en general `X` et `Y` pendant `fit`, puis predisent :

```text
Y_hat = (X - x_mean) B + y_mean
```

Pour les operateurs lineaires, l'identite centrale d'AOM est :

```text
X_b^T Y = (X A_b^T)^T Y = A_b X^T Y
```

Elle permet d'evaluer rapidement des transformations spectrales en espace de
covariance sans materialiser toutes les versions transformees de `X`.

## 3. Socle : PLS, SIMPLS et Ridge

### 3.1 PLS standard

PLS cherche des directions latentes `t = X w` qui maximisent la covariance avec
`Y`, tout en deflatant `X` et `Y` composante par composante.

Pour PLS1, une extraction NIPALS simplifiee est :

```text
w_a = normalize(X_res^T y_res)
t_a = X_res w_a
p_a = X_res^T t_a / (t_a^T t_a)
q_a = y_res^T t_a / (t_a^T t_a)
X_res <- X_res - t_a p_a^T
y_res <- y_res - t_a q_a
```

Avec `W`, `P`, `Q` empiles sur les composantes :

```text
B = W (P^T W)^+ Q^T
```

ou `+` est l'inverse si possible, sinon la pseudo-inverse.

Variantes associees :

- `PLS-standard`, `PLS-standard-numpy`, `PLS-baseline`,
  `PLSRegression` : PLS de reference.
- `PLS-tuned-cv5` : nombre de composantes choisi par CV 5-fold.
- `PLS-Tuned` / PLS papier : PLS avec recherche de preprocessing beaucoup plus
  couteuse ; ne doit pas etre confondu avec le PLS cheap.
- `SIMPLS` : alternative de de Jong, travaille par deflation de la covariance
  `X^T Y` plutot que par NIPALS.
- `IKPLS` : implementation rapide de PLS.
- `PCR` : regression sur composantes principales de `X`, sans utiliser `Y`
  pour construire les composantes.

### 3.2 PLS-DA et variantes classification

PLS-DA encode les classes comme une matrice indicatrice et applique une PLS2.
Dans la version AOM-PLS-DA, les labels sont codes avec ponderation par classe :

```text
Y_ic = 1 / sqrt(pi_c) si y_i = c, sinon 0
```

Puis les scores latents alimentent une calibration logistique ou un softmax
temperature si la calibration echoue.

Variantes :

- `PLSDA`
- `OPLSDA`
- `AOMPLSClassifier`
- `POPPLSClassifier`

### 3.3 Ridge

Ridge resout :

```text
min_B ||Yc - Xc B||_F^2 + alpha ||B||_F^2
B = (Xc^T Xc + alpha I)^-1 Xc^T Yc
```

En forme duale avec `K = Xc Xc^T` :

```text
C = (K + alpha I)^-1 Yc
Y_hat_c = K_* C
```

Variantes associees :

- `Ridge-tuned-cv5`, `Ridge-baseline`, `Ridge-raw`, `Ridge-raw-stdscale`
- `Ridge` papier : Ridge avec recherche de preprocessing large.
- Ridge comme meta-modele dans les stackings multiview/CNN.

## 4. Variantes PLS chemometriques natives

Ces modeles existent dans `nirs4all/operators/models/sklearn/`. Tous ne sont
pas champions du master, mais ils forment le vocabulaire de base.

| Modele | Principe | Usage / limite |
|---|---|---|
| `OPLS` | Retire les directions de `X` orthogonales a `Y`, puis PLS. | Utile pour bruit structure ; risque d'instabilite si trop de composantes orthogonales. |
| `MBPLS` | PLS multi-blocs, chaque vue/bloc contribue a des scores communs. | Base des idees multiview. |
| `DiPLS` | PLS invariante au domaine. | Transfert / domain shift. |
| `SparsePLS` | Penalisation sparse sur les poids PLS. | Selection de variables ; plus fragile en petit `n`. |
| `LWPLS` | PLS locale par echantillon, ponderee par similarite. | Capture heterogeneite locale ; cout prediction eleve. |
| `IntervalPLS` | Selection d'intervalles spectraux. | Interpretable, utile si signal localise. |
| `RobustPLS` | PLS robuste aux outliers. | Reference robuste, pas famille champion actuelle. |
| `RecursivePLS` | Mise a jour recursive / streaming. | Scenarios incrementaux. |
| `KernelPLS` / `NLPLS` / `KPLS` | PLS sur matrice noyau. | Non-lineaire, mais cout `n x n`. |
| `KOPLS` | Kernel + OPLS. | Non-lineaire avec filtrage orthogonal. |
| `OKLMPLS` | PLS apres featurizer explicite/approxime. | Pont vers features non-lineaires. |
| `FCKPLS` / `FractionalPLS` | Features par filtres fractionnaires + PLS. | Voir section FCK. |

## 5. AOM-PLS

### 5.1 Idee

AOM-PLS remplace le choix fixe d'un preprocessing par une banque d'operateurs
spectraux. Chaque operateur est une "lentille" candidate sur le spectre :

```text
X_b = X A_b^T
```

La PLS est ensuite ajustee apres selection d'un ou plusieurs `A_b`.

### 5.2 Coefficients effectifs

Si la composante `a` est extraite dans l'espace transforme avec direction
`r_a` sous l'operateur `A_ba`, le poids original equivalent est :

```text
z_a = A_ba^T r_a
T = X Z
B = Z (P^T Z)^+ Q^T
```

L'interet est que la prediction reste dans l'espace original :

```text
Y_hat = (X - x_mean) B + y_mean
```

### 5.3 Politiques de selection

| Politique | Principe | Variantes / labels |
|---|---|---|
| `none` | Banque identite seulement ; equivalent PLS. | Reference interne. |
| `global` | Choisit un seul operateur pour tout le modele. | `AOM-default-*`, `AOM-PLS-compact-*`. |
| `per_component` | Choisit un operateur par composante. | POP-PLS, voir section 6. |
| `soft` | Melange convexe `A_alpha = sum_b alpha_b A_b`. | Experimental sparsemax/softmax. |
| `superblock` | Concatene tous les blocs transformes. | `Superblock-raw-simpls-numpy`. |
| `active_superblock` | Prescreen covariance + diversite + superblock. | `ActiveSuperblock-*`. |

Les criteres de selection sont `cv`, `press`, `approx_press`, `hybrid`,
`covariance` ou `holdout`. Pour les claims solides, `cv`/nested CV est le
critere a privilegier.

### 5.4 Banques d'operateurs

| Banque | Taille typique | Contenu |
|---|---:|---|
| `compact` | 9 | Identite + SG/detrend/derivations fixes, strictement lineaires. |
| `family_pruned` | 15 | Familles plus larges mais dedupliquees. |
| `response_dedup` | 47 | Banque large dedupliquee par reponse. |
| `deep` / historiques | 50+ | Exploration large, plus de variance de selection. |
| `compact + FCK` | 17 | 9 operateurs AOM + 8 FCK selectionnables. |

Le resultat recurrent est que les banques compactes battent souvent les banques
tres larges : plus de candidats augmente le risque de winner's curse.

### 5.5 Variantes principales

| Variante | Description | Statut |
|---|---|---|
| `AOM-PLS-compact-numpy` | AOM-PLS compact, moteur NumPy. | Locked, baseline AOM utile. |
| `ASLS-AOM-compact-cv5-numpy` | ASLS avant AOM compact, selection CV5. | Locked, meilleur default classique PLS-side. |
| `ASLS-AOM-compact-cv3/repcv3` | Variantes CV plus legeres/repetee. | Bonnes references, un peu moins stables que CV5. |
| `ASLS-AOM-family-pruned-*` | Banque plus large. | Exploration ; gains non systematiques. |
| `ASLS-AOM-response-dedup-*` | Banque large dedupliquee. | Utile diagnostic, plus risquee. |
| `SPXY-AOM-*` | Selection AOM avec splits SPXY. | Comparaison split-aware. |
| `AOM-default-nipals-adjoint-numpy` | Moteur NIPALS adjoint. | Locked. |
| `AOM-default-simpls-covariance-numpy` | Moteur SIMPLS covariance. | Locked / equivalence technique. |
| `AOM-explorer-simpls-numpy` | Exploration active d'operateurs. | Diagnostic. |
| `ActiveSuperblock-simpls-numpy` | Superblock apres selection active. | Diagnostic fort, attention nesting. |
| `nirs4all-AOM-PLS-default` | Version package. | Reference package. |
| `Bandit AOM-PLS` | Screen rapide par R2 puis evaluation top-K. | Prototype `bench/AOM`. |
| `Enhanced AOM-PLS` | Ajoute pseudo-linear SNV a la banque. | Prototype. |
| `DARTS PLS` | Poids differentiables sur operateurs puis hard/blend. | Prototype couteux. |
| `Zero-Shot Router` | Choisit un pipeline via heuristiques spectrales. | Prototype conservateur. |
| `MoE PLS` | Experts preprocessing + OOF + meta-Ridge. | Prototype lourd. |

### 5.6 Conclusion AOM-PLS

AOM-PLS est la meilleure extension directe de PLS : rapide, interpretable,
spectroscopique. Sa version la plus saine est compacte, avec ASLS et selection
CV. Les variantes tres larges ou non-nestees servent surtout a mesurer le
headroom.

## 6. POP-PLS

POP-PLS signifie ici "Per-Operator-Per-component PLS". Au lieu de choisir un
operateur global, POP selectionne l'operateur composante par composante :

```text
pour a = 1..K:
    evaluer chaque operateur b sur l'etat residuel courant
    choisir b_a par PRESS/CV/covariance
    extraire la composante avec A_ba
```

Variantes :

- `POP-PLS-compact-numpy`
- `POP-simpls-covariance-numpy`
- `POP-nipals-adjoint-numpy`
- `POP-K8-cv3-numpy`
- `nirs4all-POP-PLS-default`
- `POPPLSClassifier`

Interet : plus flexible qu'AOM global. Limite : sur petit `n`, le choix greedy
par composante sur-apprend facilement. Le master montre une performance globale
moins bonne que AOM compact.

## 7. AOM-Ridge

### 7.1 Idee

AOM-Ridge remplace la tete PLS par Ridge ou kernel Ridge sur des branches AOM.
Pour un operateur strictement lineaire :

```text
Z_b = Xc A_b^T
K_b = Z_b Z_b^T = Xc A_b^T A_b Xc^T
C_b = (K_b + alpha I)^-1 Yc
beta_b = A_b^T A_b Xc^T C_b
```

Prediction equivalente :

```text
Y_hat_c = X_*c beta_b = K_*b C_b
```

### 7.2 Superblock Ridge

Avec plusieurs blocs ponderes :

```text
Phi(Xc) = [s_1 Xc A_1^T | ... | s_B Xc A_B^T]
K = sum_b s_b^2 Xc A_b^T A_b Xc^T
C = (K + alpha I)^-1 Yc
```

L'implementation expose un coefficient original `beta`, pas un coefficient
wide de dimension `B*p`.

### 7.3 Variantes principales

| Variante | Description | Statut |
|---|---|---|
| `AOMRidge-global-compact-none` | Choix/combinaison globale sur banque compact sans preproc externe. | Locked. |
| `AOMRidge-global-compact-snv` | Meme idee avec SNV. | Locked. |
| `AOMRidge-global-compact-none-msc/asls/split_aware` | Variantes preprocessing/split-aware. | Locked ou exploratoire selon run. |
| `AOMRidge-superblock-compact-*` | Superblock dual Ridge. | Locked / diagnostic. |
| `AOMRidge-Local-compact-knn50` | Ridge locale k-NN, evite le OOM grand `n`. | Locked. |
| `AOMRidge-Local-compact-knn-sweep` | Selection interne de `k` dans `[10,25,50,100,200]`. | Exploratoire. |
| `AOMRidge-MultiBranchMKL-compact-shrink03` | Multi-branch MKL avec shrinkage. | Locked, plus lent. |
| `AOMRidge-Blender-headline-spxy3` | Blending de branches headline. | Exploratoire, audit nested requis. |
| `AOMRidge-AutoSelect-headline-spxy3` | Auto-selection de branche. | Exploratoire, audit requis. |
| `AOMRidgePLS-compact-*` | Combinaisons AOM-Ridge / PLS, colscale/Hmax/FCK. | Exploratoire. |

### 7.4 Conclusion AOM-Ridge

AOM-Ridge est la meilleure direction classique hors TabPFN. Elle garde le prior
spectral d'AOM, mais Ridge amortit mieux certains regimes que PLS. Le danger est
la complexite de selection : global/local/blender/autoselect doivent rester
nested pour etre deployables.

## 8. Multi-kernel ridge, MKM et BLUP

### 8.1 mkR : multi-kernel Ridge par poids explicites

Chaque operateur produit un noyau centre et normalise :

```text
K_b_raw = Xc A_b^T A_b Xc^T
K_b = trace_normalize(H K_b_raw H)
K_eta = sum_b eta_b K_b
C = (K_eta + alpha I)^-1 yc
```

Les poids `eta` sont sur le simplexe :

```text
eta_b >= 0
sum_b eta_b = 1
```

Strategies :

- `uniform` : moyenne simple.
- `manual` : poids fournis puis projetes.
- `KTA-simplex` : poids via alignment supervise avec `y y^T`.
- `softmax-CV` : optimise `eta=softmax(theta)` sur perte CV interne.

Variantes observees :

- `mkR-softmax_cv`
- `mkR-softmax_cv-snv`
- `mkR-softmax_cv-default-active15-sparse*`
- `mkR-softmax_cv-asls/msc-default-active15-sparse*`

### 8.2 MKM : multi-kernel mixed model

MKM interprete les blocs comme effets aleatoires :

```text
y = X_f beta + sum_b u_b + e
u_b ~ N(0, sigma_b^2 K_b)
e   ~ N(0, sigma_e^2 I)
V = sum_b sigma_b^2 K_b + sigma_e^2 I
```

Les variances sont estimees par ML/REML. A theta fixe, MKM est equivalent a un
mkR dont :

```text
eta_b = sigma_b^2
alpha = sigma_e^2
```

Variantes :

- `MKM-reml`
- `MKM-reml-asls`
- `MKM-reml-msc`
- `MKM-reml-*-active15`

### 8.3 BLUP / E-BLUP

BLUP decompose la prediction MKM par bloc :

```text
alpha_dual = V^-1 (y - X_f beta_hat)
u_hat_b(*) = sigma_b^2 K_b_* alpha_dual
y_hat_* = X_*f beta_hat + sum_b u_hat_b(*)
```

Ce n'est pas le meilleur candidat leaderboard, mais c'est tres utile pour
expliquer quelle famille de transformations contribue.

### 8.4 Conclusion multi-kernel

Les modeles multi-kernel montrent que les transformations sont complementaires.
Ils sont competitifs, mais la famille AOM-Ridge simple/blender a donne un
meilleur compromis dans les runs recents. Leur role principal reste diagnostic,
interpretation et diversite dans `exhaustive_research`.

## 9. Multiview, MoE et Adaptive Super Learner

### 9.1 Vues spectrales

Une vue est aussi un operateur lineaire. Les blocs de longueurs d'onde utilisent
un masque diagonal `M` :

```text
M_ii = 1 si i dans le bloc, sinon 0
X_view = X M^T
```

Pour une vue "preprocessing x bloc", la convention est :

```text
X_view = X (M A_preproc)^T
```

Donc on preprocess sur tout le spectre, puis on masque le bloc. Cela evite les
artefacts de bord que produirait un preprocessing sur spectre zero-padde.

Modes de vues :

- `preproc_only(compact)` : 9 vues.
- `blocks_only(K=3)` : identite + 3 blocs.
- `combined(compact,K=3,+global)` : 36 vues.
- `combined(family_pruned,K=3,+global)` : 60 vues.
- `combined(response_dedup,K=3,+global)` : 188 vues, couteux.

### 9.2 Algorithmes multiview

| Variante | Principe | Labels observes |
|---|---|---|
| MBPLS vanilla | PLS multi-blocs classique. | `MBPLS-blocks3-vanilla`. |
| Block-sparse AOM-MBPLS | Selection de bloc par composante, deflation par bloc. | `block-sparse-V1-*`, `block-sparse-V2-*`. |
| Lazy V1 POP blocks | POP sur blocs, deflation globale. | `lazy-V1-POP-blocks3-holdout`. |
| Lazy V2 AOM combined | AOM sur banque combinee bloc x preproc. | `lazy-V2-AOM-combined-compact-holdout`. |
| MoE per-view | Un expert PLS par bloc/vue, gate hard/soft. | `moe-view-soft-pls`, `moe-view-hard-pls`. |
| MoE per-preproc | Un expert PLS par preprocessing. | `moe-preproc-soft-pls-compact`, `moe-preproc-hard-pls-compact`. |
| Multi-K MoE | Plusieurs decoupages K en parallele. | `moe-view-multiK-3-5-7`, `wide-2-10`, `auto`. |
| Mean ensemble | Moyenne fixe de plusieurs bases. | `mean-ensemble-3/4(-fixed)`, `trimmed-mean-4`. |
| Stacking | Meta-Ridge ou NNLS sur OOF predictions. | `ridge-stack-multiview`, `nnls-stack-*`. |
| Adaptive Super Learner | Ensemble adaptatif NNLS sur atomes forts. | `adaptive-super-learner`, `bigN-guarded`. |

### 9.3 Math MoE

Les experts `f_e(X)` produisent des predictions. Le gate donne des poids :

```text
y_hat = sum_e w_e f_e(X)
w_e >= 0
sum_e w_e = 1
```

Dans les versions robustes, les poids sont appris sur predictions OOF, pas sur
les predictions in-sample, pour limiter le leakage.

### 9.4 Resultat pratique

`moe-preproc-soft-pls-compact` est le meilleur default multiview full-57 :
environ 77 % de wins vs PLS-standard, 52 % vs AOM-PLS, mediane rel-RMSEP autour
de 0.929. Les variantes K plus larges gagnent sur certains datasets mais ne
generalisent pas toujours. Les stackings et ASL donnent du headroom, mais ne
sont deployables qu'avec un nesting strict.

## 10. FCK : Fractional Convolutional Kernels

### 10.1 Idee

FCK cree des features par convolution 1D de chaque spectre avec des filtres
fractionnaires. Un filtre statique typique est :

```text
idx     = arange(-m, m+1) * scale
gauss   = exp(-0.5 * (idx / sigma)^2)
k_alpha = gauss * sign(idx) * |idx|^alpha
k_alpha = k_alpha - mean(k_alpha)
k_alpha = k_alpha / sum(abs(k_alpha))
```

Les sorties des filtres sont concatenees :

```text
Phi_FCK(X) = [conv_k1(X), ..., conv_kB(X)]
```

Puis on ajuste PLS, Ridge ou AOM sur `Phi_FCK(X)`.

### 10.2 Banque statique

La banque statique planifiee/testee contient :

```text
alpha in {0.5, 1.0, 1.5, 2.0}
scale in {1, 2}
kernel_size in {15, 31}
```

soit 16 filtres. Les variantes AOM compact + FCK ont aussi utilise une banque
reduite : 9 operateurs AOM + 8 operateurs FCK = 17 candidats.

### 10.3 Variantes

| Variante | Principe | Statut |
|---|---|---|
| `FCKPLS` / `FractionalPLS` | FCK features + PLS. | Disponible package ; evidence globale limitee. |
| `FCK-PLS-static` | Banque FCK statique + PLS. | Exploratoire, `exhaustive_research` seulement. |
| `FCK-AOMPLS-static` | Banque FCK statique + AOM-PLS. | Meilleure variante FCK statique, mais no-go promotion. |
| `ASLS-FCK-PLS-static` | ASLS puis FCK puis PLS. | Evidence fast12 seulement. |
| `Concat-SNV-FCK-AOMPLS-static` | Concatene SNV et FCK avant AOM-PLS. | Bon vs PLS, moins stable vs AOM-Ridge. |
| `AOMPLS-compact-with-fck-full57` | FCK ajoute a la banque AOM. | FCK choisi sur 17/57, mais gate strict echoue. |
| `AOMRidgePLSCV-compact-with-fck` | Variante AOM-Ridge/PLS avec FCK. | Exploratoire. |
| `FCKResidual-AOMPLS-teacher` | FCKStatic + Ridge sur residus d'un teacher AOMPLS. | No-go promotion, mais diversite possible. |
| `FCKPLSTorch V1` | Kernels libres apprenables par backprop. | Prototype gele. |
| `FCKPLSTorch V2` | `alpha/sigma` apprenables, plus interpretable. | Prototype gele, plus instable. |

### 10.4 Residuel FCK

Le schema residuel est :

```text
teacher_oof = predictions OOF du teacher
r = y - teacher_oof
g = Ridge(FCKStatic(X), r)
s* in {0, 0.25, 0.5, 0.75, 1.0} choisi par validation
y_hat = teacher_full(X_test) + s* g(X_test)
```

La presence de `s=0` donne une voie do-no-harm.

### 10.5 Verdict FCK

FCK-AOMPLS passe un smoke fast12 mais echoue l'audit20 strict contre
AOM-Ridge : mediane +12.6 %, q90 +57.1 %, worst +102.7 %. Conclusion locale :
pas de promotion dans `best_current` ni `strong_practical`, mais inclusion
possible dans `exhaustive_research` pour diversite d'ensemble.

## 11. TabPFN

### 11.1 Idee

TabPFN apporte un prior de foundation model tabulaire. Les spectres sont d'abord
preprocesses / reduits / normalises, puis presentes au regresseur TabPFN.

Variantes :

- `TabPFN-Raw` : TabPFN applique sans grande recherche de preprocessing.
- `TabPFN-HPO-preprocessing` / `TabPFN-opt` : recherche cartesian/HPO de
  preprocessing avant TabPFN.

Contraintes pratiques du registry :

```text
n_train <= 5000
n_features <= 1000
```

### 11.2 Verdict TabPFN

C'est la meilleure famille observee en oracle strict. Son defaut n'est pas la
precision, mais le cout, l'interpretabilite et la dependance a un budget HPO de
preprocessing. Il doit rester dans `strong_practical`, `best_current` et
`exhaustive_research`, avec budget explicite.

## 12. NICON / CNN / deep spectral models

### 12.1 Socle CNN

Les CNN spectraux traitent `X` comme un signal 1D :

```text
Conv1D -> Norm -> GELU/ELU -> SpatialDropout -> MaxPool
...
GlobalAveragePooling -> Dense
```

Perte regression :

```text
L = mean((y_scaled - y_hat_scaled)^2) + weight_decay
```

Preprocessing / augmentations documentes :

- SNV, MSC, SG fixe.
- concat derivatives : `[x, SG_d1(x), SG_d2(x)]`.
- Bjerrum offset/slope/multiplicative.
- Mixup et C-Mixup.
- Learnable EMSC, deep ensembles, conformal calibration : spec presentes, pas
  axe champion actuel.

### 12.2 Variantes principales

| Variante | Principe | Verdict |
|---|---|---|
| `NICON-baseline`, `DECON-baseline` | CNN de reference papier. | Depasses par variants internes, mais faibles vs AOM/Ridge. |
| `V1a` | Reparations minimales tete/activation. | Iteration. |
| `V1b` | Concat augmentation. | Iteration. |
| `V1c` | Small kernels + GAP. | Base stacking. |
| `V2A`, `V2M`, `V2L` | Architectures AOM-inspired / learnable RMS. | CNN pur insuffisant. |
| `V2H-lowrank-r32` | Low-rank spectral operator. | Diagnostic. |
| `V6b-DistillExtended-V2M` | Distillation depuis teacher etendue. | Signal seed-0 non robuste. |
| `V6b-LucasPretrained-V2M` | Pretrain LUCAS soil. | Aide sols, domain mismatch ailleurs. |
| `Stack-Ridge-PLS-V1c` | Stack CNN features/preds avec Ridge/PLS. | Locked, utile mais pas champion global. |
| `V2L-Residual-AOMPLS` | CNN predit residus d'un teacher AOMPLS. | Seul chemin NN prometteur, mais pas promu. |
| `V2L-Residual-AOMPLS-shrinkage` | Residuel + shrinkage CV. | Exploratoire/no-go promotion. |
| `V2L-Boost-AOMPLS` | Boost autour d'AOMPLS. | Locked mais couteux. |

### 12.3 Verdict CNN

Le programme CNN pur est arrete comme chemin champion : les datasets sont trop
souvent petit-`n`/grand-`p` et le prior chimique est trop faible. Les CNN restent
utiles comme correcteurs residuels, encodeurs geles ou diagnostics, jamais comme
remplacement direct de PLS/AOM/Ridge dans le preset fort.

## 13. Hybrides, selectors et ensembles

### 13.1 Selecteurs / meta-modeles

Ces modeles choisissent ou combinent des candidats :

- `mean-ensemble-3/4(-fixed)`
- `trimmed-mean-4`
- `moe-preproc-soft-*`
- `moe-view-soft-*`
- `AdaptiveSuperLearner`
- `bestof-multiview-asls`
- `AOMRidgeBlender`
- `AOMRidgeAutoSelector`
- `ridge-stack-multiview`
- `nnls-stack-multiview`
- `nnls-stack-atoms`
- `nnls-stack-calibrated`

Mathematiquement, ils estiment :

```text
y_hat = sum_m w_m f_m(X)
```

ou bien selectionnent `argmin_m` selon un score interne.

### 13.2 Risque principal

La complementarite est reelle, mais la selection est le premier risque de
leakage. Toute ligne qui choisit un expert ou des poids a partir de performances
hors nested CV doit etre traitee comme oracle/diagnostic, meme si le score est
tres bon.

## 14. References papier externes

| Reference | Role |
|---|---|
| `paper-CNN-reference` / paper CNN | Baseline externe CNN. |
| `paper-CatBoost-reference` | Baseline gradient boosting. |
| `paper Ridge/PLS` | Souvent Ridge/PLS avec gros preprocessing search, pas equivalent aux sentinelles cheap. |
| `TabPFN-opt` papier | Alias de `TabPFN-HPO-preprocessing` dans le registry. |

Ces lignes servent a situer les methodes, mais certaines ne sont pas
runnables localement via le harness.

## 15. Registry des candidats runnable

Extrait synthetique de `bench/scenarios/model_registry.yaml`.

| Canonical name | Classe | Maturite | Tier |
|---|---|---|---|
| `PLS-tuned-cv5` | `PLSRegression` | locked | fast |
| `Ridge-tuned-cv5` | `Ridge` | locked | fast |
| `ASLS-AOM-compact-cv5-numpy` | `AOMPLSRegressor` | locked | fast |
| `AOM-PLS-compact-numpy` | `AOMPLSRegressor` | locked | fast |
| `AOM-default-nipals-adjoint-numpy` | `AOMPLSRegressor` | locked | fast |
| `POP-PLS-compact-numpy` | `POPPLSRegressor` | locked | fast |
| `AOMRidge-global-compact-none` | `AOMRidgeRegressor` | locked | medium |
| `AOMRidge-global-compact-snv` | `AOMRidgeRegressor` | locked | medium |
| `AOMRidge-Local-compact-knn50` | `AOMLocalRidge` | locked | medium |
| `AOMRidge-Local-compact-knn-sweep` | `AOMLocalRidge` | exploratory | medium |
| `AOMRidge-MultiBranchMKL-compact-shrink03` | `AOMMultiBranchMKL` | locked | slow |
| `AOMRidge-Blender-headline-spxy3` | `AOMRidgeBlender` | exploratory | slow |
| `AOMRidge-AutoSelect-headline-spxy3` | `AOMRidgeAutoSelector` | exploratory | slow |
| `MKM-reml-default` | `MKMRegressor` | locked | medium |
| `mkR-softmax-cv-default` | `SoftmaxCVMultiKernelRidge` | locked | medium |
| `TabPFN-Raw` | `TabPFNRegressor` | locked | medium |
| `TabPFN-HPO-preprocessing` | `TabPFNRegressor` | locked | very_slow |
| `moe-preproc-soft-pls-compact` | `MoEPreprocSoftPLS` | locked | medium |
| `AOMMultiView-MeanEnsemble4-fixed` | `AOMMeanEnsemble` | locked | medium |
| `AdaptiveSuperLearner-recipe-nnls` | `AdaptiveSuperLearner` | exploratory | very_slow |
| `AdaptiveSuperLearner-bigN-guarded` | `AdaptiveSuperLearner` | exploratory | slow |
| `Stack-Ridge-PLS-V1c` | `StackRidgePLS` | locked | slow |
| `V2L-Residual-AOMPLS` | `V2LResidualAOMPLS` | exploratory | very_slow |
| `V2L-Boost-AOMPLS` | `V2LBoostAOMPLS` | locked | very_slow |
| `FCK-AOMPLS-static` | `AOMPLSRegressor` | exploratory | medium |
| `FCK-PLS-static` | `PLSRegression` | exploratory | medium |
| `Concat-SNV-FCK-AOMPLS-static` | `AOMPLSRegressor` | exploratory | slow |
| `AOMPLS-compact-with-fck-full57` | `AOMPLSRegressor` | exploratory | medium |
| `ASLS-FCK-PLS-static` | `PLSRegression` | exploratory | medium |
| `AOMRidgePLSCV-compact-with-fck` | `AOMRidgePLSCV` | exploratory | medium |
| `FCKResidual-AOMPLS-teacher` | `FCKResidualRegressor` | exploratory | medium |
| `paper-CNN-reference` | `PaperCNNReference` | locked | very_slow |
| `paper-CatBoost-reference` | `PaperCatBoostReference` | locked | medium |

## 16. Presets pratiques

| Preset | Intention | Candidats naturels |
|---|---|---|
| `fast_reliable` | Runs courts, references solides. | PLS/Ridge CV5, AOM-PLS compact/ASLS, eventuellement Ridge compact. |
| `strong_practical` | Bon compromis production. | AOM-PLS, AOM-Ridge locked, TabPFN si budget, MoE compact si nested. |
| `best_current` | Cherche le meilleur score deployable. | TabPFN-HPO, AOM-Ridge, AOM-PLS ASLS, MoE/ASL seulement apres gates. |
| `exhaustive_research` | Explorer la complementarite. | Ajoute FCK, residual CNN/FCK, ASL, blenders, auto-selectors, paper refs. |

## 17. Synthese decisionnelle

1. Le socle fiable reste `PLS/Ridge -> ASLS-AOM-PLS -> AOM-Ridge`.
2. TabPFN est le champion empirique actuel, mais son cout et son faible niveau
   d'interpretation imposent de le separer des baselines rapides.
3. AOM-PLS est le meilleur modele spectroscopique simple ; la version compacte
   est souvent preferable aux banques larges.
4. AOM-Ridge est la meilleure direction classique pour battre les references
   lineaires sans basculer dans un modele lourd.
5. Multiview/MoE montre une vraie complementarite, mais tout claim doit etre
   nested.
6. FCK est utile pour la diversite et l'analyse, pas pour un preset fort dans
   l'etat actuel.
7. CNN pur est clos comme chemin champion ; les reseaux n'ont du sens qu'en
   residuel, encoding ou pretraining strictement gate.
8. Les scores "oracle" et "bestof" doivent rester etiquetes comme optimistes
   tant que la selection n'est pas entierement dans la boucle train interne.

