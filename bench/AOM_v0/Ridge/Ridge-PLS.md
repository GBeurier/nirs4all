# Roadmap AOM Ridge-PLS

Objectif : tester une vraie extension de **AOM-PLS** où les composantes PLS sont conservées, mais où la régression finale sur les scores est **régularisée par ridge**.

Je nommerais la méthode :

[
\boxed{\text{AOM-Ridge-PLS}}
]

ou dans le code :

```python
AOMRidgePLS
```

À distinguer clairement de :

```python
AOMMultiKernelRidge
```

qui est l’axe AOM-ridge / multi-kernel.

---

# 1. Idée générale

AOM-PLS actuel :

[
X \rightarrow Z_{\text{AOM}} \rightarrow \text{PLS}
]

avec :

[
Z_{\text{AOM}} = [Z_1, Z_2, \dots, Z_B]
]

où chaque bloc est un prétraitement ou opérateur AOM :

[
Z_b = T_b(X)
]

La PLS construit des scores supervisés :

[
T = Z_{\text{AOM}} R
]

puis fait une régression de (Y) sur ces scores.

L’idée Ridge-PLS est de remplacer la régression finale non pénalisée :

[
Y \approx TC
]

par une ridge :

[
\min_C |Y - TC|_F^2 + \lambda |C|_F^2
]

Donc :

[
\hat{C}_{\lambda}
=================

(T^\top T + \lambda I)^{-1}T^\top Y
]

et la prédiction devient :

[
\hat{Y}_*
=========

T_*\hat{C}_{\lambda}
]

avec :

[
T_* = Z_*R
]

---

# 2. Pourquoi c’est pertinent à tester

PLS régularise déjà par le **nombre de composantes** (H). Mais c’est une régularisation dure :

[
\text{garder } H \text{ composantes, jeter les autres}
]

Ridge-PLS ajoute une régularisation douce :

[
\text{garder les composantes, mais les shrinker}
]

Quand les scores PLS sont orthogonaux, on a :

[
T^\top T = D
]

avec :

[
D = \operatorname{diag}(d_1,\dots,d_H)
]

Alors, pour chaque composante :

[
\hat{c}_{h,\lambda}
===================

\frac{t_h^\top Y}{d_h+\lambda}
]

alors que PLS classique donne :

[
\hat{c}_{h,PLS}
===============

\frac{t_h^\top Y}{d_h}
]

Donc Ridge-PLS applique un facteur de shrinkage :

[
s_h
===

\frac{d_h}{d_h+\lambda}
]

Les composantes avec faible variance de score, souvent les composantes tardives et plus bruitées, sont davantage pénalisées.

C’est intéressant parce que ça transforme le choix brutal du nombre de composantes en une forme plus continue :

[
H_{\text{eff}}(\lambda)
=======================

\sum_{h=1}^{H}
\frac{d_h}{d_h+\lambda}
]

Donc tu peux tester des modèles avec plus de composantes que la PLS classique, tout en laissant ridge réduire leur contribution.

---

# 3. Formulation mathématique complète

## 3.1 Superbloc AOM

On construit :

[
Z = [s_1Z_1, s_2Z_2, \dots, s_BZ_B]
]

avec :

[
Z_b = T_b(X)
]

et (s_b) un facteur de pondération du bloc.

Par exemple :

[
s_b =
\frac{\sqrt{n}}{|Z_b|_F}
]

pour mettre les blocs sur une échelle comparable.

---

## 3.2 Centrage et scaling

On travaille sur :

[
\tilde{Z} = \text{scale}(Z)
]

[
\tilde{Y} = Y - \bar{Y}
]

Optionnellement :

[
\tilde{Y} = \frac{Y-\bar{Y}}{s_Y}
]

C’est important que tous ces scalings soient appris uniquement sur le train.

---

## 3.3 Décomposition PLS

PLS produit une matrice de rotations :

[
R_H \in \mathbb{R}^{q \times H}
]

et des scores :

[
T_H = \tilde{Z}R_H
]

où :

[
T_H \in \mathbb{R}^{n \times H}
]

En PLS classique :

[
\hat{B}_{PLS}
=============

R_H
(T_H^\top T_H)^{-1}
T_H^\top \tilde{Y}
]

---

## 3.4 Ridge-PLS

On remplace l’étape finale par :

[
\hat{C}_\lambda
===============

(T_H^\top T_H + \lambda I)^{-1}
T_H^\top \tilde{Y}
]

Puis :

[
\hat{B}_{RPLS}
==============

R_H\hat{C}_\lambda
]

La prédiction standardisée est :

[
\hat{\tilde{Y}}_*
=================

\tilde{Z}** \hat{B}*{RPLS}
]

ou équivalemment :

[
\hat{\tilde{Y}}_*
=================

T_*\hat{C}_\lambda
]

avec :

[
T_* = \tilde{Z}_*R_H
]

Ensuite on revient à l’échelle originale de (Y).

---

# 4. Les trois variantes à tester

## Variante 1 — AOM-Ridge-PLS simple

C’est le MVP.

Pipeline :

[
X
\rightarrow
Z_{\text{AOM}}
\rightarrow
\text{PLS scores}
\rightarrow
\text{Ridge on scores}
]

Hyperparamètres :

[
H = \text{nombre de composantes PLS}
]

[
\lambda = \text{pénalité ridge}
]

C’est la version à coder en premier.

---

## Variante 2 — AOM-Ridge-PLS avec (H_{\max}) élevé

Ici, tu choisis un nombre de composantes relativement large :

[
H_{\max}
]

puis tu laisses ridge shrinker les composantes inutiles.

Au lieu de chercher seulement :

[
H \in {1,2,\dots,20}
]

tu peux tester :

[
H_{\max} = 20, 30, 40
]

et laisser :

[
\lambda
]

contrôler le nombre effectif de composantes.

C’est conceptuellement intéressant parce que :

[
\text{PLS classique} = \text{sélection dure des composantes}
]

alors que :

[
\text{Ridge-PLS} = \text{sélection douce des composantes}
]

---

## Variante 3 — AOM-Ridge-PLS avec pénalité par composante

Version plus avancée.

Au lieu de :

[
\lambda I
]

on peut utiliser :

[
\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_H)
]

et résoudre :

[
\hat{C}
=======

(T^\top T + \Lambda)^{-1}T^\top Y
]

Par exemple :

[
\lambda_h = \lambda \cdot h^\rho
]

avec :

[
\rho \geq 0
]

Cela pénalise davantage les composantes tardives.

Mais je ne commencerais pas par ça. Il faut d’abord voir si la version simple apporte quelque chose.

---

# 5. Architecture informatique recommandée

Je ferais un estimateur unique :

```python
AOMRidgePLS
```

avec une structure compatible scikit-learn.

## Paramètres principaux

```python
AOMRidgePLS(
    blocks,
    n_components=10,
    ridge_alpha=1.0,
    block_scaling="frobenius",
    column_scaling=False,
    center_y=True,
    scale_y=False,
    pls_scale=False,
    score_scaling="none",
    max_iter=500,
    tol=1e-06
)
```

Recommandations par défaut :

```python
block_scaling="frobenius"
column_scaling=False
pls_scale=False
score_scaling="none"
```

Pourquoi `score_scaling="none"` ?

Parce que si tu standardises les scores PLS à variance 1, alors ridge shrinke toutes les composantes presque pareil. Or l’intérêt de Ridge-PLS est justement de shrinker davantage les composantes de faible variance :

[
\frac{d_h}{d_h+\lambda}
]

Donc, pour un premier test, je garderais les scores PLS dans leur échelle naturelle.

---

# 6. Pseudo-code du modèle

## Fit

```python
def fit(self, X, y):
    # 1. Construire le superbloc AOM sur train uniquement
    Z_blocks = []
    self.block_slices_ = []
    start = 0

    for block in self.blocks:
        Zb = block.fit_transform(X, y=None)

        if self.block_scaling == "frobenius":
            scale = np.sqrt(Zb.shape[0]) / np.linalg.norm(Zb, "fro")
            Zb = scale * Zb
            block.scale_ = scale

        stop = start + Zb.shape[1]
        self.block_slices_.append(slice(start, stop))
        start = stop

        Z_blocks.append(Zb)

    Z = np.concatenate(Z_blocks, axis=1)

    # 2. Centrer / scaler Z
    self.z_mean_ = Z.mean(axis=0)
    Zc = Z - self.z_mean_

    if self.column_scaling:
        self.z_scale_ = Zc.std(axis=0, ddof=1)
        self.z_scale_[self.z_scale_ == 0] = 1.0
    else:
        self.z_scale_ = np.ones(Z.shape[1])

    Zs = Zc / self.z_scale_

    # 3. Centrer / scaler y
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]

    self.y_mean_ = y.mean(axis=0)
    Yc = y - self.y_mean_

    if self.scale_y:
        self.y_scale_ = Yc.std(axis=0, ddof=1)
        self.y_scale_[self.y_scale_ == 0] = 1.0
    else:
        self.y_scale_ = np.ones(y.shape[1])

    Ys = Yc / self.y_scale_

    # 4. Fit PLS uniquement pour obtenir les rotations et scores
    self.pls_ = PLSRegression(
        n_components=self.n_components,
        scale=False,
        max_iter=self.max_iter,
        tol=self.tol
    )

    self.pls_.fit(Zs, Ys)

    T = self.pls_.x_scores_
    R = self.pls_.x_rotations_

    # 5. Ridge sur les scores
    H = self.n_components
    G = T.T @ T + self.ridge_alpha * np.eye(H)
    RHS = T.T @ Ys

    self.C_ = np.linalg.solve(G, RHS)

    # 6. Coefficients dans l'espace Z standardisé
    self.B_z_scaled_ = R @ self.C_

    return self
```

## Predict

```python
def predict(self, X):
    # 1. Transformer chaque bloc avec les paramètres appris sur train
    Z_blocks = []

    for block in self.blocks:
        Zb = block.transform(X)
        Zb = block.scale_ * Zb
        Z_blocks.append(Zb)

    Z = np.concatenate(Z_blocks, axis=1)

    # 2. Appliquer le scaling train
    Zs = (Z - self.z_mean_) / self.z_scale_

    # 3. Scores test
    T_test = Zs @ self.pls_.x_rotations_

    # 4. Prédiction standardisée
    Yhat_s = T_test @ self.C_

    # 5. Retour échelle originale
    Yhat = Yhat_s * self.y_scale_ + self.y_mean_

    if Yhat.shape[1] == 1:
        return Yhat.ravel()

    return Yhat
```

Point important : ne pas utiliser :

```python
self.pls_.predict(...)
```

car cela donnerait la prédiction PLS classique, pas la prédiction Ridge-PLS.

---

# 7. Backlog d’implémentation

## Phase 1 — Prototype minimal

Objectif : avoir une première version testable rapidement.

| ID       | Tâche                                             | Priorité |
| -------- | ------------------------------------------------- | -------: |
| RPLS-1.1 | Créer la classe `AOMRidgePLS`                     |    haute |
| RPLS-1.2 | Réutiliser les blocs AOM existants                |    haute |
| RPLS-1.3 | Construire le superbloc (Z_{\text{AOM}})          |    haute |
| RPLS-1.4 | Ajouter scaling de bloc par norme de Frobenius    |    haute |
| RPLS-1.5 | Fitter `PLSRegression` sur (Z)                    |    haute |
| RPLS-1.6 | Récupérer `x_scores_` et `x_rotations_`           |    haute |
| RPLS-1.7 | Implémenter ridge fermée sur les scores           |    haute |
| RPLS-1.8 | Implémenter `predict` sans utiliser `pls.predict` |    haute |

Critères d’acceptation :

* avec `ridge_alpha=0`, le modèle doit reproduire quasiment AOM-PLS ;
* avec `ridge_alpha>0`, les prédictions doivent changer ;
* `predict(X_train)` doit fonctionner pour (y) mono-sortie et multi-sortie.

---

## Phase 2 — Vérifications mathématiques

Objectif : s’assurer que le modèle fait bien ce qu’on croit.

| ID       | Tâche                                                  | Priorité |
| -------- | ------------------------------------------------------ | -------: |
| RPLS-2.1 | Vérifier que (T = ZR) numériquement                    |    haute |
| RPLS-2.2 | Vérifier que (T^\top T) est quasi diagonal             |  moyenne |
| RPLS-2.3 | Comparer PLS classique vs Ridge-PLS avec (\lambda=0)   |    haute |
| RPLS-2.4 | Calculer les facteurs de shrinkage (d_h/(d_h+\lambda)) |    haute |
| RPLS-2.5 | Calculer (H_{\text{eff}})                              |  moyenne |

Ajouter des attributs :

```python
model.score_sumsq_
model.shrinkage_factors_
model.effective_components_
```

avec :

```python
d = np.diag(T.T @ T)
shrink = d / (d + ridge_alpha)
h_eff = shrink.sum()
```

Ces diagnostics seront très utiles pour comprendre ce que ridge fait réellement.

---

## Phase 3 — Cross-validation interne

Objectif : sélectionner proprement (H) et (\lambda).

Hyperparamètres :

```python
n_components_grid = range(1, H_max + 1)
ridge_alpha_grid = np.logspace(...)
```

Mais comme l’échelle de (\lambda) dépend de (T^\top T), je recommande une grille relative.

Par exemple, pour chaque fit :

[
\lambda = r \cdot \operatorname{median}(d_h)
]

avec :

[
r \in 10^{[-4,\dots,4]}
]

En code, tu peux avoir :

```python
ridge_alpha_mode="absolute"
```

ou :

```python
ridge_alpha_mode="relative_to_score_variance"
```

Backlog :

| ID       | Tâche                                   | Priorité |
| -------- | --------------------------------------- | -------: |
| RPLS-3.1 | Ajouter `n_components_grid`             |    haute |
| RPLS-3.2 | Ajouter `ridge_alpha_grid`              |    haute |
| RPLS-3.3 | Ajouter scoring RMSE / R² / corrélation |    haute |
| RPLS-3.4 | Ajouter CV interne                      |    haute |
| RPLS-3.5 | Ajouter sélection par RMSE moyen        |    haute |
| RPLS-3.6 | Stocker `cv_results_`                   |  moyenne |
| RPLS-3.7 | Ajouter grille relative de (\lambda)    |  moyenne |

Critère important : dans la CV interne, il faut refitter toute la chaîne dans chaque fold :

[
\text{AOM transforms} \rightarrow \text{PLS} \rightarrow \text{ridge}
]

Ne pas pré-calculer le superbloc ou les composantes PLS sur tout le train avant la CV interne.

---

## Phase 4 — Comparaison aux baselines

Objectif : savoir si Ridge-PLS apporte vraiment quelque chose.

Comparer au minimum :

| Modèle                     | Description                        |
| -------------------------- | ---------------------------------- |
| AOM-PLS                    | baseline directe                   |
| AOM-Ridge                  | ton modèle ridge actuel            |
| AOM-Ridge-PLS              | nouveau modèle                     |
| Ridge sur superbloc AOM    | ridge directe sur (Z_{\text{AOM}}) |
| PLS sur meilleur bloc seul | contrôle utile                     |

Scénarios à tester :

1. même split externe pour tous les modèles ;
2. même liste de blocs AOM ;
3. mêmes métriques ;
4. même traitement de (Y) ;
5. CV interne pour les hyperparamètres.

Backlog :

| ID       | Tâche                                     | Priorité |
| -------- | ----------------------------------------- | -------: |
| RPLS-4.1 | Créer benchmark commun                    |    haute |
| RPLS-4.2 | Ajouter AOM-PLS baseline                  |    haute |
| RPLS-4.3 | Ajouter AOM-Ridge baseline                |    haute |
| RPLS-4.4 | Ajouter Ridge superbloc                   |  moyenne |
| RPLS-4.5 | Produire tableau par fold                 |    haute |
| RPLS-4.6 | Produire performance moyenne ± écart-type |    haute |

---

## Phase 5 — Interprétation des composantes

Objectif : comprendre quels blocs AOM participent aux composantes Ridge-PLS.

La rotation PLS est :

[
R_H
]

On peut regarder, pour chaque bloc (b), la norme des rotations dans le slice correspondant :

[
I_{b,h}
=======

\frac{
|R_{b,h}|^2
}{
\sum_j |R_{j,h}|^2
}
]

Puis pondérer par le shrinkage ridge :

[
I^{ridge}_{b,h}
===============

I_{b,h}
\cdot
\frac{d_h}{d_h+\lambda}
]

Cela donne une importance de bloc par composante.

Backlog :

| ID       | Tâche                                 | Priorité |
| -------- | ------------------------------------- | -------: |
| RPLS-5.1 | Stocker les slices des blocs          |    haute |
| RPLS-5.2 | Calculer importance bloc × composante |  moyenne |
| RPLS-5.3 | Pondérer par shrinkage ridge          |  moyenne |
| RPLS-5.4 | Agréger importance globale par bloc   |  moyenne |
| RPLS-5.5 | Produire heatmap bloc × composante    |    basse |

Sorties utiles :

```python
model.block_component_importance_
model.block_importance_
model.component_shrinkage_
```

---

## Phase 6 — Back-projection des coefficients

Objectif : récupérer un coefficient dans l’espace du superbloc, puis éventuellement dans l’espace spectral original.

Coefficient dans l’espace (Z) standardisé :

[
\hat{B}_{Z,std}
===============

R_H\hat{C}_\lambda
]

Pour revenir à l’espace (Z) non standardisé :

[
\hat{B}_{Z,j}
=============

\frac{\hat{B}_{Z,std,j}}{s_j}
]

où (s_j) est l’écart-type de la colonne (j), si autoscaling.

Ensuite, pour chaque bloc :

[
\hat{B}_b
]

est extrait via son slice.

Si le bloc est linéaire :

[
Z_b = X A_b^\top
]

alors on peut revenir vers l’espace spectral original :

[
\hat{\beta}_b
=============

s_b A_b^\top \hat{B}_b
]

et sommer :

[
\hat{\beta}
===========

\sum_b \hat{\beta}_b
]

Backlog :

| ID       | Tâche                                        | Priorité |
| -------- | -------------------------------------------- | -------: |
| RPLS-6.1 | Calculer `coef_superblock_`                  |    haute |
| RPLS-6.2 | Découper les coefficients par bloc           |    haute |
| RPLS-6.3 | Ajouter méthode `coef_by_block()`            |  moyenne |
| RPLS-6.4 | Ajouter back-projection pour blocs linéaires |  moyenne |
| RPLS-6.5 | Documenter les blocs non back-projectables   |  moyenne |

Attention : SNV, MSC, EMSC ou certains prétraitements dépendants des individus peuvent rendre la back-projection moins directe.

---

# 8. Protocole expérimental recommandé

## 8.1 Grille initiale

Je commencerais par :

```python
n_components_grid = [2, 3, 4, 5, 7, 10, 15, 20, 30]
```

en respectant :

[
H < n_{\text{train}}
]

Pour ridge :

```python
ridge_alpha_factors = np.logspace(-4, 4, 25)
```

avec :

[
\lambda = r \cdot \operatorname{median}(\operatorname{diag}(T^\top T))
]

C’est plus stable qu’une grille absolue.

---

## 8.2 Deux stratégies à comparer

### Stratégie A — PLS classique régularisée

Tu tunes :

[
H
]

et :

[
\lambda
]

en même temps.

C’est la comparaison directe à AOM-PLS.

---

### Stratégie B — Beaucoup de composantes + ridge

Tu fixes un (H_{\max}) plus grand :

```python
H_max = 30
```

ou :

```python
H_max = min(40, n_train - 2)
```

et tu tunes surtout :

[
\lambda
]

Cette stratégie teste vraiment l’idée :

[
\text{ridge remplace partiellement le choix dur du nombre de composantes}
]

---

# 9. Résultats à inspecter

Pour chaque fold externe, stocker :

```python
fold
trait
method
rmse
r2
corr
mae
n_components
ridge_alpha
effective_components
```

Pour Ridge-PLS, ajouter :

```python
component_shrinkage
score_sumsq
block_importance
```

Ce que tu veux observer :

## Cas favorable

Ridge-PLS est intéressant si :

```text
AOM-Ridge-PLS > AOM-PLS
```

et si :

```text
ridge_alpha > 0
```

avec un nombre de composantes effectif raisonnable :

[
H_{\text{eff}} < H
]

Cela signifie que ridge exploite des composantes supplémentaires sans les laisser surajuster.

---

## Cas neutre

Si la meilleure valeur est :

[
\lambda \approx 0
]

alors Ridge-PLS retombe sur AOM-PLS.

Conclusion : AOM-PLS suffit.

---

## Cas défavorable

Si Ridge-PLS est instable ou moins bon, les causes possibles sont :

1. scaling de blocs incorrect ;
2. trop de composantes PLS ;
3. CV interne trop petite ;
4. ridge trop fort ;
5. PLS déjà suffisamment régularisée par (H).

---

# 10. Tests unitaires prioritaires

| Test                            | Objectif                                    |
| ------------------------------- | ------------------------------------------- |
| `test_lambda_zero_matches_pls`  | Ridge-PLS avec (\lambda=0) ≈ PLS            |
| `test_predict_shape`            | forme correcte mono/multi-output            |
| `test_no_pls_predict_used`      | vérifier que le modèle utilise bien ridge   |
| `test_block_scaling_no_leakage` | scaling appris uniquement sur train         |
| `test_score_shrinkage`          | shrinkage (d/(d+\lambda)) correct           |
| `test_components_sum`           | prédiction via (T C) = prédiction via (Z B) |
| `test_cv_selects_alpha`         | la CV interne sélectionne bien (\lambda)    |

---

# 11. Structure de fichiers possible

```text
aom/
  blocks/
    base.py
    spectral.py
    derivatives.py
    corrections.py

  transformers/
    superblock.py
    scaling.py

  models/
    aom_pls.py
    aom_ridge_pls.py
    aom_ridge.py
    aom_multikernel_ridge.py

  validation/
    nested_cv.py
    metrics.py
    grids.py

  diagnostics/
    pls.py
    block_importance.py
    plots.py

  tests/
    test_aom_ridge_pls.py
    test_ridge_pls_equivalence.py
    test_block_scaling.py
```

---

# 12. Ordre de développement conseillé

## Sprint 1 — Version testable

Livrable :

```python
AOMRidgePLS(n_components=H, ridge_alpha=lambda_)
```

Capable de :

```python
fit(X_train, y_train)
predict(X_test)
```

avec une liste de blocs AOM.

---

## Sprint 2 — CV hyperparamètres

Livrable :

```python
AOMRidgePLSCV(
    n_components_grid=...,
    ridge_alpha_factors=...
)
```

avec :

```python
model.best_n_components_
model.best_ridge_alpha_
model.cv_results_
```

---

## Sprint 3 — Diagnostics

Livrable :

```python
model.shrinkage_factors_
model.effective_components_
model.block_importance_
```

Cela permettra de savoir si ridge fait vraiment quelque chose.

---

## Sprint 4 — Benchmark

Livrable :

```text
AOM-PLS vs AOM-Ridge-PLS vs AOM-Ridge
```

sur les mêmes folds externes.

---

## Sprint 5 — Interprétation avancée

Livrable :

```python
model.coef_by_block()
model.predict_components_by_block()
```

pour analyser les contributions des opérateurs AOM.

---

# 13. Décision de go / no-go

Je garderais Ridge-PLS dans la suite si au moins une de ces conditions est vraie :

1. il améliore la prédiction externe par rapport à AOM-PLS ;
2. il réduit la variance entre folds ;
3. il permet d’utiliser plus de composantes avec un (H_{\text{eff}}) plus stable ;
4. il donne des résultats proches de AOM-PLS mais plus robustes ;
5. il améliore certains traits difficiles même si le gain moyen est modéré.

Je l’abandonnerais comme priorité si :

1. (\lambda) sélectionné est presque toujours nul ;
2. les performances sont identiques à AOM-PLS ;
3. le modèle devient plus instable ;
4. l’interprétation par bloc n’apporte rien de plus.

---

# 14. Résumé opérationnel

Le premier modèle à coder est :

[
\boxed{
\hat{C}_{\lambda}
=================

(T^\top T+\lambda I)^{-1}T^\top Y
}
]

avec :

[
T = Z_{\text{AOM}}R_H
]

et :

[
Z_{\text{AOM}} = [s_1Z_1,\dots,s_BZ_B]
]

La classe doit faire :

```python
AOM transforms
→ superblock scaling
→ PLS fit
→ ridge on PLS scores
→ prediction
→ diagnostics
```

Le test clé :

```text
ridge_alpha = 0  →  AOM-PLS
ridge_alpha > 0  →  shrinkage des composantes PLS
```

