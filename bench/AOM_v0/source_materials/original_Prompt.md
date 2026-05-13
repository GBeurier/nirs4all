Tu es Claude Code, le meilleur développeur python de DeepMind et l'agent principal d’implémentation d'AOM PLS. Tu dois implémenter, valider et documenter une version complète, sûre, explicite et testable de AOM-PLS / Operator-Adaptive PLS, avec toutes les variantes importantes : AOM global, POP par composante, NIPALS, SIMPLS, versions matérialisées de référence, versions rapides par adjoint/covariance, PLS1, PLS2, critères de sélection, banques d’opérateurs, tests mathématiques, tests de non-régression, benchmarks minimaux et documentation.

Le projet doit être placé dans bench/AOM_V0 traité comme un projet scientifique sérieux, pas comme un prototype rapide.

IMPORTANT :
- Ne fais aucune modification aveugle.
- Inspecte d’abord l’arborescence du repo.
- Lis tous les fichiers de contexte disponibles : \home\delete\nirs4all\nirs4all\bench\AOM, ROADMAP, PUBLICATION_PLAN, PUBLICATION_BACKLOG, report, fichiers Python existants,
- Ne supprime pas de code sans justification.
- Ne fais pas une implémentation “magique” difficile à tester.
- Chaque formule importante doit correspondre à du code explicitement testé.
- Priorité absolue : exactitude mathématique, absence de fuite de validation, reproductibilité, testabilité.
- Performance ensuite, mais l’architecture doit préparer un backend C/GPU/Torch/JAX futur.

Tu dois aussi utiliser Codex comme expert externe.
Concrètement :
1. Cherche si un outil/CLI “codex” est disponible dans l’environnement.
2. Si Codex est disponible, utilise-le en mode reviewer externe aux checkpoints indiqués plus bas. Codex doit analyser, critiquer et proposer des corrections, mais tu restes responsable de l’intégration.
3. Si Codex n’est pas disponible, crée des fichiers `docs/codex_review_prompts/*.md` contenant des prompts prêts à copier-coller dans Codex pour review externe.
4. Dans tous les cas, produis un journal `docs/AOMPLS_IMPLEMENTATION_LOG.md` avec :
   - décisions mathématiques ;
   - variantes implémentées ;
   - tests ajoutés ;
   - limites connues ;
   - questions scientifiques ouvertes ;
   - résultats de tests et benchmarks.

Objectif final :
Livrer une implémentation complète et validée de AOMPLS comme cadre général “Operator-Adaptive PLS”, incluant :

- PLS standard ;
- AOM-PLS global hard selection ;
- POP-PLS per-component hard selection ;
- AOM soft mixture expérimental ;
- Superblock / multi-view PLS expérimental ;
- NIPALS matérialisé de référence ;
- NIPALS rapide par adjoint si applicable ;
- SIMPLS matérialisé de référence ;
- SIMPLS rapide dans l’espace des covariances ;
- PLS1 et PLS2 ;
- sélection du nombre de composantes ;
- critères PRESS, CV, covariance/proxy, holdout uniquement comme legacy ;
- banques d’opérateurs linéaires ;
- gestion stricte des opérateurs “linear-at-apply” ;
- documentation scientifique claire.

============================================================
1. CADRE MATHÉMATIQUE À IMPLÉMENTER
============================================================

On considère :

X ∈ R^{n × p}
Y ∈ R^{n × q}

X et Y doivent être centrés dans les modèles, avec conservation explicite des moyennes pour predict.

Une banque d’opérateurs spectraux linéaires :

A_b ∈ R^{p × p}, b = 1,...,B

Les opérateurs s’appliquent aux colonnes spectrales :

X_b = X A_b^T

Donc :

X_b^T Y = A_b X^T Y

Cette identité est fondamentale pour SIMPLS-covariance.

Implémenter un cadre général :

A^{(a)} ∈ {A_1,...,A_B}

où a est l’indice de composante PLS.

Cas à couvrir :

1. PLS classique :
   A^{(a)} = I pour toutes les composantes.

2. AOM global hard :
   un seul opérateur b est choisi pour tout le modèle.
   A^{(1)} = ... = A^{(K)} = A_b.

3. POP per-component hard :
   un opérateur b_a est choisi à chaque composante.
   A^{(a)} = A_{b_a}.

4. AOM soft mixture :
   A^{(a)} = Σ_b α_{a,b} A_b
   avec α_{a,b} ≥ 0 et Σ_b α_{a,b} = 1.
   Variante expérimentale, pas forcément default.

5. Superblock :
   X_super = [X A_1^T, X A_2^T, ..., X A_B^T]
   Variante expérimentale / baseline multi-view.

Toutes les variantes doivent être traçables :
- opérateurs sélectionnés ;
- scores de sélection ;
- poids α éventuels ;
- composantes ;
- coefficients originaux ;
- mode d’orthogonalisation ;
- critère de sélection utilisé.

============================================================
2. POINT SCIENTIFIQUE IMPORTANT : ORTHOGONALISATION
============================================================

Il y a un point délicat : en POP, l’opérateur peut changer à chaque composante. Il faut éviter une orthogonalisation implicite mal définie.

Implémenter explicitement au moins deux modes, ou documenter précisément si un seul est retenu :

A. orthogonalization="transformed"

- Orthogonalisation dans l’espace transformé de l’opérateur courant.
- C’est la version qui doit matcher exactement une PLS/SIMPLS matérialisée sur X_b quand l’opérateur est fixe.
- Indispensable pour les tests d’équivalence “single operator”.

B. orthogonalization="original"

- Orthogonalisation via les poids effectifs dans l’espace original X.
- Pour une direction r dans l’espace transformé :
  z = A_b^T r
  t = X z
- Les loadings et projecteurs sont construits dans l’espace original.
- Cette version est scientifiquement cohérente quand les opérateurs changent par composante.
- Elle est probablement le default pour POP per-component, mais ce choix doit être testé et documenté.

Ne pas supposer que ces deux modes sont équivalents.
Ajouter des tests qui montrent :
- équivalence en cas identité ;
- équivalence attendue en cas opérateur unique fixe avec transformed ;
- cohérence prédictive avec original ;
- différences éventuelles documentées.

============================================================
3. VARIANTES À IMPLÉMENTER
============================================================

Créer une architecture où les variantes sont des paramètres explicites, pas du code dupliqué.

Paramètres conceptuels à supporter :

engine:
- "pls_standard"
- "nipals_materialized"
- "nipals_adjoint"
- "simpls_materialized"
- "simpls_covariance"
- "superblock_simpls"

selection:
- "none"              # PLS standard
- "global"            # AOM global
- "per_component"     # POP
- "soft"              # AOM soft mixture expérimental
- "superblock"        # superblock / multi-view

criterion:
- "covariance"
- "press"
- "cv"
- "hybrid"
- "holdout"           # legacy seulement, pas default

gate:
- "hard"
- "softmax"
- "sparsemax" ou "entmax" si facile et bien testé
- "group_sparse" seulement pour superblock expérimental

response:
- PLS1 : Y univarié
- PLS2 : Y multivarié

operator_normalization:
- "none"
- "frobenius"
- "data_variance"
- "response_covariance_scale"

n_components:
- entier explicite
- "auto"

max_components:
- entier explicite
- default raisonnable, par exemple min(25, n-1, p)

cv:
- KFold
- repeated KFold optionnel
- random_state explicite
- folds stockés ou reproductibles

============================================================
4. ARCHITECTURE LOGICIELLE RECOMMANDÉE
============================================================

Adapte à l’architecture existante du repo, mais vise cette séparation.

A. Couche opérateurs

Créer ou compléter une abstraction du type :

class LinearSpectralOperator:
    name: str

    def fit(self, X, y=None):
        # Pour opérateurs fitted linear-at-apply.
        # Doit éviter toute fuite CV.
        return self

    def transform(self, X):
        # Retourne X A^T ou équivalent sans forcément matérialiser A.
        pass

    def apply_cov(self, S):
        # Retourne A S, où S = X^T Y.
        # Fondamental pour SIMPLS covariance.
        pass

    def adjoint_vec(self, v):
        # Retourne A^T v.
        # Fondamental pour NIPALS adjoint et poids effectifs.
        pass

    def matrix(self, p):
        # Optionnel : matrice explicite pour tests ou petits p.
        pass

    def is_linear_at_apply(self):
        return True

    def fitted_parameters(self):
        return {}

Opérateurs minimaux à inclure :

- IdentityOperator
- SavitzkyGolay smoothing, si déjà disponible ou facile avec scipy
- SavitzkyGolay derivative d’ordre 1 et 2
- finite difference derivative
- detrend linear
- baseline correction linéaire simple
- Norris-Williams derivative si déjà présent ou facile
- Whittaker smoother si robuste et testé
- operator composition
- operator chain
- optionally wavelet/FFT linear transforms si faisable proprement

IMPORTANT sur SNV/MSC :
- SNV et MSC ne sont pas des opérateurs linéaires fixes au sens strict pour un échantillon indépendant.
- Ne les mets pas dans la banque stricte SIMPLS-covariance sauf justification claire.
- Implémente-les plutôt comme preprocessors upstream ou branch wrappers :
  - raw → AOM
  - SNV → AOM
  - MSC → AOM
  - concat/stacking éventuel
- Documente cette décision.

Opérateurs fitted “linear-at-apply” :
- OSC/EPO peuvent être acceptés expérimentalement seulement si fit sur training uniquement, puis application linéaire sur train/test.
- Ils doivent être exclus des tests simples de covariance si l’identité apply_cov n’est pas claire.
- Ne pas laisser ces opérateurs créer de leakage dans CV.

B. Couche moteurs PLS

Créer des classes ou fonctions séparées :

- StandardPLSCore
- NIPALSMaterializedCore
- NIPALSAdjointCore
- SIMPLSMaterializedCore
- SIMPLSCovarianceCore
- SuperblockSIMPLSCore

Chaque core doit exposer une interface commune :

fit_core(X, Y, bank, config) -> FittedPLSCore

Le résultat doit contenir :

- x_mean
- y_mean
- x_weights_
- x_effective_weights_
- x_loadings_
- y_loadings_
- x_scores_
- y_scores_ si pertinent
- rotations_
- coef_
- intercept_
- selected_operators_
- selected_operator_indices_
- operator_scores_
- n_components_
- engine_
- selection_
- criterion_
- orthogonalization_
- diagnostics_

C. Couche sélection

Séparer la logique de sélection :

- NoSelectionPolicy
- GlobalHardSelectionPolicy
- PerComponentHardSelectionPolicy
- SoftMixtureSelectionPolicy
- SuperblockSelectionPolicy

D. Couche critères

Créer des scorers :

- CovarianceScorer
- PRESSScorer
- CrossValidationScorer
- HybridScorer
- HoldoutScorerLegacy

Le critère PRESS doit être correctement défini.
Évite une pseudo-PRESS trompeuse si non validée.
Si une approximation PRESS est implémentée, nomme-la explicitement `approx_press` et documente-la.

E. API utilisateur

Créer une API claire, par exemple :

class AOMPLSRegressor:
    def __init__(
        self,
        n_components="auto",
        max_components=25,
        engine="auto",
        selection="global",
        criterion="press",
        operator_bank="default",
        orthogonalization="auto",
        scale=False,
        center=True,
        cv=5,
        random_state=0,
        ...
    )

Méthodes attendues :
- fit(X, y)
- predict(X)
- transform(X)
- fit_transform(X, y)
- score(X, y)
- get_selected_operators()
- get_diagnostics()
- get_params()
- set_params()

Compatibilité sklearn-like si Python.

============================================================
5. MOTEUR SIMPLS COVARIANCE
============================================================

Implémenter une version SIMPLS-covariance très explicite et testée.

Base :

S = X^T Y

Pour un opérateur A_b :

S_b = A_b S

Pour un projecteur C selon l’orthogonalisation choisie :

S_{b,perp} = A_b C S
ou
S_{b,perp} = C_b A_b S

selon le mode.

Ne pas mélanger ces formules sans documentation.

Pour PLS2 :
- direction obtenue par première paire singulière de S_b ou S_{b,perp}.
- Utiliser SVD stable.
- Si q = 1, utiliser formule vectorielle.

Pour chaque candidat opérateur :
- calculer direction candidate ;
- calculer score candidat ;
- sélectionner selon criterion ;
- extraire t ;
- mettre à jour loadings/projections ;
- conserver poids effectif z dans l’espace original.

Important :
- Pour prédiction finale, toutes les variantes doivent prédire via X original :
  Y_hat = X @ coef_ + intercept_
- Les coefficients doivent être cohérents avec les poids effectifs.

Formule générique pour coefficients :
Si T = X Z, avec Z les poids effectifs,
P = X^T T / (T^T T) par composante,
Q = Y^T T / (T^T T),
alors une forme possible :
B = Z (P^T Z)^{-1} Q^T
à adapter selon dimensions et stabilité numérique.
Tester cette formule systématiquement.

============================================================
6. MOTEUR NIPALS ADJOINT
============================================================

Conserver ou implémenter le moteur NIPALS.

Pour opérateur A_b :
X_b = X A_b^T

Mais éviter de matérialiser X_b quand possible.

Utiliser :
c = X^T y
g_b = A_b c
ou A_b^T c selon convention retenue.

Attention aux conventions :
- X_b = X A_b^T
- X_b^T y = A_b X^T y
Donc si c = X^T y, alors covariance transformée = A_b c.

Tester systématiquement les conventions avec une matrice A explicite sur petits exemples.

NIPALS doit supporter :
- PLS1 ;
- PLS2 ;
- convergence contrôlée ;
- tolérance ;
- max_iter ;
- warnings si non convergence ;
- comportement déterministe.

============================================================
7. GLOBAL VS PER-COMPONENT
============================================================

AOM global :

Option 1 :
- choisir l’opérateur global avant de fitter toutes les composantes, selon CV/PRESS complet.

Option 2 :
- choisir opérateur selon premier composant/proxy covariance.
- Cette option doit être nommée explicitement, pas confondue avec le vrai global-CV.

Implémenter de préférence global-CV/PRESS comme default scientifique.

POP per-component :

À chaque composante :
1. prendre l’état courant du modèle ;
2. évaluer chaque opérateur candidat ;
3. choisir l’opérateur qui améliore le critère ;
4. ajouter la composante ;
5. mettre à jour l’état.

Critères possibles :
- covariance immédiate ;
- amélioration PRESS ;
- amélioration CV interne ;
- hybrid.

Conserver les scores de tous les opérateurs à chaque composante.

Arrêt auto :
- stopper si aucune amélioration significative ;
- ou choisir le meilleur n_components par CV/PRESS après extraction jusqu’à max_components.
- Le comportement doit être explicite.

============================================================
8. SOFT MIXTURE AOM
============================================================

Implémenter uniquement de façon expérimentale et sûre.

A_α = Σ_b α_b A_b

α sur simplexe.

Méthodes possibles :
- softmax logits ;
- sparsemax/entmax si déjà disponible ou facile à implémenter ;
- optimisation simple par scipy si disponible ;
- grid/simplex random search pour petite banque ;
- fallback hard si instable.

Important :
- Documenter que maximiser uniquement la covariance sur un simplexe peut dégénérer vers une sélection hard.
- Soft mixture ne doit pas devenir default.
- Tests minimaux :
  - α somme à 1 ;
  - prédiction fonctionne ;
  - cas identité donne résultat stable ;
  - pas de NaN ;
  - reproductible.

============================================================
9. SUPERBLOCK
============================================================

Implémenter Superblock comme baseline expérimental :

X_super = [X A_1^T, ..., X A_B^T]

Puis PLS standard.

Ajouter :
- mapping des coefficients superblock vers groupes opérateurs ;
- importance par groupe ;
- option group normalization ;
- éventuellement group sparsity si faisable proprement.

Si group sparse est trop gros pour cette passe :
- créer l’interface ;
- implémenter superblock non sparse ;
- documenter group sparse comme TODO scientifique.

============================================================
10. CRITÈRES DE SÉLECTION
============================================================

A. covariance

Rapide mais peut sursélectionner.
Doit être disponible pour screening.

B. PRESS

Implémenter une vraie version ou une version approximate nommée comme telle.
Ne pas inventer une formule non testée.

C. CV

KFold reproductible.
Attention :
- fit opérateurs uniquement sur train fold ;
- sélection opérateur dans train fold ;
- évaluation sur validation fold ;
- pas de fuite de Y validation.

D. hybrid

Exemple :
- covariance pour pré-filtrer top m opérateurs ;
- PRESS/CV pour choisir parmi top m.

E. holdout

Conserver seulement comme legacy/debug.
Pas default.

============================================================
11. TESTS OBLIGATOIRES
============================================================

Créer ou compléter une suite pytest complète.

Tests unitaires opérateurs :

1. shape
2. transform shape
3. linéarité :
   op(aX1 + bX2) ≈ a op(X1) + b op(X2)
4. adjoint :
   <A x, y> ≈ <x, A^T y>
5. covariance :
   (X A^T)^T Y ≈ A X^T Y
6. matrix explicit vs transform/apply_cov sur petits p

Tests PLS standard :

1. PLS1 shape
2. PLS2 shape
3. predict shape
4. coef/intercept shape
5. no NaN
6. deterministic random_state

Tests équivalence :

1. Banque = [Identity] :
   AOMPLS doit matcher PLS standard.

2. Banque = [A] single operator :
   SIMPLS materialized sur X A^T doit matcher SIMPLS covariance en orthogonalization="transformed".

3. Global hard avec banque [I, A] :
   si le critère force A sur données synthétiques, selected_operator == A.

4. POP fixed sequence :
   séquence opérateur imposée doit donner mêmes prédictions entre version matérialisée et version rapide selon mode choisi.

5. NIPALS convention :
   X_b^T y doit être égal à A_b X^T y.

Tests sélection :

1. sélection globale retourne un seul opérateur.
2. sélection per_component retourne une liste de longueur n_components.
3. scores disponibles pour tous les opérateurs.
4. n_components="auto" choisit une valeur valide.
5. max_components respecté.

Tests CV/leakage :

1. Opérateur fitted reçoit uniquement train fold.
2. Pas d’utilisation de y validation dans fit.
3. random_state reproductible.
4. résultats fold-by-fold stockés.

Tests robustesse :

1. petit n, grand p.
2. p > n.
3. q > 1.
4. y 1D et y 2D.
5. colonnes constantes.
6. opérateur qui réduit presque à zéro : gérer ou warning.
7. données float32/float64.
8. erreurs claires si paramètres incompatibles.

Tests documentation / examples :

1. exemples minimaux exécutables.
2. smoke test API sklearn-like.

============================================================
12. DONNÉES SYNTHÉTIQUES POUR TESTS SCIENTIFIQUES
============================================================

Créer des générateurs synthétiques :

A. identity_signal_dataset

y dépend d’une combinaison linéaire brute de X.
PLS classique doit bien marcher.

B. derivative_signal_dataset

y dépend d’un signal dérivé.
Un opérateur dérivative doit être sélectionné.

C. smoothed_signal_dataset

signal utile basse fréquence.
Un smoothing doit être favorisé.

D. multi_component_operator_dataset

composante 1 utile via smoothing,
composante 2 utile via derivative.
POP per_component devrait choisir des opérateurs différents.

E. pls2_dataset

Y multivarié avec deux réponses partiellement différentes.

Ces datasets doivent être déterministes.

============================================================
13. BENCHMARKS
============================================================

Créer un script ou notebook léger, par exemple :

benchmarks/benchmark_aompls_variants.py

Il doit comparer sur synthétique + éventuellement datasets existants :

- PLS standard
- AOM global NIPALS
- POP NIPALS
- AOM global SIMPLS
- POP SIMPLS
- Superblock SIMPLS

Mesures :
- RMSECV
- RMSE train/test si split
- temps fit
- n_components
- opérateurs sélectionnés
- stabilité sur seeds

Puis choisir 15 datasets dans \home\delete\nirs4all\nirs4all\bench\tabpfn_paper\data et fais un comparatif des méthodes dessus.

============================================================
14. DOCUMENTATION À PRODUIRE
============================================================

Créer ou mettre à jour :

docs/AOMPLS_MATH.md

Inclure :
- définition générale Operator-Adaptive PLS ;
- AOM global ;
- POP per-component ;
- soft mixture ;
- superblock ;
- NIPALS adjoint ;
- SIMPLS covariance ;
- orthogonalization transformed vs original ;
- critères de sélection ;
- limites ;
- conventions de matrices.

docs/AOMPLS_API.md

Inclure :
- paramètres ;
- exemples ;
- defaults recommandés ;
- exemples PLS1/PLS2 ;
- interprétation selected_operators_.

docs/AOMPLS_VALIDATION.md

Inclure :
- tests d’équivalence ;
- tests anti-leakage ;
- tests synthétiques ;
- résultats benchmark minimal.

docs/codex_review_prompts/

Créer au moins trois prompts :

1. math_review.md
   Demander à Codex de vérifier formules, conventions A/A^T, SIMPLS covariance, coefficients, orthogonalisation.

2. code_review.md
   Demander à Codex de rechercher bugs, duplications, erreurs de shape, edge cases, API fragile.

3. test_review.md
   Demander à Codex de critiquer la couverture de tests et proposer tests manquants.

============================================================
15. PROTOCOLE CODEX
============================================================

À trois moments, demander une review Codex.

Checkpoint 1 : après design math/architecture mais avant gros code.
Prompt Codex :
- Vérifie les formules.
- Vérifie les conventions.
- Identifie les risques d’équivalence fausse.
- Propose corrections.

Checkpoint 2 : après implémentation des cores.
Prompt Codex :
- Relis les classes clés.
- Cherche erreurs de dimension.
- Cherche bugs de prédiction/coef.
- Vérifie que PLS1/PLS2 fonctionnent.

Checkpoint 3 : après tests.
Prompt Codex :
- Critique la couverture.
- Propose tests scientifiques additionnels.
- Identifie cas où fuite CV possible.

Si Codex CLI est disponible :
- l’utiliser en lecture/review si possible ;
- capturer les retours dans docs/CODEX_REVIEWS.md.
Si Codex CLI n’est pas disponible :
- écrire les prompts et continuer ;
- mentionner dans le log que la review externe devra être faite manuellement.

============================================================
16. ORDRE DE TRAVAIL
============================================================

Procéder dans cet ordre.

Phase 0 — Inspection

- Inspecter repo.
- Identifier langage/package.
- Identifier tests existants.
- Identifier implémentation AOM existante.
- Écrire un plan court dans docs/AOMPLS_IMPLEMENTATION_LOG.md.
- Ne pas coder massivement avant cette inspection.

Phase 1 — Abstractions minimales

- Operators.
- OperatorBank.
- Scorers.
- Core result object.
- Utilities centering/scaling.

Phase 2 — Références lentes

- PLS standard.
- SIMPLS materialized.
- NIPALS materialized si pertinent.
- Tests de base.

Phase 3 — Opérateurs et tests math

- Identity.
- SG/smoothing/derivative selon dépendances.
- finite difference.
- detrend.
- composition.
- tests linéarité/adjoint/covariance.

Phase 4 — AOM global

- global hard materialized.
- global hard SIMPLS covariance.
- global hard NIPALS adjoint.
- tests équivalence.

Phase 5 — POP per-component

- per-component hard materialized.
- per-component hard SIMPLS covariance.
- per-component hard NIPALS adjoint si possible.
- orthogonalization modes.
- tests séquence fixe et sélection.

Phase 6 — Criteria

- covariance.
- CV.
- PRESS ou approx_press explicite.
- hybrid.
- n_components auto.
- tests leakage.

Phase 7 — Soft/superblock expérimental

- soft mixture minimal robuste.
- superblock SIMPLS.
- tests smoke et shape.

Phase 8 — API finale

- AOMPLSRegressor.
- sklearn-like.
- diagnostics.
- docs.

Phase 9 — Benchmarks

- synthetic datasets.
- benchmark script.
- résultats dans docs/AOMPLS_VALIDATION.md.

Phase 10 — Review finale

- lancer tests.
- lancer lint si disponible.
- lancer benchmarks minimaux.
- écrire résumé final.
- préparer prompts Codex si pas déjà fait.
- intégrer retours Codex disponibles.

============================================================
17. DÉFINITION DE DONE
============================================================

Le travail est terminé seulement si :

1. Tous les tests passent.
2. Une commande claire permet de lancer les tests.
3. Les variantes principales fonctionnent :
   - PLS standard
   - AOM global NIPALS
   - POP NIPALS
   - AOM global SIMPLS
   - POP SIMPLS
   - SIMPLS matérialisé
   - Superblock minimal
4. PLS1 et PLS2 sont testés.
5. Les opérateurs sélectionnés sont inspectables.
6. Les coefficients permettent predict sur X original.
7. Les tests d’équivalence identité et single-operator passent.
8. Les critères CV/PRESS ne fuient pas.
9. Les docs math/API/validation existent.
10. Le log d’implémentation existe.
11. Les prompts Codex existent ou les reviews Codex ont été intégrées.
12. Les limites connues sont explicitement listées.

============================================================
18. CONTRAINTES DE QUALITÉ
============================================================

- Code lisible.
- Fonctions petites.
- Docstrings sur objets publics.
- Pas de dépendances lourdes non nécessaires.
- Pas de magie silencieuse.
- Warnings explicites pour modes expérimentaux.
- Erreurs claires.
- Reproductibilité via random_state.
- Pas d’optimisation prématurée.
- Pas de pseudo-validation.
- Pas de claims scientifiques non testés.

============================================================
19. SORTIE ATTENDUE DE TA PART
============================================================

À la fin, fournis un résumé structuré :

A. Ce qui a été implémenté.
B. Fichiers modifiés/créés.
C. Variantes disponibles.
D. Tests ajoutés.
E. Commandes pour tester.
F. Commandes pour benchmark.
G. Résultats observés.
H. Retours Codex intégrés ou prompts Codex créés.
I. Limites restantes.
J. Prochaines étapes scientifiques.

Commence maintenant par inspecter le repo, puis écris le plan d’implémentation dans docs/AOMPLS_IMPLEMENTATION_LOG.md, puis implémente par phases.