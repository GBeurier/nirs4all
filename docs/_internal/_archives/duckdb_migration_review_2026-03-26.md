# Review des documents de migration DuckDB → SQLite

Date: 2026-03-26
Documents reviewés:
- `duckdb_to_sqlite_migration_plan.md` (ci-après "Plan")
- `duckdb_backup_prediction_review_2026-03-25.md` (ci-après "Analyse")

---

## 1. Vérification factuelle du Plan de migration

### 1.1 Inventaire des fichiers — Partiellement correct

Le tableau du Plan est globalement juste mais incomplet.

| Affirmation du Plan | Vérifié | Commentaire |
|---|---|---|
| `store_schema.py` contient le DDL | Vrai | Lignes 37-159 pour SCHEMA_DDL, plus INDEX_DDL, VIEW_DDL |
| `store_queries.py` contient toutes les requêtes SQL | **Partiellement faux** | Beaucoup de SQL est aussi directement dans `workspace_store.py` (lignes 671-696, 806-835, etc.) et dans `store_schema.py` (migrations lignes 343-552) |
| `workspace_store.py` gère `duckdb.connect()` | Vrai | Ligne 231 |
| `array_store.py` n'a pas de dépendance DuckDB | Vrai | Confirmé: imports uniquement numpy, polars, pyarrow |
| `predictions.py` utilise uniquement l'API WorkspaceStore | Vrai | Aucun import duckdb direct |
| `orchestrator.py` crée le WorkspaceStore | Vrai | |

**Fichier manquant**: `migration.py` dans `pipeline/storage/` importe aussi duckdb directement. Le Plan ne le mentionne pas.

**Fichier manquant**: `cli/commands/workspace.py` référence aussi le workspace store.

### 1.2 Numéros de lignes — Obsolètes

Les numéros de lignes du Plan ne correspondent plus au code actuel:

| Référence Plan | Code actuel |
|---|---|
| `workspace_store.py:250` (duckdb.connect) | Ligne 231 |
| `workspace_store.py:253` (PRAGMA) | Lignes 234-239 |
| `workspace_store.py:247` (RLock) | Ligne 228 |

Le code a manifestement évolué depuis la rédaction du Plan. Les numéros de l'Analyse sont plus proches mais aussi légèrement décalés par endroits.

### 1.3 Features DuckDB utilisées — Sous-estimées dans le Plan

Le Plan affirme: *"The DuckDB usage is essentially relational CRUD. There are no columnar analytics or DuckDB-specific query features that would be hard to port."*

**C'est inexact.** Voici les features DuckDB-spécifiques réellement utilisées et absentes de l'inventaire du Plan:

| Feature DuckDB | Fichier | Équivalent SQLite |
|---|---|---|
| `FIRST()` (agrégat) | store_schema.py:354-372, workspace_store.py:671-696, 806-835 | **N'existe pas**. Nécessite un sous-query ou `MIN()`/ruse |
| `unnest()` | store_queries.py:299 | **N'existe pas nativement**. Nécessite `json_each()` + refactoring |
| `list_concat()` | store_queries.py:300 | **N'existe pas**. Logique à réécrire en Python ou via `json_group_array()` |
| `json_keys()` | store_queries.py:301 | **N'existe pas**. `json_each()` retourne clés+valeurs mais syntaxe différente |
| `::VARCHAR[]` (type casting) | store_queries.py:301-302 | **N'existe pas**. Pas de type array en SQLite |
| `duckdb_indexes()` | store_schema.py:520 | `PRAGMA index_list(table)` + `PRAGMA index_info(index)` |
| `.pl()` (zero-copy Arrow → Polars) | workspace_store.py:339 | **N'existe pas**. Il faudra `pl.from_pandas(cursor.fetchall())` ou similaire |
| `datasets::VARCHAR LIKE` (cast inline) | workspace_store.py:1427 | `CAST(datasets AS TEXT) LIKE` |
| `SET memory_limit`, `SET threads`, `SET checkpoint_threshold` | workspace_store.py:237-239 | **N'existent pas.** Pas d'équivalent, géré par le process OS |
| `ROW_NUMBER() OVER (PARTITION BY ...)` | store_queries.py:549, store_schema.py:545 | **Existe en SQLite 3.25+** — OK |
| `ARRAY[]::VARCHAR[]` (littéral array vide) | store_queries.py:301 | **N'existe pas** |

Le Plan affirme aussi *"Window functions (beyond basic SQL) — NOT used"*. C'est **faux**: `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` est utilisé dans les requêtes de déduplication et de ranking. Heureusement, SQLite 3.25+ supporte les window functions, donc ce n'est pas bloquant, mais c'est une erreur factuelle dans l'inventaire.

### 1.4 Style de paramétrage — Correct mais sous-estimé en effort

Le Plan dit: *"Adjust query syntax in store_queries.py (parameter style: DuckDB uses $1, SQLite uses ?). Grep and replace. ~1 hour."*

Vérifié: **100% des requêtes utilisent le style `$1, $2, $3...`** (style PostgreSQL, natif DuckDB).

Cependant, l'effort est sous-estimé:
- `store_queries.py` contient des **query builders dynamiques** qui construisent des paramètres `${idx}` avec index incrémenté (lignes 327+, 493+, 577+)
- `workspace_store.py` contient aussi **~15 requêtes SQL en dur** avec `$1` (lignes 671-696, 806-835, 1427, etc.)
- `store_schema.py` contient des requêtes paramétrées dans les migrations (ligne 435)
- **Ce n'est pas un simple grep/replace** — les query builders doivent être refactorés pour utiliser `?` avec des listes de paramètres positionnels au lieu de `$N` nommés

Estimation réaliste: **2-3 heures** pour les paramètres seuls, sans compter la réécriture des fonctions DuckDB-spécifiques.

### 1.5 Estimation globale du scope — Optimiste

| Estimation Plan | Estimation révisée | Raison |
|---|---|---|
| `store_schema.py`: ~30 min | ~1h | Types OK, mais les migrations utilisent FIRST(), unnest(), etc. |
| `store_queries.py`: ~1h | ~3-4h | Query builders dynamiques + fonctions DuckDB-spécifiques à réécrire |
| `workspace_store.py`: ~2-3h | ~4-6h | SQL inline (FIRST() x15+), .pl() → conversion manuelle, SET pragmas, duckdb_indexes() |
| Tests: ~1h | ~2-3h | Tests du store + tests d'intégration touchant les queries |
| **Total Plan: ~5-6h** | **~12-16h** | Le Plan sous-estime d'un facteur ~2.5x |

### 1.6 Outil de migration de données — Correct dans le principe

La proposition `migrate_duckdb_to_sqlite()` dans le Plan est raisonnable. À noter qu'un outil de migration existe déjà (`migration.py`): `migrate_arrays_to_parquet()`, qui a déjà migré les arrays de DuckDB vers Parquet. Le pattern est donc connu et éprouvé dans le codebase.

### 1.7 Interaction avec l'issue #36 — Déjà partiellement résolue

Le Plan recommande d'attendre la migration SQLite avant d'implémenter le stockage de metadata par sample (issue #36). Or, le commit `a7028d2` du 25 mars a **déjà implémenté** le stockage per-sample metadata dans Parquet via une colonne `sample_metadata` (JSON sérialisé en UTF-8).

La recommandation du Plan reste pertinente dans l'esprit (éviter le double schema migration), mais la réalité a avancé: la metadata est déjà stockée côté Parquet, pas côté DuckDB. Cela signifie que cette partie de la migration est de facto neutre vis-à-vis du changement DuckDB → SQLite.

---

## 2. Vérification factuelle de l'Analyse

### 2.1 Architecture de persistance — Correct

La description des 3 couches (DuckDB index + Parquet arrays + artifacts filesystem) est exacte et bien documentée.

### 2.2 Durée de vie des connexions — Correct et bien diagnostiqué

L'Analyse identifie correctement le problème central: `WorkspaceStore` ouvre une connexion RW dès `__init__` (ligne 231), et cette connexion reste vivante via la chaîne `RunResult → runner → orchestrator → WorkspaceStore`.

Vérifié dans le code:
- `RunResult` conserve bien une référence au `runner` ([result.py](nirs4all/api/result.py))
- `PipelineRunner` conserve l'orchestrator
- L'orchestrator conserve le `WorkspaceStore`

### 2.3 Retry-on-lock — Correct

L'Analyse note justement que `_retry_on_lock` attrape `duckdb.TransactionException` mais pas `duckdb.IOException` qui survient à la connexion elle-même. Vérifié: le décorateur (lignes 117-149) ne gère que `TransactionException`.

### 2.4 Lecture des prédictions inefficace — Correct

`_populate_buffer_from_store()` fait bien un `load_single()` par prédiction, ce qui rescanne le fichier Parquet à chaque fois. L'amélioration par batch groupé par dataset est évidente et justifiée.

### 2.5 Écriture non-atomique DuckDB/Parquet — Correct

Le flush fait séquentiellement `save_prediction()` (DuckDB) puis `save_batch()` (Parquet). Un crash entre les deux laisse effectivement un état incohérent. Diagnostic exact.

### 2.6 ArrayStore réécrit le fichier complet — Correct

`save_batch()` lit tout le Parquet existant, concatène, puis réécrit via temp+rename. C'est le comportement observé dans le code (array_store.py lignes 242-249 environ).

### 2.7 Version DuckDB — À vérifier

L'Analyse mentionne `duckdb==1.4.4` comme version locale testée. Le `pyproject.toml` spécifie `duckdb>=1.0.0`. La version 1.4.4 n'existe pas encore publiquement (la dernière stable connue est ~1.2.x en mars 2026). C'est probablement une version de développement locale ou une erreur de transcription.

### 2.8 Propositions de l'Analyse — Pertinentes

Les phases A/B/C proposées sont bien structurées:
- **Phase A** (corrections immédiates): détacher les connexions, mode read-only, batch loading — tout est faisable indépendamment de la migration
- **Phase B** (fiabilisation DuckDB): retry sur connect(), batch metadata, verrou inter-processus — correctifs de durée de vie
- **Phase C** (migration SQLite): abstraction WorkspaceIndex → SQLiteWorkspaceIndex

L'alternative "runs immuables sur disque + index rebuildable" est intéressante mais probablement trop radicale pour le moment.

---

## 3. Cohérence entre les deux documents

Les deux documents sont globalement cohérents entre eux. Points de convergence:
- Diagnostic identique sur le problème de lock
- Même recommandation de migrer vers SQLite WAL
- Même analyse du pattern d'écriture OLTP mal aligné avec DuckDB

Points de divergence:
- Le Plan est plus optimiste sur l'effort de migration
- L'Analyse est plus nuancée et identifie des quick-wins intermédiaires (Phase A/B)
- Le Plan ne mentionne pas les features DuckDB avancées (FIRST, unnest, etc.) que l'Analyse n'inventorie pas non plus de façon exhaustive

---

## 4. Critique du choix de migration DuckDB → SQLite WAL

### 4.1 Arguments en faveur de la migration

**Le diagnostic est juste.** DuckDB n'est pas conçu pour être un store opérationnel "live" avec des connexions long-lived en contexte multi-processus. Les arguments du Plan et de l'Analyse sont tous vérifiés:

1. **Locking exclusif**: DuckDB impose un writer exclusif au niveau fichier. Même un `read_only=True` échoue quand un writer est connecté. C'est structurel et documenté par DuckDB.

2. **Pattern OLTP**: Le workload réel est constitué de petites transactions séquentielles (begin_run, save_chain, save_prediction x N, complete_run). DuckDB est optimisé pour le bulk analytique, pas pour ça.

3. **Durée de vie des connexions**: Le design actuel garde des connexions RW vivantes bien au-delà du besoin transactionnel. C'est un problème de design applicatif, mais DuckDB le rend fatal là où SQLite WAL le tolérerait.

4. **Suppression d'une dépendance externe**: `sqlite3` est dans la stdlib Python. DuckDB est une dépendance binaire de ~50 MB qui peut poser des problèmes de compilation sur certaines plateformes.

5. **SQLite WAL**: Le mode WAL permet des readers concurrents pendant qu'un writer écrit. C'est exactement le modèle dont nirs4all a besoin (un writer principal + des readers pour l'exploration, les notebooks, la webapp).

### 4.2 Arguments contre la migration — Risques réels

**1. Perte de features SQL avancées**

Le Plan sous-estime ce point. Les fonctions suivantes n'ont pas d'équivalent direct en SQLite:
- `FIRST()` est utilisé **~15 fois** dans le code. SQLite n'a pas cet agrégat. Il faudra soit utiliser un sous-query `(SELECT col FROM t WHERE ... LIMIT 1)`, soit l'astuce `MIN()`/`MAX()` quand l'ordre n'importe pas, soit créer une fonction SQL custom via `sqlite3.create_aggregate()`.
- `unnest()` + `list_concat()` + `json_keys()` + `::VARCHAR[]` sont utilisés ensemble dans `GET_CHAIN_ARTIFACT_IDS` pour extraire les clés d'artifacts depuis les colonnes JSON. En SQLite, il faudra réécrire cette logique via `json_each()` avec une syntaxe radicalement différente.
- `.pl()` (zero-copy DuckDB → Polars via Arrow) n'a pas d'équivalent performant en SQLite. Les résultats devront être convertis manuellement, avec potentiellement un impact sur la performance des grosses requêtes.

**2. Perte potentielle de performance pour les requêtes analytiques**

Le `v_chain_summary` VIEW avec ses agrégations GROUP BY + AVG + COUNT(DISTINCT) est typiquement ce que DuckDB fait très bien. SQLite peut le faire, mais sera plus lent sur de gros workspaces (10K+ chains). Cependant, ce cas d'usage reste marginal pour la plupart des utilisateurs.

**3. Risque de régression de la migration elle-même**

50+ requêtes SQL à réécrire, dont des query builders dynamiques. Chaque réécriture est une source potentielle de bug subtil (ordre de résultats différent, gestion des NULL, sémantique des types, etc.).

**4. Le problème réel est partiellement indépendant du moteur DB**

Une partie significative du problème (durée de vie des connexions, store non-détaché après run) est un problème de design applicatif qui peut être résolu sans changer de moteur. La Phase A de l'Analyse le montre bien: fermer le store avant de retourner le RunResult, détacher les Predictions, mode read-only — tout cela réduit dramatiquement la douleur sans migration.

### 4.3 Alternatives non évaluées

**1. DuckDB avec pattern open/close court**

DuckDB supporte très bien le pattern: ouvrir → exécuter → fermer. Si le code est refactoré pour ne jamais garder de connexion long-lived (Phase A+B de l'Analyse), le problème de locking disparaît. Le coût de reconnexion DuckDB est de l'ordre de quelques millisecondes (contrairement à PostgreSQL).

Avantage: **zéro réécriture SQL**, conserve toutes les features avancées.

Inconvénient: ne résout pas le cas "deux process essaient de se connecter en même temps en écriture". Mais ce cas est gérable avec un lock fichier simple (`fcntl.flock` ou `filelock`).

**2. DuckDB read_only pour les lectures**

Ouvrir avec `duckdb.connect(path, read_only=True)` pour toutes les opérations de lecture (Predictions.from_workspace, queries exploratoires). Réserver le mode RW pour les écritures transactionnelles courtes.

Limite connue: un reader `read_only=True` ne peut pas coexister avec un writer RW dans le même fichier DuckDB. Mais il peut coexister avec d'autres readers `read_only=True`.

**3. Migration vers SQLite avec couche de compatibilité DuckDB optionnelle**

Si la migration est décidée, garder la possibilité d'utiliser DuckDB en mode lecture analytique par-dessus les fichiers SQLite + Parquet, comme le suggère l'Analyse. DuckDB peut scanner des fichiers Parquet et interroger des bases SQLite via son extension `sqlite`.

### 4.4 Verdict sur le choix de migration

**La migration vers SQLite WAL est justifiée** mais le Plan sous-estime l'effort et les risques.

Recommandation:

1. **Commencer par la Phase A** de l'Analyse (corrections de durée de vie des connexions). Cela résout 80% de la douleur utilisateur sans aucune migration SQL. C'est le meilleur rapport effort/impact.

2. **Si la Phase A ne suffit pas**, procéder à la migration SQLite en budgétant **12-16 heures** de travail, pas 5-6 comme estimé dans le Plan.

3. **Avant de commencer la migration SQL**, inventorier exhaustivement les ~15 usages de `FIRST()`, la requête `GET_CHAIN_ARTIFACT_IDS` (unnest/list_concat/json_keys), et le pattern `.pl()`. Préparer les équivalents SQLite de chacun.

4. **La migration des données** (store.duckdb → store.sqlite) est le point le moins risqué: le pattern est déjà éprouvé avec `migrate_arrays_to_parquet()`.

---

## 5. Corrections à apporter aux documents

### Plan (`duckdb_to_sqlite_migration_plan.md`)

| Section | Correction |
|---|---|
| Ligne 37 | `workspace_store.py:250` → `workspace_store.py:231` |
| Ligne 38 | `workspace_store.py:253` → `workspace_store.py:234-239` |
| Ligne 48 | `workspace_store.py:247` → `workspace_store.py:228` |
| Ligne 47 | Paramètres DuckDB: `$1` (pas `$1 or ?`) — seul `$1` est utilisé |
| Tableau "DuckDB-specific features NOT used" | Retirer "Window functions" — `ROW_NUMBER() OVER (PARTITION BY...)` est utilisé dans store_queries.py:549 et store_schema.py:545 |
| Tableau "DuckDB-specific features used" | Ajouter: `FIRST()` (agrégat, ~15 usages), `unnest()`, `list_concat()`, `json_keys()`, `::VARCHAR[]` (type casting), `duckdb_indexes()`, `.pl()` (zero-copy Arrow), `SET memory_limit/threads/checkpoint_threshold` |
| Section "File changes estimate" | Revoir les estimations à la hausse (facteur ~2.5x). Mentionner le SQL inline dans workspace_store.py et store_schema.py |
| Section "Interaction with Issue #36" | Mettre à jour: la metadata per-sample est déjà implémentée dans Parquet (commit a7028d2, 2026-03-25). Le point est de facto neutre |
| Section "Key insight" | Reformuler: "L'usage est principalement CRUD, mais utilise aussi des fonctions DuckDB-spécifiques (FIRST, unnest, json_keys, list_concat, type casting) qui nécessitent une réécriture non-triviale pour SQLite" |

### Analyse (`duckdb_backup_prediction_review_2026-03-25.md`)

| Section | Correction |
|---|---|
| Ligne 8 | `duckdb==1.4.4` — Vérifier: cette version n'est pas publiquement connue. Le pyproject.toml spécifie `duckdb>=1.0.0` |
| Section "Problèmes" | Ajouter un point sur les features DuckDB-spécifiques (FIRST, unnest, etc.) qui compliquent une éventuelle migration |
| Section "Plan de transition" | Ajouter une estimation de temps révisée pour la Phase C |

---

## 6. Résumé

| Aspect | Plan | Analyse | Réalité vérifiée |
|---|---|---|---|
| Diagnostic du problème de locking | Correct | Correct et plus détaillé | Confirmé |
| Inventaire DuckDB features | Incomplet | Non inventorié | 6+ features DuckDB-spécifiques non mentionnées |
| Estimation d'effort migration | ~5-6h (optimiste) | Non chiffré | ~12-16h réaliste |
| Recommandation SQLite WAL | Justifiée | Justifiée, avec quick-wins préalables | Justifiée, mais Phase A d'abord |
| Issue #36 interaction | Recommande d'attendre | Non couvert | Déjà implémenté côté Parquet |
| Cohérence des numéros de lignes | Obsolètes | Plus récents, quelques décalages | Code a évolué depuis |
