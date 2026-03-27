# Roadmap: migration DuckDB → SQLite WAL

Date: 2026-03-26
Status: Proposition revue et corrigée
Réf: `duckdb_to_sqlite_migration_plan.md`, `duckdb_backup_prediction_review_2026-03-25.md`, `duckdb_migration_review_2026-03-26.md`

---

## Objectif

Remplacer DuckDB par SQLite WAL comme moteur de métadonnées du workspace nirs4all, afin de résoudre les problèmes de locking multi-processus (#37) et de supprimer une dépendance binaire externe (~50 MB).

## Périmètre de non-changement

Ce qui ne change **pas**:
- `ArrayStore` (Parquet) — totalement indépendant du moteur DB
- `artifacts/` — fichiers joblib content-addressed
- `exports/`
- L'API publique de haut niveau (`nirs4all.run()`, `nirs4all.predict()`, etc.)
- Les signatures publiques de `WorkspaceStore` (mêmes noms de méthodes, mêmes types de retour)
- Le type de retour `pl.DataFrame` pour toutes les méthodes de requête

## Conventions du document

- Chaque tâche est identifiée `[T-XX]`
- Les tâches marquées `🗑️ remove-at-v1` sont du code transitoire à supprimer quand DuckDB sera entièrement supprimé
- Estimation en heures de travail effectif (pas en temps calendaire)

## Constats après revue du code

- Le scope ne se limite pas à `store_queries.py` / `workspace_store.py` / `store_schema.py`: `pipeline/storage/migration.py`, `data/predictions.py`, `cli/commands/workspace.py`, les dépendances packaging et de nombreuses docstrings/docs sont aussi impactées.
- Le code SQL n'est pas centralisé dans `store_queries.py`: il y a aussi du SQL inline dans `workspace_store.py` et dans les migrations de `store_schema.py`.
- `Predictions.from_file()` reconnaît aujourd'hui `*.duckdb`, mais pas `*.sqlite`; `store_stats()` pointe aussi en dur sur `store.duckdb`.
- Les tests existants ne sont pas "quasi inchangés": de nombreux tests unitaires et d'intégration asservent explicitement la présence de `store.duckdb`.
- Le packaging ne se résume pas à `pyproject.toml`: `requirements.txt`, `requirements-test.txt`, `conda-forge/meta.yaml` et `scripts/sync_conda_recipe.py` mentionnent aussi DuckDB.
- La migration vers SQLite ne règle pas à elle seule l'atomicité entre métadonnées SQL et arrays Parquet: le flush DB puis `ArrayStore.save_batch()` reste un point de cohérence à traiter explicitement.

---

## Phase 0 — Corrections de durée de vie des connexions (pré-migration)

**Objectif**: Réduire immédiatement la douleur utilisateur sans changer de moteur DB.
**Estimation**: 4-6h
**Dépendances**: Aucune

Ces corrections sont utiles quelle que soit la décision finale sur le moteur, et bénéficieront aussi à SQLite.

### [T-01] Détacher le store du RunResult (1h)

Fichiers: `api/run.py`, `api/result.py`, `pipeline/runner.py`

Actuellement `RunResult` conserve une référence `runner → orchestrator → WorkspaceStore`, gardant la connexion DB vivante tant que le résultat existe. Le `RunResult` doit devenir détaché:
- Après `run()`, fermer le runner non-session (`runner.close()` ou au minimum `store.close()`) avant de retourner le résultat
- `RunResult` ne conserve que ce dont il a réellement besoin: `workspace_path`, les données en mémoire, et les métadonnées minimales pour l'export
- Adapter explicitement les dépendances cachées actuelles: `RunResult.export()`, `RunResult.export_model()`, `artifacts_path` et les accès qui supposent un runner vivant
- Les sessions gardent leur comportement actuel: un runner/session partagé peut rester ouvert jusqu'au `session.close()`

### [T-02] Détacher Predictions.from_workspace() et élargir la détection de fichier (1-1.5h)

Fichier: `data/predictions.py`

`Predictions.from_workspace()` garde le `WorkspaceStore` ouvert après chargement. Refactorer pour:
- Ouvrir le store
- Charger les metadata + arrays en batch (par dataset, pas un par un)
- Fermer le store immédiatement
- Retourner un objet en mémoire pure
- Accepter un workspace, `store.duckdb` **ou** `store.sqlite` dans `Predictions.from_file()`
- Mettre à jour `store_stats()` pour ne plus hardcoder `store.duckdb`

### [T-03] Batch loading des arrays dans predictions (1h)

Fichier: `data/predictions.py`

`_populate_buffer_from_store()` fait un `load_single()` par prédiction (rescan Parquet à chaque fois). Grouper les `prediction_id` par `dataset_name` et faire un seul `load_batch()` par dataset.

### [T-04] Tests de non-régression Phase 0 (0.5h)

- Test que `RunResult` ne maintient pas de connexion ouverte
- Test que `Predictions.from_workspace()` ne maintient pas de connexion ouverte
- Test que `Predictions.from_file("store.sqlite")` fonctionne
- Test du batch loading (performance sur 100+ prédictions)

### Point manquant à traiter pendant ou juste après Phase 0

La séquence actuelle "écriture metadata SQL puis écriture arrays Parquet" n'est pas atomique. SQLite ne corrige pas ce point. Il faut prévoir un durcissement explicite:
- soit un ordre d'écriture + mécanisme de reprise/réconciliation au redémarrage
- soit des tests d'intégrité/réparation couvrant les crashs entre SQL et Parquet

---

## Phase 1 — Implémentation SQLite dans WorkspaceStore

**Objectif**: Remplacer l'implémentation interne de `WorkspaceStore` de DuckDB vers SQLite WAL.
**Estimation**: 14-17h
**Dépendances**: Phase 0 terminée (ou en parallèle)

### [T-05] Réécriture du DDL dans store_schema.py (1.5h)

Fichier: `pipeline/storage/store_schema.py`

Changements de types:
| DuckDB | SQLite |
|---|---|
| `VARCHAR` | `TEXT` |
| `DOUBLE` | `REAL` |
| `BIGINT` | `INTEGER` |
| `JSON` | `TEXT` (JSON stocké comme texte, fonctions json_* de SQLite 3.38+ pour requêtes) |
| `TIMESTAMP DEFAULT current_timestamp` | Déclaration conservée, mais conversion Python à expliciter (`datetime` vs `str`) |

Changements structurels:
- Supprimer `SET memory_limit`, `SET threads`, `SET checkpoint_threshold`
- Supprimer `PRAGMA enable_progress_bar=false`
- Ajouter `PRAGMA journal_mode=WAL`
- Ajouter `PRAGMA foreign_keys=ON`
- Ajouter `PRAGMA busy_timeout=5000` (remplace le retry-on-lock pour le cas multi-processus)

Réécriture des migrations:
- `_migrate_schema()` utilise `FIRST()`, `duckdb_indexes()`, `ROW_NUMBER() OVER` — réécrire chaque usage
- `FIRST(col)` → sous-query `(SELECT col FROM t WHERE ... LIMIT 1)` ou agrégat custom
- `duckdb_indexes()` → `PRAGMA index_list('table_name')`
- `ROW_NUMBER() OVER (PARTITION BY ...)` → supporté nativement en SQLite 3.25+, pas de changement
- `_auto_migrate_prediction_arrays()` → cette migration legacy reste utile pour les vieux workspaces déjà ouverts côté SQLite; coordonner son avenir avec `pipeline/storage/migration.py`
- Vérifier explicitement si les vues de compatibilité supprimées (`v_aggregated_predictions`, `v_aggregated_predictions_all`) doivent être recréées, remplacées ou supprimées définitivement

### [T-06] Réécriture des requêtes dans store_queries.py (3-4h)

Fichiers: `pipeline/storage/store_queries.py`, `pipeline/storage/workspace_store.py`, `pipeline/storage/store_schema.py`

Changement du style de paramètres:
- **Toutes** les requêtes utilisent `$1, $2, $3...` → remplacer par `?, ?, ?...`
- Les query builders dynamiques (`build_prediction_query`, `build_top_predictions_query`, `build_chain_summary_query`, `build_top_chains_query`) construisent des `${idx}` avec index incrémenté → refactorer pour utiliser `?` avec des listes de paramètres positionnels
- ~50 requêtes statiques + 5 query builders dynamiques, plus du SQL inline dans `workspace_store.py` et dans les migrations `store_schema.py`

Réécriture de `GET_CHAIN_ARTIFACT_IDS` (la requête la plus complexe):
```sql
-- DuckDB actuel:
SELECT DISTINCT unnest(
    list_concat(
        COALESCE(json_keys(fold_artifacts)::VARCHAR[], ARRAY[]::VARCHAR[]),
        COALESCE(json_keys(shared_artifacts)::VARCHAR[], ARRAY[]::VARCHAR[])
    )
) AS key_name, fold_artifacts, shared_artifacts
FROM chains WHERE pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = $1)

-- SQLite équivalent:
SELECT DISTINCT je.key AS key_name, c.fold_artifacts, c.shared_artifacts
FROM chains c, json_each(c.fold_artifacts) je
WHERE c.pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = ?)
UNION
SELECT DISTINCT je.key AS key_name, c.fold_artifacts, c.shared_artifacts
FROM chains c, json_each(c.shared_artifacts) je
WHERE c.pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = ?)
```

Réécriture des vues (`VIEW_DDL`):
- `v_chain_summary` utilise des colonnes stockées dans la table `chains` (pré-calculées). Pas de FIRST() dans la vue elle-même — vérifier.

### [T-07] Réécriture de workspace_store.py (5-6h)

Fichier: `pipeline/storage/workspace_store.py`

Changements principaux:

**1. Connexion** (remplace les lignes 225-254):
```python
import sqlite3

db_path = self._workspace_path / "store.sqlite"
self._conn = sqlite3.connect(
    str(db_path),
    check_same_thread=False,
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
)
self._conn.row_factory = sqlite3.Row  # pour dict-like access
self._conn.execute("PRAGMA journal_mode=WAL")
self._conn.execute("PRAGMA foreign_keys=ON")
self._conn.execute("PRAGMA busy_timeout=5000")
```

**2. Retry-on-lock** (lignes 108-149):
- Supprimer `_retry_on_lock` (le `PRAGMA busy_timeout` de SQLite WAL gère les contentions)
- Ou adapter le décorateur pour attraper `sqlite3.OperationalError` si on veut garder un retry applicatif

**3. `_fetch_pl()`** (lignes 333-342):
- Remplacer `.pl()` (zero-copy DuckDB→Arrow→Polars) par construction Polars depuis les résultats SQLite
- Utiliser `pl.DataFrame(cursor.fetchall(), schema=...)` ou `pl.from_dicts([dict(row) for row in cursor])`
- La performance sera légèrement inférieure pour les gros résultats, mais acceptable pour les métadonnées
- Vérifier la stabilité des types de retour (notamment `datetime`/timestamps) pour ne pas casser les adapters webapp/tests existants

**4. `_fetch_one()`** (lignes 322-331):
- Adapter le `dict(row)` de `sqlite3.Row` (compatible dict-like avec `row_factory=sqlite3.Row`)

**5. `transaction()`** context manager:
- SQLite: `conn.execute("BEGIN")` ... `conn.commit()` / `conn.rollback()`

**6. `close()`**:
- Supprimer `CHECKPOINT` (pas nécessaire en SQLite WAL — le checkpoint est automatique)
- Garder `conn.close()`

**7. SQL inline dans les méthodes** (~15 requêtes avec FIRST()):
- `update_chain_summary()` (lignes 671-696): remplacer `FIRST()` par sous-query
- `bulk_update_chain_summaries()` (lignes 806-835): idem
- `list_runs()` (ligne 1427): `datasets::VARCHAR LIKE` → `CAST(datasets AS TEXT) LIKE`

**8. `_safe_execute()`**:
- Remplacer `duckdb.TransactionException` par `sqlite3.OperationalError`

**9. Docstrings / messages techniques dans le fichier**:
- Remplacer les références "DuckDB" / `store.duckdb` / zero-copy DuckDB dans les docstrings qui deviennent fausses après migration

### [T-08] Fichier store.sqlite — nommage et détection (0.5h)

Fichiers: `workspace_store.py`, `pipeline/storage/__init__.py`, `data/predictions.py`

- Le fichier DB s'appelle désormais `store.sqlite` (au lieu de `store.duckdb`)
- `WorkspaceStore.__init__()` détecte automatiquement:
  1. Si `store.sqlite` existe → l'utiliser
  2. Si `store.duckdb` existe et pas `store.sqlite` → lancer la migration automatique [T-09]
  3. Sinon → créer `store.sqlite`
- `Predictions.from_file()` et les helpers associés reconnaissent aussi `store.sqlite`

### [T-09] Script de conversion DuckDB → SQLite (3h) `🗑️ remove-at-v1`

Fichier: `pipeline/storage/migration.py` (à étendre)

Script `migrate_duckdb_to_sqlite(workspace_path)`:

```python
def migrate_duckdb_to_sqlite(workspace_path: Path) -> MigrationReport:
    """Migrate metadata from store.duckdb to store.sqlite.

    This function requires duckdb to be installed. It is a one-time
    migration tool and will be removed in v1.0 when DuckDB support
    is fully deprecated.

    Steps:
    1. Acquire an inter-process migration lock
    2. Open store.duckdb in read-only mode
    3. Create store.sqlite.tmp with the new schema
    4. Copy all rows table by table (runs, pipelines, chains,
       predictions, artifacts, logs, projects)
    5. Verify row counts + PRAGMA integrity_check
    6. Atomically rename store.sqlite.tmp → store.sqlite
    7. Rename store.duckdb → store.duckdb.bak
    """
```

Ordre de copie (respecter les FK):
1. `projects`
2. `runs`
3. `pipelines`
4. `chains`
5. `predictions`
6. `artifacts`
7. `logs`

Gestion des types:
- Les colonnes JSON (VARCHAR en DuckDB) restent du texte → pas de conversion nécessaire
- Les TIMESTAMP DuckDB → conserver une stratégie cohérente côté Python (texte ISO 8601 ou convertisseur `datetime`, mais choix unique et testé)
- Les DOUBLE → REAL (transparent)

La migration est déclenchée automatiquement par `WorkspaceStore.__init__()` [T-08] quand:
- `store.duckdb` existe
- `store.sqlite` n'existe pas
- `duckdb` est importable

Si `duckdb` n'est pas installé et que seul `store.duckdb` existe → lever une erreur claire avec message d'installation: `pip install duckdb` pour effectuer la migration.

Points indispensables:
- Ne jamais écrire directement dans `store.sqlite` final tant que la copie n'est pas validée
- Gérer explicitement les restes de `store.sqlite.tmp` / lock file après crash
- Empêcher deux migrations automatiques concurrentes sur le même workspace

### [T-10] Rétrocompatibilité duckdb optionnelle dans le packaging (1h) `🗑️ remove-at-v1`

Fichiers: `pyproject.toml`, `requirements.txt`, `requirements-test.txt`, `conda-forge/meta.yaml`, `scripts/sync_conda_recipe.py`

- Retirer `duckdb` des dépendances obligatoires
- L'ajouter en extra optionnel: `[project.optional-dependencies]` → `migration = ["duckdb>=1.0.0"]`
- Les nouveaux utilisateurs n'installent plus duckdb du tout
- Les anciens utilisateurs qui ont un `store.duckdb` font `pip install nirs4all[migration]` pour migrer
- Garder DuckDB dans l'environnement de test/CI qui exécute encore les tests de migration `🗑️ remove-at-v1`

---

## Phase 2 — Tests

**Objectif**: Couverture complète de la nouvelle implémentation SQLite + tests de migration.
**Estimation**: 5-7h
**Dépendances**: Phase 1

### [T-11] Adapter les tests unitaires existants du store (2h)

Fichiers:
- `tests/unit/pipeline/storage/test_workspace_store.py`
- `tests/unit/pipeline/storage/test_workspace_store_api.py`
- `tests/unit/pipeline/storage/test_store_schema.py`
- `tests/unit/pipeline/storage/test_aggregated_predictions.py`
- `tests/unit/pipeline/storage/test_cross_run_cache.py`
- `tests/unit/pipeline/storage/test_chain_replay.py`
- `tests/unit/pipeline/storage/test_export_chain.py`
- `tests/unit/pipeline/storage/test_export_roundtrip.py`
- `tests/unit/data/test_predictions_store.py`

Les tests existants créent un `WorkspaceStore(tmp_path)` et exercent les méthodes CRUD. Ils doivent passer tels quels si les signatures publiques n'ont pas changé. Changements attendus:
- Supprimer les assertions qui vérifiaient des types DuckDB (`duckdb.DuckDBPyConnection`, etc.)
- Vérifier que le fichier créé est bien `store.sqlite` (pas `store.duckdb`)
- Vérifier que le mode WAL est actif (PRAGMA query)
- Vérifier la compatibilité de `Predictions.from_file(store.sqlite)` et `store_stats()`
- Vérifier le comportement des timestamps (`datetime` vs `str`) là où les adapters/tests l'attendent explicitement

### [T-12] Tests de migration DuckDB → SQLite (1.5h) `🗑️ remove-at-v1`

Fichier principal: `tests/unit/pipeline/storage/test_migration.py` (à étendre) `🗑️ remove-at-v1`

Note: un fichier dédié `test_duckdb_to_sqlite_migration.py` reste possible, mais le repo a déjà un fichier `test_migration.py` pour la migration legacy DuckDB → Parquet; il est plus cohérent de l'étendre ou de le scinder volontairement, pas d'introduire un faux nouveau point d'entrée par accident.

```python
@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
class TestDuckDBToSQLiteMigration:
    """Tests for one-time DuckDB → SQLite workspace migration.

    These tests will be removed in v1.0 when DuckDB support is
    fully deprecated.
    """

    def test_migration_creates_sqlite_from_duckdb(self, tmp_path):
        """A workspace with store.duckdb but no store.sqlite triggers migration."""

    def test_migration_preserves_all_rows(self, tmp_path):
        """Every row in every table is copied exactly."""

    def test_migration_renames_duckdb_to_bak(self, tmp_path):
        """After migration, store.duckdb is renamed to store.duckdb.bak."""

    def test_migration_is_idempotent(self, tmp_path):
        """Running migration twice does not corrupt data."""

    def test_auto_migration_on_workspace_open(self, tmp_path):
        """WorkspaceStore.__init__ triggers migration if store.duckdb exists."""

    def test_missing_duckdb_package_gives_clear_error(self, tmp_path):
        """If duckdb is not installed and store.duckdb exists, raise ImportError with message."""

    def test_migration_handles_empty_tables(self, tmp_path):
        """Migration works on a workspace with empty tables."""

    def test_migration_handles_json_columns(self, tmp_path):
        """JSON columns (config, scores, etc.) are preserved as text."""

    def test_migration_handles_timestamps(self, tmp_path):
        """Timestamps are correctly converted to ISO 8601 text."""
```

### [T-13] Tests de concurrence SQLite WAL (1h)

Fichier: `tests/unit/pipeline/storage/test_sqlite_concurrency.py`

```python
class TestSQLiteConcurrency:
    """Verify that SQLite WAL resolves the locking issues that motivated the migration."""

    def test_concurrent_readers(self, tmp_path):
        """Multiple WorkspaceStore instances can read simultaneously."""

    def test_writer_does_not_block_readers(self, tmp_path):
        """A write transaction does not block concurrent reads."""

    def test_successive_runs_same_workspace(self, tmp_path):
        """Two sequential nirs4all.run() calls on the same workspace succeed."""

    def test_reader_after_writer_closes(self, tmp_path):
        """A reader can open the store immediately after a writer closes."""
```

### [T-14] Tests d'intégration (0.5h)

Vérifier que les tests d'intégration existants passent:
```bash
pytest tests/integration/ -x
```

Contrairement à l'hypothèse initiale, plusieurs tests d'intégration devront être adaptés car ils asservent explicitement `store.duckdb`. À minima:
- `tests/integration/storage/test_duckdb_pipeline.py`
- `tests/integration/artifacts/test_artifact_flow.py`
- `tests/integration/pipeline/test_branch_artifacts.py`
- `tests/integration/pipeline/test_branch_predict_mode.py`
- `tests/integration/pipeline/test_merge_prediction_mode.py`
- et les tests runner/unit qui vérifient le nom du fichier (`tests/unit/pipeline/test_runner_state.py`, `tests/unit/pipeline/test_runner_comprehensive.py`, `tests/unit/pipeline/test_runner_regression_prevention.py`)

---

## Phase 3 — Mise à jour de la webapp

**Objectif**: Adapter la webapp pour fonctionner avec le nouveau store SQLite.
**Estimation**: 3-4h
**Dépendances**: Phase 1

### Analyse d'impact webapp

La webapp interagit avec le storage nirs4all via 3 fichiers:

| Fichier webapp | Usage | Impact migration |
|---|---|---|
| `api/store_adapter.py` | Wraps `WorkspaceStore` — appelle les méthodes publiques | **Aucun changement de code** si les signatures sont identiques |
| `api/aggregated_predictions.py` | Crée `WorkspaceStore` directement, vérifie `store.duckdb` | **Changement**: détection de fichier + docstrings |
| `api/workspace_manager.py` | Vérifie `store.duckdb` pour détecter un workspace valide | **Changement**: détection de fichier + docstrings |

Le webapp n'a **aucune dépendance directe à duckdb** — tout passe par `WorkspaceStore`. Donc si les signatures publiques ne changent pas, l'impact est minime.

### [T-15] Mise à jour de la détection de fichier dans la webapp (1h)

Fichiers:
- `nirs4all_webapp/api/workspace_manager.py` (lignes 133-142)
- `nirs4all_webapp/api/aggregated_predictions.py` (lignes 170-175)

Changements:

```python
# workspace_manager.py — AVANT:
store_db = self.workspace_dir / "store.duckdb"
if not store_db.exists():
    store_db = self.workspace_path / "store.duckdb"

# APRÈS:
def _find_store_file(base: Path) -> Path | None:
    """Locate the workspace store file (SQLite preferred, DuckDB legacy)."""
    for name in ("store.sqlite", "store.duckdb"):
        for parent in (base / "workspace", base):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None
```

```python
# aggregated_predictions.py — AVANT:
db_path = workspace_path / "store.duckdb"
if not db_path.exists():
    raise HTTPException(status_code=404, detail="No DuckDB store found...")

# APRÈS:
store_file = _find_store_file(workspace_path)
if store_file is None:
    raise HTTPException(status_code=404, detail="No workspace store found...")
```

La détection doit chercher `store.sqlite` **en premier**, puis `store.duckdb` en fallback (pour les workspaces pas encore migrés). Quand un `store.duckdb` est trouvé, `WorkspaceStore.__init__` déclenchera la migration automatique [T-08/T-09].

### [T-16] Mise à jour des docstrings et messages d'erreur webapp (0.5h)

Fichiers:
- `nirs4all_webapp/api/aggregated_predictions.py` (docstring module, lignes 1-11)
- `nirs4all_webapp/api/workspace_manager.py` (docstrings, lignes 91-97, 144-146, 160-162)
- `nirs4all_webapp/api/store_adapter.py` (docstring, lignes 1-6)

Remplacer toutes les références "DuckDB" par "workspace store" ou "SQLite" dans les docstrings et messages d'erreur. Exemples:
- `"Aggregated predictions API endpoints backed by DuckDB"` → `"Aggregated predictions API endpoints backed by workspace store"`
- `"Valid nirs4all workspace (DuckDB store)"` → `"Valid nirs4all workspace (store found)"`
- `"nirs4all library is required for DuckDB store access"` → `"nirs4all library is required for workspace store access"`

### [T-17] Mise à jour des tests webapp (1h)

Fichiers:
- `nirs4all_webapp/tests/test_aggregated_predictions_api.py` (lignes 170-173, 750-755)
- `nirs4all_webapp/tests/test_store_integration.py` (lignes 249-343)

Changements:
- `(workspace_dir / "store.duckdb").touch()` → `(workspace_dir / "store.sqlite").touch()`
- Adapter les assertions qui vérifient la présence de `store.duckdb`
- Ajouter un test de fallback: quand seul `store.duckdb` existe, la détection fonctionne quand même

### [T-18] Mise à jour de la documentation interne webapp (1h)

Fichier: `nirs4all_webapp/docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md`

Ce document référence `store.duckdb` **~40 fois**. Il faut:
- Remplacer `store.duckdb` par `store.sqlite` partout
- Mettre à jour les mentions de "DuckDB's ACID properties" → "SQLite WAL"
- Mettre à jour le schéma d'architecture workspace
- Ajouter une note sur la migration automatique pour les anciens workspaces

---

## Phase 4 — Documentation des changements publics

**Objectif**: Documenter clairement ce qui a changé pour les utilisateurs et les développeurs de la webapp.
**Estimation**: 3-4h
**Dépendances**: Phases 1-3

### [T-19] Document de migration pour les utilisateurs (1h)

Fichier: `docs/source/migration/duckdb_to_sqlite.md`

Contenu:

```markdown
# Migration DuckDB → SQLite (v0.9)

## Ce qui change

### Fichier workspace
- Le fichier de métadonnées passe de `store.duckdb` à `store.sqlite`
- La migration est **automatique**: à la première ouverture d'un ancien workspace,
  nirs4all convertit `store.duckdb` → `store.sqlite` et renomme l'ancien en `.bak`

### Dépendance duckdb
- `duckdb` n'est plus une dépendance obligatoire
- Pour migrer un ancien workspace: `pip install nirs4all[migration]`
- Les nouveaux workspaces n'ont pas besoin de duckdb

### Comportement identique
- Toutes les API publiques sont identiques: run(), predict(), explain(), etc.
- Les signatures de WorkspaceStore sont identiques
- Les types de retour (pl.DataFrame) sont identiques
- ArrayStore (Parquet) et artifacts (joblib) ne changent pas

## Ce qui ne change PAS

| Composant | Changement |
|-----------|-----------|
| `nirs4all.run()` | Aucun |
| `nirs4all.predict()` | Aucun |
| `nirs4all.explain()` | Aucun |
| `Predictions` API | Aucun |
| `WorkspaceStore` signatures publiques | Aucun |
| `ArrayStore` (Parquet) | Aucun |
| `artifacts/` (joblib) | Aucun |
| Pipeline syntax | Aucun |

## Actions requises

### Pour les utilisateurs de la lib Python
Pas d'action pour les nouveaux workspaces.
Pour un ancien workspace contenant encore `store.duckdb`, la migration est transparente **si** DuckDB est disponible; sinon il faut installer l'extra `migration` une fois.

### Pour les développeurs qui importent WorkspaceStore directement
Le `WorkspaceStore` conserve exactement les mêmes méthodes et signatures.
Le seul changement visible est le nom du fichier (store.sqlite au lieu de store.duckdb).

### Pour les développeurs de la webapp
Voir le document `storage_migration_webapp.md` pour les détails.
```

### [T-20] Document de migration pour les développeurs webapp (1h)

Fichier: `docs/source/migration/storage_migration_webapp.md`

Contenu:

```markdown
# Storage Migration Guide — Webapp Developers

## Résumé des changements

| Avant (v0.8) | Après (v0.9) |
|---|---|
| `store.duckdb` | `store.sqlite` |
| Dépendance `duckdb` obligatoire | `sqlite3` (stdlib) |
| `duckdb.connect()` | `sqlite3.connect()` |
| Locking exclusif fichier | WAL: readers concurrents + 1 writer |
| Paramètres `$1, $2, $3` | Paramètres `?, ?, ?` |

## Impact sur la webapp

### Pas de changement sur StoreAdapter
`StoreAdapter` n'utilise que les méthodes publiques de `WorkspaceStore`.
Toutes les méthodes sont identiques (mêmes noms, mêmes arguments, mêmes retours).

### Détection de fichier
La webapp doit chercher `store.sqlite` en premier, puis `store.duckdb` en fallback.
Quand un ancien `store.duckdb` est trouvé, `WorkspaceStore.__init__` déclenche
automatiquement la migration vers `store.sqlite`.

### Polars DataFrame
Les méthodes de requête de `WorkspaceStore` retournent toujours des `pl.DataFrame`.
L'implémentation interne change (plus de `.pl()` zero-copy, remplacé par
construction depuis les résultats SQLite), mais le type de retour est identique.

### Messages d'erreur et docstrings
Les références à "DuckDB" dans les messages d'erreur et docstrings webapp
doivent être mises à jour.

## Fichiers webapp à modifier

| Fichier | Nature du changement |
|---|---|
| `api/workspace_manager.py` | Détection: `store.sqlite` + fallback `store.duckdb` |
| `api/aggregated_predictions.py` | Idem + mise à jour docstrings |
| `api/store_adapter.py` | Mise à jour docstrings uniquement |
| `tests/test_aggregated_predictions_api.py` | `store.duckdb` → `store.sqlite` dans les fixtures |
| `tests/test_store_integration.py` | Idem |
| `docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md` | ~40 références `store.duckdb` → `store.sqlite` |

## Ce que la webapp n'a PAS besoin de faire

- Installer duckdb (n'a jamais été une dépendance directe de la webapp)
- Modifier les endpoints API (mêmes routes, mêmes réponses)
- Modifier le frontend (aucun changement visible côté UI)
- Modifier les Pydantic models (mêmes schémas)
```

### Sweep documentaire indispensable

En plus des deux nouveaux guides de migration, il faut corriger la documentation existante qui référence encore DuckDB ou `store.duckdb`. À minima:
- `docs/source/reference/workspace.md`
- `docs/source/reference/storage.md`
- `docs/source/api/workspace.md`
- `docs/source/api/storage.md`
- `docs/source/onboarding/workspace_intro.md`
- `docs/source/developer/artifacts.md`
- `docs/source/developer/artifacts_internals.md`
- `docs/source/user_guide/predictions/index.md`
- `docs/source/user_guide/predictions/understanding_predictions.md`
- `examples/user/06_deployment/U03_workspace_management.py`
- les docstrings/messages CLI et API qui citent encore DuckDB (`cli/commands/workspace.py`, `api/predict.py`, `api/result.py`, `api/session.py`, `pipeline/runner.py`)

---

## Phase 5 — Nettoyage et dépréciation `🗑️ remove-at-v1`

**Objectif**: Supprimer tout le code DuckDB quand la migration est considérée stable.
**Estimation**: 2-3h
**Dépendances**: Après validation en production, probablement v1.0

### [T-21] Supprimer le code de migration DuckDB

Fichiers à supprimer / nettoyer:
- `pipeline/storage/migration.py` — supprimer `migrate_duckdb_to_sqlite()` et `migrate_arrays_to_parquet()` (les arrays sont déjà en Parquet depuis longtemps)
- `tests/unit/pipeline/storage/test_migration.py` — supprimer les cas DuckDB legacy devenus hors support
- `pyproject.toml`, `requirements*.txt`, `conda-forge/meta.yaml`, `scripts/sync_conda_recipe.py` — supprimer la rétrocompatibilité DuckDB
- `WorkspaceStore.__init__()` — supprimer la logique de détection/migration automatique de `store.duckdb`

### [T-22] Supprimer les fallbacks store.duckdb dans la webapp

Fichiers:
- `nirs4all_webapp/api/workspace_manager.py` — ne chercher que `store.sqlite`
- `nirs4all_webapp/api/aggregated_predictions.py` — idem
- `nirs4all_webapp/tests/` — supprimer les tests de fallback DuckDB

### [T-23] Supprimer les documents internes de migration

Fichiers:
- `docs/_internal/duckdb_to_sqlite_migration_plan.md`
- `docs/_internal/duckdb_backup_prediction_review_2026-03-25.md`
- `docs/_internal/duckdb_migration_review_2026-03-26.md`
- `docs/_internal/duckdb_to_sqlite_roadmap.md` (ce document)
- `docs/source/migration/duckdb_to_sqlite.md`
- `docs/source/migration/storage_migration_webapp.md`

---

## Planning récapitulatif

| Phase | Tâches | Estimation | Bloquant pour |
|---|---|---|---|
| **Phase 0** — Connexion lifecycle | T-01 → T-04 | 4-6h | Rien (peut être fait seul) |
| **Phase 1** — SQLite impl | T-05 → T-10 | 14-17h | Phase 2, Phase 3 |
| **Phase 2** — Tests | T-11 → T-14 | 5-7h | Phase 4 |
| **Phase 3** — Webapp | T-15 → T-18 | 3-4h | Phase 4 |
| **Phase 4** — Documentation | T-19 → T-20 | 3-4h | — |
| **Phase 5** — Nettoyage v1 | T-21 → T-23 | 2-3h | Release v1.0 |
| **Total** | 23 tâches + sweep doc | **~31-41h** | |

## Ordre d'exécution recommandé

```
Phase 0 ──────────────────────────────┐
                                      ├──► Phase 4 (docs)
Phase 1 ──► Phase 2 (tests) ─────────┤
       └──► Phase 3 (webapp) ────────┘

                        ... production validation ...

                                      Phase 5 (cleanup v1)
```

Phases 0 et 1 peuvent démarrer en parallèle (pas de dépendance directe). Phase 0 est recommandée en premier car elle apporte un bénéfice immédiat avec peu de risque.

---

## Inventaire complet des fichiers affectés

### nirs4all (lib)

| Fichier | Phase | Nature |
|---|---|---|
| `pipeline/storage/workspace_store.py` | 1 | Réécriture interne (connexion, SQL, .pl()) |
| `pipeline/storage/store_schema.py` | 1 | DDL types + migrations |
| `pipeline/storage/store_queries.py` | 1 | Paramètres $N → ? + fonctions DuckDB |
| `pipeline/storage/migration.py` | 1 | Ajout `migrate_duckdb_to_sqlite()` 🗑️ |
| `pipeline/storage/__init__.py` | 1 | Éventuellement exporter la migration |
| `data/predictions.py` | 0 | Détacher store + batch loading |
| `api/run.py` | 0 | Détacher store du RunResult |
| `api/result.py` | 0 | Ne plus garder le runner vivant |
| `api/predict.py` | 4 | Docstrings/messages "DuckDB store" |
| `api/session.py` | 4 | Docstrings/messages de cleanup |
| `pipeline/runner.py` | 0 | Adapter le cycle de vie du store |
| `cli/commands/workspace.py` | 4 | Messages CLI + nom de fichier affiché |
| `pyproject.toml` | 1 | duckdb → optional extra |
| `requirements.txt` | 1 | Retirer DuckDB des dépendances runtime |
| `requirements-test.txt` | 1 | Conserver/retirer DuckDB selon stratégie CI migration |
| `conda-forge/meta.yaml` | 1 | recipe conda-forge |
| `scripts/sync_conda_recipe.py` | 1 | mapping duckdb ↔ python-duckdb |
| `tests/unit/pipeline/storage/test_*.py` | 2 | Adaptations mineures |
| `tests/unit/data/test_predictions_store.py` | 2 | `.sqlite` + `store_stats()` |
| `tests/unit/pipeline/storage/test_migration.py` | 2 | Étendre puis supprimer à terme 🗑️ |
| `tests/unit/pipeline/storage/test_sqlite_concurrency.py` | 2 | Nouveau |
| `docs/source/migration/duckdb_to_sqlite.md` | 4 | Nouveau |
| `docs/source/migration/storage_migration_webapp.md` | 4 | Nouveau |

### nirs4all_webapp

| Fichier | Phase | Nature |
|---|---|---|
| `api/workspace_manager.py` | 3 | Détection store.sqlite + fallback |
| `api/aggregated_predictions.py` | 3 | Détection + docstrings |
| `api/store_adapter.py` | 3 | Docstrings uniquement |
| `tests/test_aggregated_predictions_api.py` | 3 | Fixtures store.sqlite |
| `tests/test_store_integration.py` | 3 | Fixtures + assertions |
| `docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md` | 3 | ~40 refs store.duckdb |

---

## Risques identifiés et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Régression SQL (sémantique NULL, tri, types) | Moyenne | Élevé | Tests unitaires exhaustifs [T-11], tests d'intégration [T-14] |
| Performance dégradée sur gros workspaces (>10K chains) | Faible | Moyen | Benchmark avant/après. Les workspaces courants ont <1K chains |
| Perte de `.pl()` zero-copy | Certaine | Faible | Construction manuelle `pl.DataFrame`. Impact négligeable pour les volumes de métadonnées |
| Migration automatique interrompue laisse un `store.sqlite` partiel | Moyenne | Élevé | Écrire dans `store.sqlite.tmp`, lock inter-processus, `integrity_check`, rename atomique [T-09] |
| Migration de données corrompue | Faible | Élevé | Vérification des row counts, backup `.bak`, tests exhaustifs [T-12] |
| Dérive de type sur les timestamps (`datetime` → `str`) | Moyenne | Moyen | Décision explicite sur la représentation + tests webapp/store [T-07, T-11] |
| SQLite ne résout pas l'incohérence SQL/Parquet | Moyenne | Élevé | Traiter explicitement l'atomicité / reprise après crash, ne pas considérer le changement de moteur comme suffisant |
| Ancien workspace sans duckdb installé | Moyenne | Moyen | Message d'erreur clair + documentation. Extra `pip install nirs4all[migration]` |
| SQLite version trop ancienne pour json_* | Très faible | Élevé | Vérifier explicitement `sqlite3.sqlite_version >= 3.38` en CI et documenter le prérequis runtime |
