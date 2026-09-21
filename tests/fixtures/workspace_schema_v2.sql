-- Frozen SQLite schema from nirs4all-tools 0.0.7 workspace_v2.py.
-- Provenance fixture for upgrading a real prior metadata layout, not just its version number.

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config TEXT,
    datasets TEXT,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    summary TEXT,
    error TEXT,
    project_id TEXT
);

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    name TEXT NOT NULL,
    expanded_config TEXT,
    original_template TEXT,
    generator_choices TEXT,
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    best_val REAL,
    best_test REAL,
    metric TEXT,
    duration_ms INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS chains (
    chain_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    steps TEXT NOT NULL,
    model_step_idx INTEGER NOT NULL,
    model_class TEXT NOT NULL,
    preprocessings TEXT DEFAULT '',
    fold_strategy TEXT DEFAULT 'per_fold',
    fold_artifacts TEXT,
    shared_artifacts TEXT,
    branch_path TEXT,
    source_index INTEGER,
    model_name TEXT,
    metric TEXT,
    task_type TEXT,
    best_params TEXT,
    dataset_name TEXT,
    cv_val_score REAL,
    cv_test_score REAL,
    cv_train_score REAL,
    cv_fold_count INTEGER DEFAULT 0,
    cv_scores TEXT,
    final_test_score REAL,
    final_train_score REAL,
    final_scores TEXT,
    final_agg_test_score REAL,
    final_agg_train_score REAL,
    final_agg_scores TEXT,
    relation_replay_manifest TEXT,
    relation_replay_version INTEGER,
    relation_replay_fingerprint TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    chain_id TEXT REFERENCES chains(chain_id),
    dataset_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_class TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    partition TEXT NOT NULL,
    val_score REAL,
    test_score REAL,
    train_score REAL,
    metric TEXT NOT NULL,
    task_type TEXT NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    scores TEXT,
    best_params TEXT,
    preprocessings TEXT DEFAULT '',
    branch_id INTEGER,
    branch_name TEXT,
    exclusion_count INTEGER DEFAULT 0,
    exclusion_rate REAL DEFAULT 0.0,
    refit_context TEXT DEFAULT NULL,
    prediction_scope TEXT,
    prediction_level TEXT,
    evaluation_scope TEXT,
    reduction_role TEXT,
    reduction_id TEXT,
    physical_sample_id TEXT,
    origin_sample_id TEXT,
    derived_unit_id TEXT,
    unit_level TEXT,
    unit_id TEXT,
    row_id TEXT,
    sample_influence_weight REAL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    operator_class TEXT,
    artifact_type TEXT,
    format TEXT DEFAULT 'joblib',
    size_bytes INTEGER,
    ref_count INTEGER DEFAULT 1,
    chain_path_hash TEXT,
    input_data_hash TEXT,
    dataset_hash TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    step_idx INTEGER NOT NULL,
    operator_class TEXT,
    event TEXT NOT NULL,
    duration_ms INTEGER,
    message TEXT,
    details TEXT,
    level TEXT DEFAULT 'info',
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    color TEXT DEFAULT '#14b8a6',
    created_at TIMESTAMP DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_pipelines_run_id ON pipelines(run_id);
CREATE INDEX IF NOT EXISTS idx_pipelines_dataset ON pipelines(dataset_name);
CREATE INDEX IF NOT EXISTS idx_chains_pipeline_id ON chains(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_predictions_pipeline_id ON predictions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_predictions_chain_id ON predictions(chain_id);
CREATE INDEX IF NOT EXISTS idx_predictions_dataset ON predictions(dataset_name);
CREATE INDEX IF NOT EXISTS idx_predictions_val_score ON predictions(val_score);
CREATE INDEX IF NOT EXISTS idx_predictions_partition ON predictions(partition);
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_natural_key_v2 ON predictions(
    pipeline_id,
    chain_id,
    fold_id,
    partition,
    model_name,
    branch_id
);
CREATE INDEX IF NOT EXISTS idx_logs_pipeline_id ON logs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_content_hash ON artifacts(content_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_cache_key ON artifacts(chain_path_hash, input_data_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_dataset_hash ON artifacts(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);

CREATE VIEW IF NOT EXISTS v_chain_summary AS
SELECT
    c.chain_id,
    c.pipeline_id,
    c.model_class,
    c.model_step_idx,
    c.model_name,
    c.preprocessings,
    c.branch_path,
    c.source_index,
    c.metric,
    c.task_type,
    c.best_params,
    c.dataset_name,
    c.cv_val_score,
    c.cv_test_score,
    c.cv_train_score,
    c.cv_fold_count,
    c.cv_scores,
    c.final_test_score,
    c.final_train_score,
    c.final_scores,
    c.final_agg_test_score,
    c.final_agg_train_score,
    c.final_agg_scores,
    c.relation_replay_manifest,
    c.relation_replay_version,
    c.relation_replay_fingerprint,
    pl.run_id,
    pl.status AS pipeline_status
FROM chains c
JOIN pipelines pl ON c.pipeline_id = pl.pipeline_id
WHERE EXISTS (
    SELECT 1
    FROM predictions p
    WHERE p.chain_id = c.chain_id
);

PRAGMA user_version = 2;
