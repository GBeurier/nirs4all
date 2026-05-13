import sys
import tempfile
from pathlib import Path

import nirs4all
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.synthesis.builder import SyntheticDatasetBuilder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler


def make_dataset():
    return (
        SyntheticDatasetBuilder(n_samples=60, random_state=42)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .with_partitions(train_ratio=0.8)
        .build()
    )


def main(n_jobs: int) -> None:
    dataset = make_dataset()
    pipeline = [
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
        {"_or_": [None, StandardScaler()]},
        {"model": PLSRegression(n_components=4)},
    ]
    authoring_template = PipelineConfigs(pipeline).original_template

    tmp_root = Path('D:/nirs4all/.tmp')
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'orig-template-{n_jobs}-', dir=tmp_root) as tmpdir:
        workspace = Path(tmpdir) / f'workspace_{n_jobs}'
        nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
            n_jobs=n_jobs,
            workspace_path=str(workspace),
            random_state=42,
        )

        store = WorkspaceStore(workspace_path=workspace)
        try:
            rows = list(store.list_pipelines().iter_rows(named=True))
            print(f'n_jobs={n_jobs} rows={len(rows)} workspace={workspace}')
            for idx, row in enumerate(rows, start=1):
                stored = store.get_pipeline(row['pipeline_id'])
                print(
                    f"row{idx}: pipeline_id={row['pipeline_id']} name={row['name']} status={row['status']} "
                    f"original_eq={stored['original_template'] == authoring_template} "
                    f"expanded_eq={stored['expanded_config'] == authoring_template}"
                )
            if len(rows) >= 3:
                row = rows[2]
                print(f"third_row: pipeline_id={row['pipeline_id']} name={row['name']} status={row['status']}")
            else:
                print(f'third_row: none (only {len(rows)} rows)')
        finally:
            store.close()


if __name__ == '__main__':
    main(int(sys.argv[1]))
