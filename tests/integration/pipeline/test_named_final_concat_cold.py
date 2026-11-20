"""Named FINAL exports preserve fitted concat features across inference batches."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.bundle import BundleGenerator
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from tests.integration.parity.cases_branches_merges import _factory_dup_named_with_metamodel


def test_named_final_concat_replays_cold_across_batch_sizes(tmp_path):
    data_path = Path(__file__).resolve().parents[3] / "examples" / "sample_data" / "regression"
    dataset = DatasetConfigs(str(data_path))
    raw = dataset.get_dataset_at(0).x({"processing": [["raw"]]}, layout="2d")
    workspace = tmp_path / "workspace"
    with nirs4all.run(_factory_dup_named_with_metamodel(), dataset, engine="legacy", workspace_path=workspace,
                      save_artifacts=True, save_charts=False, verbose=0) as result:
        rows = result.predictions.filter_predictions(load_arrays=True)
        final = [row for row in rows if str(row["fold_id"]) == "final" and row["partition"] == "test"]
        assert len(final) == 5
    # Close/cleanup has completed before reopening the actual persisted states.
    with WorkspaceStore(workspace) as store:
        for index, row in enumerate(final):
            selected_raw = raw[np.asarray(row["sample_indices"], dtype=int)]
            bundle = BundleGenerator(workspace, store=store).export(row, tmp_path / f"{index}.n4a")
            assert bundle.is_file()
            np.savez(tmp_path / f"{index}.npz", raw_X=selected_raw, expected=np.asarray(row["y_pred"]))
    probe = r'''
import pathlib,sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
blocked=pathlib.Path(sys.argv[1]).resolve()
def audit(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
        path=pathlib.Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if path==blocked or blocked in path.parents:raise AssertionError('cold replay accessed producer workspace')
def forbidden(*args,**kwargs):raise AssertionError('cold replay attempted fit or WorkspaceStore access')
sys.addaudithook(audit)
WorkspaceStore.__init__=forbidden
for cls in (PLSRegression,PCA,TruncatedSVD,RandomForestRegressor,Ridge):cls.fit=forbidden
root=pathlib.Path(sys.argv[2])
for index in range(5):
    arrays=np.load(root/f'{index}.npz')
    x=arrays['raw_X'];expected=arrays['expected'].ravel()
    with BundleLoader(root/f'{index}.n4a') as loader:
        np.testing.assert_allclose(loader.predict(x).ravel(),expected,rtol=1e-5,atol=1e-8)
        split=len(x)//2
        for start,stop in ((0,split),(split,len(x)),(0,1),(len(x)-1,len(x))):
            np.testing.assert_allclose(loader.predict(x[start:stop]).ravel(),expected[start:stop],rtol=1e-5,atol=1e-8)
'''
    completed = subprocess.run([sys.executable, "-c", probe, str(workspace), str(tmp_path)],
                               cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
