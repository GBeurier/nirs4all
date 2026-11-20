"""Source-local stacking refits retain their complete raw-input fitted closure."""

import json
import subprocess
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def test_source_stacking_final_keeps_source_axes_prefixes_and_cold_predictions(tmp_path):
    rng = np.random.default_rng(92)
    sources = [rng.normal(size=(30, 5)).astype(np.float32), (rng.normal(size=(30, 4)) * 3 + 11).astype(np.float32)]
    for values in sources:
        values[24:] += 100  # Held-out rows must not influence any fitted state.
    y = (sources[0][:, 0] + sources[1][:, 2] * .7).astype(np.float32)
    headers = [[str(900 + index) for index in range(5)], [str(1400 + index) for index in range(4)]]
    dataset = SpectroDataset("source_stack")
    dataset.add_samples([values[:24] for values in sources], {"partition": "train"}, headers=headers, header_unit="nm")
    dataset.add_samples([values[24:] for values in sources], {"partition": "test"}, headers=headers, header_unit="nm")
    dataset.add_targets(y)
    steps = [StandardScaler(), KFold(3, shuffle=True, random_state=42), {"branch": {"by_source": True, "steps": {
        "source_0": [StandardScaler(), {"model": Ridge(alpha=.5), "name": "A"}],
        "source_1": [PCA(n_components=2), {"model": Ridge(alpha=.8), "name": "B"}],
    }}}, {"merge": "predictions"}, {"model": Ridge(alpha=.1), "name": "META"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, dataset, engine="legacy", refit=True, workspace_path=workspace,
                      save_artifacts=True, save_charts=False, verbose=0) as result:
        rows = list(result.predictions.iter_entries())
        finals = [row for row in rows if str(row["fold_id"]) == "final"]
        meta = [row for row in finals if row["model_name"] == "META"]
        assert meta and {row["partition"] for row in meta} >= {"train", "test"}
        assert {row["n_features"] for row in meta} == {2}
        bases = [row for row in finals if row["model_name"] in ("A", "B")]
        assert {(row["model_name"], row["branch_id"], row["n_features"]) for row in bases} == {("A", 0, 5), ("B", 1, 2)}
        selected = next(row for row in meta if row["partition"] == "train")

    # Independent full-TRAIN oracle: the established FINAL policy uses
    # in-sample predictions of full-TRAIN bases, with no held-out fitting.
    prefixes = [StandardScaler().fit(values[:24]) for values in sources]
    normalized = [prefix.transform(values) for prefix, values in zip(prefixes, sources, strict=True)]
    source_prefixes = [StandardScaler().fit(normalized[0][:24]), PCA(n_components=2).fit(normalized[1][:24])]
    features = [prefix.transform(values) for prefix, values in zip(source_prefixes, normalized, strict=True)]
    models = [Ridge(alpha=.5).fit(features[0][:24], y[:24]), Ridge(alpha=.8).fit(features[1][:24], y[:24])]
    merged = np.column_stack([model.predict(values) for model, values in zip(models, features, strict=True)])
    expected = Ridge(alpha=.1).fit(merged[:24], y[:24]).predict(merged)
    for row in meta:
        indices = np.asarray(row["sample_indices"], dtype=int)
        np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected[indices], rtol=1e-5, atol=1e-5)
    raw_x = np.hstack(sources)
    with WorkspaceStore(workspace) as store:
        chains = store.get_chains_for_pipeline(selected["pipeline_uid"]).to_dicts()
        chain = next(chain for chain in chains if selected["model_artifact_id"] in json.loads(chain["fold_artifacts"]).values())
        predictor = store.get_raw_input_predictor(chain["chain_id"])
        assert predictor is not None
        pending = [predictor.root]
        source_nodes = []
        replay_headers = []
        while pending:
            node = pending.pop()
            if node.operation == "source":
                source_nodes.append((node.state["source_id"], node.state["index"], node.state["widths"]))
            if node.operation == "transform":
                replay_headers.extend(node.state.headers)
            pending.extend(node.inputs)
        assert set(source_nodes) == {("source_0", 0, (5, 4)), ("source_1", 1, (5, 4))}
        assert all(header in replay_headers for header in headers)
        np.testing.assert_allclose(store.replay_chain(chain["chain_id"], raw_x), expected, rtol=1e-5, atol=1e-5)
        archive = store.export_chain(chain["chain_id"], tmp_path / "source-final.n4a")
    inputs = tmp_path / "source-inputs.npz"
    np.savez(inputs, raw_x=raw_x, expected=expected)
    probe = r'''
import pathlib, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
archive, inputs, workspace = sys.argv[1:]
data = np.load(inputs)
blocked = pathlib.Path(workspace).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if path == blocked or blocked in path.parents:
            raise AssertionError('Source stacking cold replay accessed its workspace')
sys.addaudithook(audit)
def no_fit(*args, **kwargs):
    raise AssertionError('Source stacking cold replay attempted FIT or opened a Store')
WorkspaceStore.__init__ = no_fit
Ridge.fit = StandardScaler.fit = PCA.fit = PCA.fit_transform = no_fit
with BundleLoader(archive) as loader:
    np.testing.assert_allclose(loader.predict(data['raw_x']), data['expected'], rtol=1e-5, atol=1e-5)
'''
    subprocess.run([sys.executable, "-B", "-c", probe, str(archive), str(inputs), str(workspace)], check=True)
