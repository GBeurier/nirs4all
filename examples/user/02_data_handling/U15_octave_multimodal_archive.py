"""Tune four numeric sources in Octave, export and replay five native models.

Run ``python U15_octave_multimodal_archive.py --help`` for the train/predict
commands. Requires matching SDK/DAG/Core wheels, Octave and compiled Methods
MEX bindings. Input is the existing Node/R qualification capture; this example
does not generate data or implement numerical models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import nirs4all


def adapter_tools(root: Path) -> Any:
    """Load the shipped Octave controller's preparation and worker helpers."""
    source = root / "scripts/qualify_multimodal_methods_hpo_octave.py"
    spec = importlib.util.spec_from_file_location("octave_controller_tools", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Octave adapter helpers unavailable at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train(args: argparse.Namespace, tools: Any) -> None:
    """Native HPO, CV/SELECT/REFIT and portable capture through public APIs."""
    args.workdir.mkdir(parents=True, exist_ok=False)
    trained = tools.train_from_python_api(
        args.node_capture, args.octave, args.workdir / "training",
        hpo_search=nirs4all.run_host_hpo_search, execute_training=nirs4all.execute_training,
    )
    archive = args.workdir / "five-octave-models.n4a"
    reference = nirs4all.write_portable_predictor_archive_v2(
        archive, archive_id="archive:octave.four-sources", outcome=trained["outcome"], package=trained["package"],
    )
    current = {
        "heldout": trained["heldout"], "operators": trained["operators"],
        "replay_request": trained["replay_request"].json(), "replay_envelopes": trained["replay_envelopes"],
    }
    (args.workdir / "heldout-replay.json").write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    search = trained["search"]
    print(json.dumps({"archive": str(archive.resolve()), "archive_sha256": reference["archive_sha256"],
                      "trials": len(search["trials"]), "selected_trial_index": search["selected_trial_index"]}))


def predict(args: argparse.Namespace, tools: Any) -> None:
    """A fresh target-free Octave worker hydrates the saved native states."""
    args.workdir.mkdir(parents=True, exist_ok=False)
    current = json.loads(args.replay_inputs.read_text())
    prepared = tools.prepare_octave(args.octave, args.workdir, "prediction", current["heldout"], current["operators"], None)
    package = nirs4all.read_portable_predictor_archive_v2(args.archive)
    with tools.OctaveWorker(prepared) as worker:
        outcome = nirs4all.replay_portable_predictor_archive_v2(
            args.archive, current["replay_request"], current["replay_envelopes"], [tools.OCTAVE_MANIFEST], worker.operator,
            outcome_id="outcome:octave.heldout", run_id="run:octave.heldout", artifact_callback=worker.artifact,
        ).to_dict()
    lifecycle = Counter(json.loads(line)["operation"] for line in Path(prepared["audit_path"]).read_text().splitlines())
    if lifecycle["fit"] or worker.hydrated:
        raise RuntimeError("Prediction performed training or retained native model states")
    result = {"predictor_models": len(package.to_dict()["artifact_bindings"]),
              "predictions": outcome["outputs"][0]["predictions"], "lifecycle": dict(lifecycle)}
    (args.workdir / "predictions.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-ml-root", type=Path, required=True, help="DAG-ML checkout containing the shipped Octave adapter")
    parser.add_argument("--octave", type=Path, required=True, help="Real Octave executable")
    parser.add_argument("--workdir", type=Path, required=True, help="New directory for outputs")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Run three recorded proposals and capture five native models")
    training.add_argument("--node-capture", type=Path, required=True, help="Fresh four-source Node/R fixture capture")
    prediction = commands.add_parser("predict", help="Replay the archive in a fresh Octave process without FIT")
    prediction.add_argument("--archive", type=Path, required=True)
    prediction.add_argument("--replay-inputs", type=Path, required=True, help="Target-free heldout-replay.json written by train")
    args = parser.parse_args()
    tools = adapter_tools(args.dag_ml_root.resolve(strict=True))
    if args.command == "train":
        train(args, tools)
    else:
        predict(args, tools)


if __name__ == "__main__":
    main()
