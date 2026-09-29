"""Atomic paired native DAG search and N4MOPT optimizer checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import tuning_adapters as adapters
from .tuning_contracts import DagMLTuningSpec, tcv1_sha256


class HostSearchOptimizer:
    """Translate native ask/tell/fail events without owning evaluation or selection."""

    def __init__(self, tuning: DagMLTuningSpec, *, n_folds: int) -> None:
        self.tuning = tuning
        self.api = adapters._import_n4m_optimizer()
        space, self.slots = adapters._make_n4m_space(self.api, tuning.space)
        self.path = adapters._n4m_checkpoint_path(tuning)
        self.resume_checkpoint = None
        self.pruned: set[int] = set()
        if tuning.resume:
            if self.path is None or not self.path.exists():
                raise ValueError("multimodal resume requires an existing paired native checkpoint")
            payload = json.loads(self.path.read_text())
            pair = {key: payload[key] for key in ("checkpoint_fingerprint", "native_checkpoint") if key in payload}
            if "native_checkpoint" not in pair or payload.get("pair_fingerprint") != tcv1_sha256(pair):
                raise ValueError("multimodal paired checkpoint fingerprint mismatch")
            self.optimizer = adapters._load_n4m_optimizer_checkpoint(self.api, tuning, self.path)
            self.resume_checkpoint = payload["native_checkpoint"]
            try:
                self._validate_history(self.resume_checkpoint)
            except Exception:
                self.optimizer.close()
                raise
        else:
            adapters._reject_existing_n4m_checkpoint_without_resume(tuning, self.path)
            pruner = adapters._n4m_pruner(tuning, self.api)
            options = {
                "sampler": adapters._n4m_enum(self.api.Sampler, adapters._n4m_sampler_name(tuning.sampler)),
                "direction": adapters._n4m_enum(self.api.Direction, tuning.direction),
                "seed": tuning.seed or 0,
            }
            if pruner is not None:
                options.update(pruner=pruner, n_startup_trials=10,
                               max_resource=n_folds if tuning.pruner == "hyperband" else 0,
                               reduction_factor=0)
            self.optimizer = self.api.Optimizer(space, **options)
            adapters._enqueue_n4m_force_params(self.optimizer, tuning, adapters._slot_categorical_codecs(self.slots))
        # Parallel DAG-ML may have asked ahead of its contiguous terminal
        # checkpoint. Keep those RUNNING native trials so resume replays their
        # original IDs and parameters instead of asking for new proposals.
        self.pending: dict[int, Any] = (
            {
                record.id: record
                for record in self.optimizer.get_trials()[len(self.resume_checkpoint["trials"]):]
            }
            if self.resume_checkpoint is not None else {}
        )

    def _validate_history(self, checkpoint: dict[str, Any]) -> None:
        records = self.optimizer.get_trials()
        trials = checkpoint["trials"]
        if len(records) < len(trials):
            raise ValueError("native DAG and optimizer checkpoint trial counts disagree")
        if any(record.id != index or adapters._n4m_trial_state(record.status) != "RUNNING"
               for index, record in enumerate(records[len(trials):], start=len(trials))):
            raise ValueError("native DAG and optimizer checkpoint has non-running pending trials")
        for record, trial in zip(records[:len(trials)], trials, strict=True):
            evidence = trial.get("evidence", trial)
            params = adapters._decode_n4m_record_params(record.params, self.slots)
            state = {"complete": "COMPLETE", "pruned": "PRUNED", "failed": "FAIL"}.get(trial["state"])
            if state is None:
                raise ValueError("native DAG checkpoint has an unsupported terminal state")
            if (record.id != evidence["trial_index"] or params != evidence["params"]
                    or adapters._n4m_trial_state(record.status) != state
                    or (state == "COMPLETE" and record.score != evidence["score"])):
                raise ValueError("native DAG and optimizer checkpoint histories disagree")

    def __call__(self, event: dict[str, Any]) -> Any:
        index = event["trial_index"]
        if event["operation"] == "ask":
            trial = self.pending.get(index)
            if trial is not None:
                return adapters._decode_n4m_record_params(trial.params, self.slots)
            trial = self.optimizer.ask()
            if trial.id != index:
                raise ValueError("native DAG and optimizer trial IDs disagree")
            self.pending[index] = trial
            return adapters._n4m_trial_params(trial, self.slots)
        if event["operation"] == "report_intermediate":
            if index in self.pruned:
                raise ValueError("native optimizer received feedback after pruning")
            should_prune = self.optimizer.tell_intermediate(index, event["step"], event["score"])
            if should_prune:
                self.pruned.add(index)
            return should_prune
        trial = self.pending.pop(index)
        if event["operation"] == "tell":
            if index in self.pruned:
                raise ValueError("native optimizer cannot complete a pruned trial")
            self.optimizer.tell(trial.id, event["score"])
        elif event["operation"] == "fail":
            if index in self.pruned:
                raise ValueError("native optimizer cannot fail an already pruned trial")
            self.optimizer.tell_result(trial.id, self.api.TrialStatus.FAILED, error=str(event["error"])[:200])
        elif event["operation"] == "pruned":
            if index not in self.pruned:
                raise ValueError("native optimizer did not prune this trial")
            self.pruned.remove(index)
        else:
            raise ValueError("unexpected native optimizer event")
        return None

    def checkpoint(self, event: dict[str, Any]) -> None:
        """Replace both states together only after a terminal native transition."""
        native = event["checkpoint"]
        self._validate_history(native)
        if self.path is None:
            return
        payload = adapters._n4m_checkpoint_manifest(self.tuning, self.optimizer.save())
        payload["native_checkpoint"] = native
        payload["pair_fingerprint"] = tcv1_sha256({
            "checkpoint_fingerprint": payload["checkpoint_fingerprint"], "native_checkpoint": native,
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, prefix=".host-search-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self.optimizer.close()
