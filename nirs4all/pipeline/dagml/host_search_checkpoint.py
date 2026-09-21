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

    def __init__(self, tuning: DagMLTuningSpec) -> None:
        self.tuning = tuning
        self.api = adapters._import_n4m_optimizer()
        space, self.slots = adapters._make_n4m_space(self.api, tuning.space)
        self.path = adapters._n4m_checkpoint_path(tuning)
        self.resume_checkpoint = None
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
            self.optimizer = self.api.Optimizer(
                space, sampler=adapters._n4m_enum(self.api.Sampler, adapters._n4m_sampler_name(tuning.sampler)),
                direction=adapters._n4m_enum(self.api.Direction, tuning.direction), seed=tuning.seed or 0,
            )
            adapters._enqueue_n4m_force_params(self.optimizer, tuning, adapters._slot_categorical_codecs(self.slots))
        self.pending: dict[int, Any] = {}

    def _validate_history(self, checkpoint: dict[str, Any]) -> None:
        records = self.optimizer.get_trials()
        trials = checkpoint["trials"]
        if len(records) != len(trials):
            raise ValueError("native DAG and optimizer checkpoint trial counts disagree")
        for record, trial in zip(records, trials, strict=True):
            evidence = trial.get("evidence", trial)
            params = adapters._decode_n4m_record_params(record.params, self.slots)
            complete = trial["state"] == "complete"
            if (record.id != evidence["trial_index"] or params != evidence["params"]
                    or adapters._n4m_trial_state(record.status) != ("COMPLETE" if complete else "FAIL")
                    or (complete and record.score != evidence["score"])):
                raise ValueError("native DAG and optimizer checkpoint histories disagree")

    def __call__(self, event: dict[str, Any]) -> Any:
        index = event["trial_index"]
        if event["operation"] == "ask":
            trial = self.optimizer.ask()
            if trial.id != index:
                raise ValueError("native DAG and optimizer trial IDs disagree")
            self.pending[index] = trial
            return adapters._n4m_trial_params(trial, self.slots)
        trial = self.pending.pop(index)
        if event["operation"] == "tell":
            self.optimizer.tell(trial.id, event["score"])
        elif event["operation"] == "fail":
            self.optimizer.tell_result(trial.id, self.api.TrialStatus.FAILED, error=str(event["error"])[:200])
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
