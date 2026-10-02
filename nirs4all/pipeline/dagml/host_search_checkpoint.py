"""Atomic paired native DAG search and N4MOPT optimizer checkpoints."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from . import tuning_adapters as adapters
from .tuning_contracts import DagMLTuningSpec, tcv1_sha256


class HostSearchOptimizer:
    """Translate native ask/tell/fail events without owning evaluation or selection."""

    def __init__(self, tuning: DagMLTuningSpec, *, n_folds: int, structural_catalogue: dict[str, Any] | None = None) -> None:
        self.tuning = tuning
        self.structural_catalogue = copy.deepcopy(structural_catalogue)
        self.api = adapters._import_n4m_optimizer()
        space, self.slots = (adapters._make_n4m_space(self.api, tuning.space) if self.structural_catalogue is None
                            else adapters._make_n4m_space(self.api, tuning.space, structural_catalogue=self.structural_catalogue))
        self.path = adapters._n4m_checkpoint_path(tuning)
        self.resume_checkpoint = None
        self.pruned: set[int] = set()
        if tuning.resume:
            if self.path is None or not self.path.exists():
                raise ValueError("multimodal resume requires an existing paired native checkpoint")
            payload = json.loads(self.path.read_text())
            pair = self._pair_preimage(payload)
            if "native_checkpoint" not in pair or payload.get("pair_fingerprint") != tcv1_sha256(pair):
                raise ValueError("multimodal paired checkpoint fingerprint mismatch")
            saved_structure = payload.get("structural_binding")
            if self.structural_catalogue is None:
                if saved_structure is not None:
                    raise ValueError("structural checkpoint cannot resume through a fixed-topology search")
            elif not isinstance(saved_structure, dict) or saved_structure.get("catalogue") != self.structural_catalogue:
                raise ValueError("structural checkpoint catalogue or activation contract mismatch")
            self.optimizer = adapters._load_n4m_optimizer_checkpoint(self.api, tuning, self.path)
            self.resume_checkpoint = payload["native_checkpoint"]
            try:
                if self.structural_catalogue is not None:
                    self._validate_native_configuration(space, n_folds=n_folds)
                if self.structural_catalogue is not None and saved_structure != self._structural_binding():
                    raise ValueError("structural checkpoint native activation masks disagree")
                self._validate_history(self.resume_checkpoint)
            except BaseException:
                self.optimizer.close()
                raise
        else:
            adapters._reject_existing_n4m_checkpoint_without_resume(tuning, self.path)
            self.optimizer = self.api.Optimizer(space, **self._optimizer_options(n_folds=n_folds))
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

    def _optimizer_options(self, *, n_folds: int) -> dict[str, Any]:
        options = {
            "sampler": adapters._n4m_enum(self.api.Sampler, adapters._n4m_sampler_name(self.tuning.sampler)),
            "direction": adapters._n4m_enum(self.api.Direction, self.tuning.direction),
            "seed": self.tuning.seed or 0,
        }
        pruner = adapters._n4m_pruner(self.tuning, self.api)
        if pruner is not None:
            options.update(pruner=pruner, n_startup_trials=10,
                           max_resource=n_folds if self.tuning.pruner == "hyperband" else 0,
                           reduction_factor=0)
        return options

    def _validate_native_configuration(self, space: Any, *, n_folds: int) -> None:
        """Compare immutable native configuration before reading or asking trials.

        Methods compares its own ordered space, constraints and normalized
        options. No host parsing of N4MOPT or inference from observed history
        can attest an as-yet unseen recipe. The total trial budget is a DAG
        control, not a native optimizer option, and may increase on resume.
        """
        matches = getattr(self.optimizer, "configuration_matches", None)
        if not callable(matches):
            raise ValueError("structural resume requires native Optimizer.configuration_matches(...); upgrade n4m bindings and library")
        expected = self.api.Optimizer(space, **self._optimizer_options(n_folds=n_folds))
        try:
            if matches(expected) is not True:
                raise ValueError("native optimizer checkpoint space or options contract mismatch")
        finally:
            expected.close()

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
            params = self._record_params(record)
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
                return self._record_params(trial)
            trial = self.optimizer.ask()
            if trial.id != index:
                raise ValueError("native DAG and optimizer trial IDs disagree")
            self.pending[index] = trial
            params = (adapters._n4m_trial_params(trial, self.slots) if self.structural_catalogue is None
                      else adapters._n4m_trial_params(trial, self.slots, active_only=True))
            self._validate_structural_params(params)
            return params
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
        if self.structural_catalogue is not None:
            payload["structural_binding"] = self._structural_binding()
        payload["pair_fingerprint"] = tcv1_sha256(self._pair_preimage(payload))
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

    def _pair_preimage(self, payload: dict[str, Any]) -> dict[str, Any]:
        keys = ["checkpoint_fingerprint", "native_checkpoint"]
        if "structural_binding" in payload:
            keys.append("structural_binding")
        return {key: payload[key] for key in keys if key in payload}

    def _record_params(self, record: Any) -> dict[str, Any]:
        params = (adapters._n4m_active_record_params(record, self.slots) if self.structural_catalogue is not None
                  else adapters._decode_n4m_record_params(record.params, self.slots))
        self._validate_structural_params(params)
        return params

    def _validate_structural_params(self, params: dict[str, Any]) -> None:
        if self.structural_catalogue is None:
            return
        selector = self.structural_catalogue["selector_path"]
        recipe = next((entry for entry in self.structural_catalogue["entries"] if entry["recipe_id"] == params.get(selector)), None)
        if recipe is None or set(params) != {selector, *recipe["parameter_bindings"]}:
            raise ValueError("structural proposal recipe or active parameter mask mismatch")
        slots = {path: (kind, codec) for path, kind, codec in self.slots}
        for path in recipe["parameter_bindings"]:
            value = params[path]
            kind, codec = slots[path]
            valid = (type(value) is int if path == "model.n_components"
                     else not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value))
            if codec is not None:
                choices = codec.decoder.values() if codec.decoder is not None else codec.choices
                valid = valid and value in choices
            else:
                declaration = self.tuning.space[path]
                # Native float step=0 is continuous, whereas the shared range
                # validator represents that same contract with step=None.
                if isinstance(declaration, dict) and kind == "float" and declaration.get("step") == 0:
                    declaration = {**declaration, "step": None}
                valid = valid and adapters._optuna_resume_value_matches_space_spec(value, declaration)
            if not valid:
                raise ValueError(f"structural proposal value is outside the declared search domain: {path}")

    def _structural_binding(self) -> dict[str, Any]:
        masks = []
        for record in self.optimizer.get_trials():
            params = self._record_params(record)
            masks.append({"trial_index": record.id, "active_paths": sorted(params)})
        return {"catalogue": self.structural_catalogue, "activation_masks": masks}

    def close(self) -> None:
        self.optimizer.close()
