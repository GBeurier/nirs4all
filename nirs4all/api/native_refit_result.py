"""Public, scoreless result of one fresh native Methods full refit.

Package V3 deliberately records a new REFIT artifact rather than copying the
parent CV/SELECT score set.  This result therefore exposes identity-bound
PREDICT replay and its durable child package, but never invents a new
cross-validation score or a legacy workspace projection.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import PredictResult
from nirs4all.pipeline.dagml.fit_identity import normalize_predict_identity
from nirs4all.pipeline.dagml.native_client import DagMLNativeClient
from nirs4all.pipeline.dagml.raw_replay_lowerer import (
    RawArrayMethodsPortableRefitReplayCompiler,
)


class NativeMethodsRefitResult:
    """A detached V3 child with explicit-identity native prediction only."""

    def __init__(
        self,
        package: Any,
        *,
        methods_library_path: str,
        dagml_module: str,
        decoder: Callable[[Any, Any], np.ndarray],
    ) -> None:
        self._package = package
        self._methods_library_path = methods_library_path
        self._client = DagMLNativeClient(dagml_module)
        self._decoder = decoder

    @property
    def package(self) -> Any:
        """The exact DAG-ML Package V3 child returned by native REFIT."""

        return self._package

    def package_json(self) -> str:
        """Return the strict, self-fingerprinted Package V3 JSON payload."""

        serializer = getattr(self._package, "json", None)
        if not callable(serializer):
            raise RuntimeError("native Methods V3 package has no strict JSON serializer")
        package_json = serializer()
        if not isinstance(package_json, str):
            raise RuntimeError("native Methods V3 package serializer returned a non-string payload")
        return package_json

    def save_package(self, path: str | Path) -> Path:
        """Persist the signed V3 package without invoking a legacy exporter.

        This is intentionally a Package V3 JSON file, not an Archive V3
        container.  Core remains the owner of archive containers and their
        publication policy.  Existing paths are refused and a failed write is
        removed, so callers never receive a partially written replacement.
        """

        output = Path(path)
        payload = self.package_json().encode("utf-8")
        try:
            with output.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            with suppress(OSError):
                output.unlink(missing_ok=True)
            raise
        return output

    @classmethod
    def from_package_json(
        cls,
        package_json: str,
        *,
        methods_library_path: str,
        dagml_module: str,
        decoder: Callable[[Any, Any], np.ndarray],
    ) -> NativeMethodsRefitResult:
        """Load and strictly validate a detached V3 package JSON payload."""

        if not isinstance(package_json, str):
            raise TypeError("native Methods V3 package JSON must be a string")
        dag_ml = importlib.import_module(dagml_module)
        package_type = getattr(dag_ml, "PortableRefitPackageV3", None)
        if not callable(package_type):
            raise RuntimeError("installed DAG-ML lacks the strict Package V3 reader")
        package = package_type(package_json)
        return cls(
            package,
            methods_library_path=methods_library_path,
            dagml_module=dagml_module,
            decoder=decoder,
        )

    @classmethod
    def load_package(
        cls,
        path: str | Path,
        *,
        methods_library_path: str,
        dagml_module: str,
        decoder: Callable[[Any, Any], np.ndarray],
    ) -> NativeMethodsRefitResult:
        """Load one explicit Package V3 JSON file for native PREDICT replay."""

        package_json = Path(path).read_text(encoding="utf-8")
        return cls.from_package_json(
            package_json,
            methods_library_path=methods_library_path,
            dagml_module=dagml_module,
            decoder=decoder,
        )

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any,
        groups: Any = None,
        metadata: Any = None,
    ) -> PredictResult:
        """Run scheduler-owned PREDICT from the V3 raw N4MM child.

        ``sample_ids`` are mandatory.  No positional IDs, source estimator,
        Python model or host-sidecar is used by this operation.
        """

        identity = normalize_predict_identity(
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=True,
        )
        replay = RawArrayMethodsPortableRefitReplayCompiler(
            self._package,
            methods_library_path=self._methods_library_path,
            dagml_module=self._client.module_name,
        ).compile_replay(None, X, mode="predict", identity_frame=identity)
        if replay.methods_inputs is None or replay.methods_library_path is None:
            raise RuntimeError("V3 native replay compiler omitted Methods inputs")
        outcome = self._client.replay_loaded_methods_portable_refit_package_v3(
            self._package,
            replay.request,
            replay.data_envelopes,
            replay.methods_inputs,
            methods_library_path=replay.methods_library_path,
            outcome_id=replay.outcome_id,
            run_id=replay.run_id,
            warnings=replay.warnings,
            diagnostics=replay.diagnostics,
        )
        values = self._decoder(outcome, identity)
        return PredictResult(
            y_pred=values,
            metadata={"engine": "native", "sample_ids": list(identity.sample_ids)},
            model_name="MethodsN4MM",
            preprocessing_steps=[],
        )

    def export(
        self,
        output_path: str | Path,
        *,
        archive_id: str | None = None,
        core_module: str = "nirs4all_core",
        **kwargs: Any,
    ) -> Path:
        """Write a strict Core Archive V3 without any legacy bundle fallback.

        DAG-ML assembles the exact V3 semantic closure from this child package.
        Core alone writes and validates the ZIP container.  The optional
        ``archive_id`` is passed to the DAG-ML assembler; a deterministic,
        portable identifier is used when it is omitted.
        """

        if kwargs:
            raise TypeError(f"native Archive V3 export does not accept options: {sorted(kwargs)}")
        package_json = self.package_json()
        if archive_id is None:
            archive_id = f"native-refit:{sha256(package_json.encode('utf-8')).hexdigest()[:32]}"
        if not isinstance(archive_id, str):
            raise TypeError("archive_id must be a string when supplied")
        dag_ml = importlib.import_module(self._client.module_name)
        assemble = getattr(dag_ml, "build_archive_v3_native_refit_payloads", None)
        if not callable(assemble):
            raise RuntimeError("installed DAG-ML lacks the Archive V3 native refit assembler")
        try:
            core = importlib.import_module(core_module)
        except ImportError as error:
            raise RuntimeError(
                "native Archive V3 export requires the matching nirs4all-core native bridge"
            ) from error
        write_archive = getattr(core, "write_archive_v3_from_native_payloads", None)
        if not callable(write_archive):
            raise RuntimeError("installed nirs4all-core lacks the Archive V3 writer")
        manifest, members = assemble(archive_id, self._package)
        write_archive(output_path, manifest, members)
        return Path(output_path)

    @classmethod
    def load_archive(
        cls,
        path: str | Path,
        *,
        methods_library_path: str,
        dagml_module: str,
        decoder: Callable[[Any, Any], np.ndarray],
        core_module: str = "nirs4all_core",
    ) -> NativeMethodsRefitResult:
        """Load a Core-validated Archive V3, then strictly parse Package V3."""

        try:
            core = importlib.import_module(core_module)
        except ImportError as error:
            raise RuntimeError(
                "native Archive V3 load requires the matching nirs4all-core native bridge"
            ) from error
        read_archive = getattr(core, "read_portable_refit_package_v3", None)
        if not callable(read_archive):
            raise RuntimeError("installed nirs4all-core lacks the Archive V3 reader")
        package_bytes = read_archive(path)
        if not isinstance(package_bytes, bytes):
            raise RuntimeError("Core Archive V3 reader returned non-bytes package data")
        return cls.from_package_json(
            package_bytes.decode("utf-8"),
            methods_library_path=methods_library_path,
            dagml_module=dagml_module,
            decoder=decoder,
        )


__all__ = ["NativeMethodsRefitResult"]
