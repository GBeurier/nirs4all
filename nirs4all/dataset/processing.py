from __future__ import annotations
from pathlib import Path
from typing import Hashable, Mapping, Any
import json


class TransformationPath(tuple):
    """Chemin de transformation immutable."""

    def plus(self, step: Hashable) -> "TransformationPath":
        """Ajoute une étape au chemin."""
        return TransformationPath((*self, step))

    @classmethod
    def from_string(cls, s: str) -> "TransformationPath":
        """Crée un chemin depuis une chaîne."""
        if not s:
            return cls()
        return cls(s.split("→"))


class ProcessingRegistry:
    """
    TransformationPath ↔ processing_id (+ meta: n_features, dtype…).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._path_to_id: dict[TransformationPath, int] = {}
        self._id_to_path: dict[int, TransformationPath] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._next_id = 0

    def get_or_create(self,
                      path: TransformationPath,
                      n_features: int,
                      dtype,
                      meta: Mapping[str, Any] | None = None) -> int:
        """Obtient ou crée un ID de processing."""
        if path in self._path_to_id:
            return self._path_to_id[path]

        proc_id = self._next_id
        self._next_id += 1

        self._path_to_id[path] = proc_id
        self._id_to_path[proc_id] = path

        metadata = {
            "n_features": n_features,
            "dtype": str(dtype),
            "path": "→".join(str(step) for step in path)
        }
        if meta:
            metadata.update(meta)

        self._metadata[proc_id] = metadata
        return proc_id

    def meta(self, proc_id: int) -> Mapping[str, Any]:
        """Récupère les métadonnées d'un processing."""
        return self._metadata.get(proc_id, {})

    def chain(self, proc_id: int) -> TransformationPath:
        """Récupère le chemin de transformation d'un processing."""
        return self._id_to_path.get(proc_id, TransformationPath())

    def save(self) -> None:
        """Sauvegarde le registre."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            registry_data = {
                "path_to_id": {str(path): proc_id for path, proc_id in self._path_to_id.items()},
                "id_to_path": {proc_id: list(path) for proc_id, path in self._id_to_path.items()},
                "metadata": self._metadata,
                "next_id": self._next_id
            }

            with open(self.cache_dir / "processing_registry.json", "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2)

    def load(self) -> None:
        """Charge le registre."""
        if self.cache_dir:
            registry_file = self.cache_dir / "processing_registry.json"
            if registry_file.exists():
                with open(registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._path_to_id = {
                    TransformationPath.from_string(path_str): proc_id
                    for path_str, proc_id in data.get("path_to_id", {}).items()
                }
                self._id_to_path = {
                    proc_id: TransformationPath(path_list)
                    for proc_id, path_list in data.get("id_to_path", {}).items()
                }
                self._metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
                self._next_id = data.get("next_id", 0)
