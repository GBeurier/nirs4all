"""
PipelineTree - Fitted pipeline persistence with structure preservation

Mirrors the exact structure of pipeline config but with fitted objects
Supports optimal saving for different frameworks (sklearn, torch, tensorflow, etc.)
"""
import json
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class PipelineTree:
    """
    Mirrors the exact structure of pipeline config but with fitted objects

    Structure examples:
    - Linear: [op1, op2, op3] → [fitted_op1, fitted_op2, fitted_op3]
    - Dispatch: {"dispatch": [branch1, branch2]} → {"dispatch": [fitted_branch1, fitted_branch2]}
    - Nested: [op1, {"dispatch": [...]}, op2] → [fitted_op1, {"dispatch": [...]}, fitted_op2]
    """
    def __init__(self, structure: Any = None):
        self.structure = structure  # Mirrors original config structure
        self.metadata = {}
        self._object_counter = 0
        self.fitted_objects = {}  # Simple dict for storing fitted objects

    def add_fitted_object(self, key: str, fitted_obj: Any):
        """Add a fitted object to the tree"""
        self.fitted_objects[key] = fitted_obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary for serialization"""
        return {
            "structure": self.structure,
            "metadata": self.metadata,
            "fitted_objects": self.fitted_objects
        }

    def save(self, filepath: Union[str, Path], additional_metadata: Optional[Dict] = None):
        """Save the tree structure with each object using its optimal format"""
        path = Path(filepath)

        # For simple cases, just use pickle
        if self.fitted_objects:
            save_data = {
                "structure": self.structure,
                "metadata": {**self.metadata, **(additional_metadata or {})},
                "fitted_objects": self.fitted_objects
            }

            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"💾 Pipeline tree saved to {path}")
            return path

        # Original complex save logic for structured pipelines
        base_dir = path.parent / f"{path.stem}_pipeline"
        base_dir.mkdir(exist_ok=True)

        # Save structure map and metadata
        structure_map, object_refs = self._create_structure_map()

        with open(base_dir / "structure.json", "w") as f:
            json.dump({
                "structure": structure_map,
                "metadata": {**self.metadata, **(additional_metadata or {})},
                "object_refs": object_refs
            }, f, indent=2, default=str)

        # Save each fitted object with its optimal method
        self._save_objects(base_dir)

        print(f"📁 Pipeline tree saved to {base_dir}")
        return base_dir

    def _create_structure_map(self):
        """Create JSON-serializable structure map with object references"""
        object_refs = {}

        def map_structure(item, path="root"):
            if isinstance(item, list):
                return [map_structure(subitem, f"{path}[{i}]")
                       for i, subitem in enumerate(item)]

            elif isinstance(item, dict):
                if "dispatch" in item:
                    return {
                        "dispatch": [map_structure(branch, f"{path}.dispatch[{i}]")
                                   for i, branch in enumerate(item["dispatch"])],
                        **{k: v for k, v in item.items() if k != "dispatch"}
                    }
                else:
                    return {k: map_structure(v, f"{path}.{k}")
                           for k, v in item.items()}

            else:
                # This is a fitted object - create reference
                obj_id = f"obj_{len(object_refs)}"
                object_refs[obj_id] = {
                    "path": path,
                    "type": type(item).__name__,
                    "module": type(item).__module__,
                    "save_method": self._get_save_method(item)
                }
                return {"$ref": obj_id}

        structure_map = map_structure(self.structure)
        return structure_map, object_refs

    def _get_save_method(self, obj) -> str:
        """Determine optimal save method for each object type"""
        obj_type_str = str(type(obj))

        if hasattr(obj, 'save') and 'tensorflow' in obj_type_str.lower():
            return "tensorflow"
        elif hasattr(obj, 'state_dict') and 'torch' in obj_type_str.lower():
            return "pytorch"
        elif hasattr(obj, 'save_model') and 'xgboost' in obj_type_str.lower():
            return "xgboost"
        elif hasattr(obj, 'get_params') and hasattr(obj, 'set_params'):
            return "sklearn"
        else:
            return "pickle"

    def _save_objects(self, base_dir: Path):
        """Save each object using its optimal method"""
        objects_dir = base_dir / "objects"
        objects_dir.mkdir(exist_ok=True)

        self._object_refs_for_saving = {}

        def save_recursive(item, path="root"):
            if isinstance(item, list):
                for i, subitem in enumerate(item):
                    save_recursive(subitem, f"{path}[{i}]")

            elif isinstance(item, dict):
                if "dispatch" in item:
                    for i, branch in enumerate(item["dispatch"]):
                        save_recursive(branch, f"{path}.dispatch[{i}]")
                else:
                    for k, v in item.items():
                        save_recursive(v, f"{path}.{k}")

            else:
                # Save the actual fitted object
                obj_id = self._get_object_id_for_path(path)
                if obj_id:
                    save_method = self._get_save_method(item)

                    try:
                        if save_method == "tensorflow":
                            item.save(objects_dir / f"{obj_id}.tf")

                        elif save_method == "pytorch":
                            import torch
                            torch.save(item.state_dict(), objects_dir / f"{obj_id}.pth")
                            # Also save model class info
                            with open(objects_dir / f"{obj_id}.pth.meta", "w") as f:
                                json.dump({
                                    "class": f"{item.__class__.__module__}.{item.__class__.__name__}",
                                    "constructor_args": getattr(item, '_constructor_args', {})
                                }, f)

                        elif save_method == "xgboost":
                            item.save_model(objects_dir / f"{obj_id}.xgb")

                        elif save_method == "sklearn":
                            import joblib
                            joblib.dump(item, objects_dir / f"{obj_id}.joblib")

                        else:  # pickle fallback
                            with open(objects_dir / f"{obj_id}.pkl", "wb") as f:
                                pickle.dump(item, f)

                    except Exception as e:
                        print(f"⚠️ Failed to save {obj_id} with {save_method}, using pickle: {e}")
                        with open(objects_dir / f"{obj_id}.pkl", "wb") as f:
                            pickle.dump(item, f)

        save_recursive(self.structure)

    def _get_object_id_for_path(self, path: str) -> Optional[str]:
        """Get object ID for a given path during saving"""
        # This is a simplified approach - in practice you'd match with the refs
        return f"obj_{abs(hash(path)) % 10000}"

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PipelineTree':
        """Load pipeline tree from saved structure"""
        path = Path(filepath)

        # Try simple pickle format first
        if path.is_file() and path.suffix == '.pkl':
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                tree = cls(data.get("structure"))
                tree.metadata = data.get("metadata", {})
                tree.fitted_objects = data.get("fitted_objects", {})
                return tree
            except Exception:
                # Fall back to complex format
                pass

        # Handle complex directory structure
        if path.is_file():
            base_dir = path.parent / f"{path.stem}_pipeline"
        elif path.is_dir():
            base_dir = path
        else:
            base_dir = path.parent / f"{path.name}_pipeline"

        # Load structure map
        with open(base_dir / "structure.json", "r") as f:
            data = json.load(f)

        structure_map = data["structure"]
        object_refs = data["object_refs"]
        metadata = data["metadata"]

        # Load all objects
        loaded_objects = cls._load_objects(base_dir / "objects", object_refs)

        # Reconstruct structure with loaded objects
        structure = cls._reconstruct_structure(structure_map, loaded_objects)

        tree = cls(structure)
        tree.metadata = metadata
        return tree

    @classmethod
    def _load_objects(cls, objects_dir: Path, object_refs: dict):
        """Load all objects using their original save methods"""
        loaded = {}

        for obj_id, ref_info in object_refs.items():
            save_method = ref_info["save_method"]

            try:
                if save_method == "tensorflow":
                    import tensorflow as tf
                    loaded[obj_id] = tf.keras.models.load_model(objects_dir / f"{obj_id}.tf")

                elif save_method == "pytorch":
                    import torch
                    # Load model metadata first
                    with open(objects_dir / f"{obj_id}.pth.meta", "r") as f:
                        meta = json.load(f)

                    # This is simplified - in practice you'd need to recreate the model architecture
                    state_dict = torch.load(objects_dir / f"{obj_id}.pth")
                    loaded[obj_id] = state_dict  # Return state dict for now

                elif save_method == "xgboost":
                    import xgboost as xgb
                    model = xgb.Booster()
                    model.load_model(objects_dir / f"{obj_id}.xgb")
                    loaded[obj_id] = model

                elif save_method == "sklearn":
                    import joblib
                    loaded[obj_id] = joblib.load(objects_dir / f"{obj_id}.joblib")

                else:  # pickle
                    with open(objects_dir / f"{obj_id}.pkl", "rb") as f:
                        loaded[obj_id] = pickle.load(f)

            except Exception as e:
                print(f"⚠️ Failed to load {obj_id}: {e}")
                # Try pickle as fallback
                try:
                    with open(objects_dir / f"{obj_id}.pkl", "rb") as f:
                        loaded[obj_id] = pickle.load(f)
                except:
                    print(f"❌ Could not load {obj_id} at all")
                    loaded[obj_id] = None

        return loaded

    @classmethod
    def _reconstruct_structure(cls, structure_map, loaded_objects):
        """Reconstruct original structure with loaded objects"""
        def reconstruct(item):
            if isinstance(item, list):
                return [reconstruct(subitem) for subitem in item]

            elif isinstance(item, dict):
                if "$ref" in item:
                    # Replace reference with actual loaded object
                    return loaded_objects.get(item["$ref"])
                elif "dispatch" in item:
                    return {
                        "dispatch": [reconstruct(branch) for branch in item["dispatch"]],
                        **{k: v for k, v in item.items() if k != "dispatch"}
                    }
                else:
                    return {k: reconstruct(v) for k, v in item.items()}

            else:
                return item

        return reconstruct(structure_map)
