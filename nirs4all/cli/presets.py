import json
import os
from nirs4all.core.config import Config
from typing import Dict, Any
import copy

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

def load_preset(preset_name: str) -> Dict[str, Any]:
    """
    Loads a preset JSON file from the presets directory.
    """
    if not preset_name.endswith(".json"):
        preset_filename = f"{preset_name}.json"
    else:
        preset_filename = preset_name
    preset_path = os.path.join(PRESET_DIR, preset_filename)
    if os.path.exists(preset_path):
        try:
            with open(preset_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from preset file {preset_path}: {e}")
        except Exception as e:
            raise IOError(f"Error reading preset file {preset_path}: {e}")
    raise FileNotFoundError(f"Preset '{preset_name}' (as '{preset_filename}') not found in {PRESET_DIR}.")

def _deep_update(source: Dict, overrides: Dict) -> Dict:
    source = copy.deepcopy(source)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source

def apply_preset_to_config(config: Config, preset_data: Dict[str, Any]) -> Config:
    if not isinstance(config, Config):
        if isinstance(config, dict):
            config = Config.from_dict(config)
        else:
            raise TypeError("config must be an instance of Config or a compatible dict")
    if not isinstance(preset_data, dict):
        raise TypeError("preset_data must be a dictionary")
    config_dict = config.to_dict()
    updated_dict = _deep_update(config_dict, preset_data)
    return Config.from_dict(updated_dict)

# Example usage (for testing purposes, can be removed or kept for module tests)
if __name__ == '__main__':
    # Create dummy preset files for testing
    os.makedirs(os.path.join(PRESET_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(PRESET_DIR, "predict"), exist_ok=True)

    dummy_train_preset_path = os.path.join(PRESET_DIR, "train", "sample_train.json")
    with open(dummy_train_preset_path, "w") as f:
        json.dump({
            "dataset": "path/from/train_preset",
            "model": {"type": "NeuralNetwork", "layers": [128, 64]},
            "experiment": {"epochs": 50, "batch_size": 64}
        }, f, indent=2)

    dummy_generic_preset_path = os.path.join(PRESET_DIR, "generic_settings.json")
    with open(dummy_generic_preset_path, "w") as f:
        json.dump({
            "seed": 42,
            "experiment": {"tracking_uri": "http://localhost:5000"}
        }, f, indent=2)

    print(f"PRESET_DIR is: {PRESET_DIR}")
    print(f"Created dummy preset: {dummy_train_preset_path}")
    print(f"Created dummy preset: {dummy_generic_preset_path}")

    try:
        # Test loading
        train_preset = load_preset("sample_train")
        print("\\nLoaded train_preset:", json.dumps(train_preset, indent=2))
        
        generic_preset = load_preset("generic_settings")
        print("\\nLoaded generic_preset:", json.dumps(generic_preset, indent=2))

        # Test applying preset
        base_config_data = {
            "dataset": "original/path",
            "model": {"type": "OldModel", "layers": [10]},
            "experiment": {"epochs": 10, "batch_size": 32, "extra_param": "keep_this"},
            "seed": 123
        }
        base_config = Config.from_dict(base_config_data)
        print("\\nBase config:", json.dumps(base_config.to_dict(), indent=2))

        # Apply train_preset
        config_after_train_preset = apply_preset_to_config(base_config, train_preset)
        print("\\nAfter applying train_preset:", json.dumps(config_after_train_preset.to_dict(), indent=2))
        # Expected: dataset, model, experiment.epochs, experiment.batch_size updated. seed and extra_param kept.

        # Apply generic_preset to the result
        config_after_generic_preset = apply_preset_to_config(config_after_train_preset, generic_preset)
        print("\\nAfter applying generic_preset:", json.dumps(config_after_generic_preset.to_dict(), indent=2))
        # Expected: seed updated, experiment.tracking_uri added.

    except Exception as e:
        print(f"Error during preset.py self-test: {e}")
    finally:
        # Clean up dummy files
        # os.remove(dummy_train_preset_path)
        # os.remove(dummy_generic_preset_path)
        # if not os.listdir(os.path.join(PRESET_DIR, "train")): os.rmdir(os.path.join(PRESET_DIR, "train"))
        # if not os.listdir(os.path.join(PRESET_DIR, "predict")): os.rmdir(os.path.join(PRESET_DIR, "predict"))
        pass

