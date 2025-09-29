from pathlib import Path

def _s_(path):
    """Convert path(s) to POSIX format. Handles both single paths and lists of paths."""
    if path is None:
        return None
    if isinstance(path, list):
        return [Path(p).as_posix() for p in path]
    return Path(path).as_posix()

def browse_folder(folder_path, global_params=None):
    config = {
        "train_x": None, "train_x_filter": None, "train_x_params": None,
        "train_y": None, "train_y_filter": None, "train_y_params": None,
        "train_group": None, "train_group_filter": None, "train_group_params": None,
        "train_params": None,
        "test_x": None, "test_x_filter": None, "test_x_params": None,
        "test_y": None, "test_y_filter": None, "test_y_params": None,
        "test_group": None, "test_group_filter": None, "test_group_params": None,
        "test_params": None,
        "global_params": global_params
    }

    files_re = {
        "train_x": ["Xcal", "X_cal", "Cal_X", "calX", "train_X", "trainX", "X_train", "Xtrain"],
        "test_x": ["Xval", "X_val", "val_X", "valX", "Xtest", "X_test", "test_X", "testX"],
        "train_y": ["Ycal", "Y_cal", "Cal_Y", "calY", "train_Y", "trainY", "Y_train", "Ytrain"],
        "test_y": ["Ytest", "Y_test", "test_Y", "testY", "Yval", "Y_val", "val_Y", "valY"],
        "train_group": ["Gcal", "G_cal", "Cal_G", "calG", "train_G", "trainG", "G_train", "Gtrain"],
        "test_group": ["Gtest", "G_test", "test_G", "testG", "Gval", "G_val", "val_G", "valG"],
    }

    dataset_dir = Path(folder_path)
    for key, patterns in files_re.items():
        matched_files = []
        for pattern in patterns:
            pattern_lower = pattern.lower()
            for file in dataset_dir.glob("*"):
                if pattern_lower in file.name.lower():
                    matched_files.append(str(file))

        if len(matched_files) == 0:
            # print(f"⚠️ Dataset does not have data for {key}.")
            # logging.warning("No %s file found for %s.", key, dataset_name)
            continue
        elif len(matched_files) == 1:
            # Single source - store as single path for backward compatibility
            config[key] = _s_(matched_files[0])
        else:
            # Multi-source - store as array of paths
            print(f"📊 Multiple {key} files found for {folder_path}: {len(matched_files)} sources detected.")
            config[key] = _s_(matched_files)

    return config


def folder_to_name(folder_path):
    path = Path(folder_path)
    for part in reversed(path.parts):
        clean_part = ''.join(c if c.isalnum() else '_' for c in part)
        if clean_part:
            return clean_part.lower()
    return "Unknown_dataset"


def parse_config(data_config):
    # a single folder path
    if isinstance(data_config, str):
        return browse_folder(data_config), folder_to_name(data_config)

    elif isinstance(data_config, dict):
        # a folder tag, idem as single path but with params
        if "folder" in data_config:
            return browse_folder(data_config["folder"], data_config.get("params")), folder_to_name(data_config["folder"])
        else:
            # a full config dict
            required_keys_pattern = ['train_x', 'test_x']
            if all(key in data_config for key in required_keys_pattern):
                train_file = data_config.get("train_x")
                if isinstance(train_file, list):
                    train_file = train_file[0]
                train_file = Path(str(train_file))
                dataset_name = f"{train_file.parent.name}_{train_file.stem}"
                return data_config, dataset_name

    print(f"❌ Error in config: unsupported dataset config >> {type(data_config)}: {data_config}")
    return None, 'Unknown_dataset'


