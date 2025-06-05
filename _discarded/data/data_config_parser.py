from pathlib import Path

def _s_(path: str) -> str:
    if path is None:
        return None
    return Path(path).as_posix()

def browse_folder(folder_path, global_params=None):
    print(">> Browsing", folder_path)

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

        if len(matched_files) > 1:
            print(f"Multiple {key} files found for {folder_path}.")
            # logging.warning("Multiple %s files found for %s.", key, dataset_name)
            continue
        if len(matched_files) == 0:
            print(f"No {key} file found for {folder_path}.")
            # logging.warning("No %s file found for %s.", key, dataset_name)
            continue

        config[key] = _s_(matched_files[0])

    return config


# def parse_selector(filter):
#     if isinstance(filter, str):
#         return {"np_slice": filter}
#     elif isinstance(filter, dict):
#         return {"from": filter.get("from", "start"), "to": filter.get("to", "end")}
#     elif isinstance(filter, list):
#         return {"cols": filter}


# def parse_XY(XY):
#     # print("Parsing XY", XY)
#     if isinstance(XY, dict):
#         return {"x": XY["X"], "y": XY["Y"], "params": XY.get("params")}
#     elif isinstance(XY, list) or isinstance(XY, tuple):
#         return {"x": XY[0], "y": XY[1]}


# def parse_file(file):
#     # print("Parsing file", file)
#     if isinstance(file, str):
#         return {"path": file}
#     elif isinstance(file, dict):
#         return {"path": file["path"], "filter": parse_selector(file.get("filter")), "params": file.get("params")}
#     elif isinstance(file, list) or isinstance(file, tuple):
#         if len(file) == 2:
#             return {"path": file[0], "filter": parse_selector(file[1]["filter"])}
#         elif len(file) == 3:
#             return {"path": file[0], "filter": parse_selector(file[1]["filter"]), "params": file[2]}


# def parse_file_or_selector(file):
#     if isinstance(file, str):
#         if bool(re.match("^(\s*-?\d+\s*)?(:\s*-?\d+\\s*)?(:\s*-?\d+\s*)?$", file)):
#             return parse_selector(file)
#         else:
#             return parse_file(file)
#     elif isinstance(file, dict):
#         if 'path' in file:
#             return parse_file(file)
#         else:
#             return parse_selector(file)
#     elif isinstance(file, list) or isinstance(file, tuple):
#         if isinstance(file[0], str):
#             return parse_file(file)
#         else:
#             return parse_selector(file)


# def format_config(train_XY=None, test_XY=None, global_params=None, train_x=None, test_x=None, train_y=None, test_y=None, train_params=None, test_params=None):

#     if (train_x is not None and train_XY is not None) or (test_x is not None and test_XY is not None):
#         print("ERROR: both XY and X,Y are provided")

#     if (train_x is not None and train_y is None) or (test_x is not None and test_y is None):
#         print("ERROR: both X and Y must be provided")

#     train_x_path, train_x_filter, train_x_params = None, None, None
#     train_y_path, train_y_filter, train_y_params = None, None, None
#     test_x_path, test_x_filter, test_x_params = None, None, None
#     test_y_path, test_y_filter, test_y_params = None, None, None
#     train_params, test_params = None, None

#     if train_XY is not None:
#         p_XY = parse_XY(train_XY)
#         train_params = p_XY.get("params")
#         file_x = parse_file(p_XY["x"])
#         train_x_path, train_x_filter, train_x_params = file_x["path"], file_x.get("filter"), file_x.get("params")
#         p_y = parse_file_or_selector(p_XY["y"])
#         if 'path' in p_y:
#             train_y_path, train_y_filter, train_y_params = p_y["path"], p_y.get("filter"), p_y.get("params")
#         else:
#             train_y_filter = p_y
#             train_y_path = "From X"

#     if test_XY is not None:
#         p_XY = parse_XY(test_XY)
#         test_params = p_XY.get("params")
#         file_x = parse_file(p_XY["x"])
#         test_x_path, test_x_filter, test_x_params = file_x["path"], file_x.get("filter"), file_x.get("params")
#         p_y = parse_file_or_selector(p_XY["y"])
#         if 'path' in p_y:
#             test_y_path, test_y_filter, test_y_params = p_y["path"], p_y.get("filter"), p_y.get("params")
#         else:
#             test_y_filter = p_y
#             test_y_path = "From X"

#     if train_x is not None:
#         p_x = parse_file(train_x)
#         train_x_path, train_x_filter, train_x_params = p_x["path"], p_x.get("filter"), p_x.get("params")

#         p_y = parse_file_or_selector(train_y)
#         if 'path' in p_y:
#             train_y_path, train_y_filter, train_y_params = p_y["path"], p_y.get("filter"), p_y.get("params")
#         else:
#             train_y_filter = p_y
#             train_y_path = "From X"

#     if test_x is not None:
#         p_x = parse_file(test_x)
#         test_x_path, test_x_filter, test_x_params = p_x["path"], p_x.get("filter"), p_x.get("params")

#         p_y = parse_file_or_selector(test_y)
#         if 'path' in p_y:
#             test_y_path, test_y_filter, test_y_params = p_y["path"], p_y.get("filter"), p_y.get("params")
#         else:
#             test_y_filter = p_y
#             test_y_path = "From X"

#     return {
#         "train_x": _s_(train_x_path), "train_x_filter": train_x_filter, "train_x_params": train_x_params,
#         "train_y": _s_(train_y_path), "train_y_filter": train_y_filter, "train_y_params": train_y_params,
#         "train_params": train_params,
#         "test_x": _s_(test_x_path), "test_x_filter": test_x_filter, "test_x_params": test_x_params,
#         "test_y": _s_(test_y_path), "test_y_filter": test_y_filter, "test_y_params": test_y_params,
#         "test_params": test_params,
#         "global_params": global_params
#     }


def parse_config(data_config):
    if isinstance(data_config, str):
        return browse_folder(data_config)
    elif isinstance(data_config, dict):
        if "path" in data_config:  # If path is present, browse folder
            return browse_folder(data_config["path"], data_config.get("params"))
        else:  # Otherwise, assume it's an already parsed config dictionary
            # TODO: Add more robust validation here if needed in the future
            # For now, if it's a dict without 'path', assume it's usable by get_dataset
            required_keys_pattern = ['train_x', 'test_x']  # Basic check
            if all(key in data_config for key in required_keys_pattern):
                return data_config
            else:
                print(f"CONFIG ERROR _ obj >> Missing one of {required_keys_pattern} in dict: {data_config}")
                return None
    # If data_config is not a string or a recognized dictionary structure, return None
    print(f"CONFIG ERROR _ unsupported type or structure >> {type(data_config)}: {data_config}")
    return None
