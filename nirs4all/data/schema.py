import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError


range_selector_schema = {
    "anyOf": [
        {
            "type": "string",
            "pattern": "^(\s*-?\d+\s*)?(:\s*-?\d+\s*)?(:\s*-?\d+\s*)?$",
            "$provider": "faker.numpy_slice"
        },
        {
            "type": "object",
            "properties": {
                "from": {"type": "integer"},
                "to": {"type": "integer"}
            },
            "oneOf": [
                {"required": ["from"]},
                {"required": ["to"]}
            ]
        },
        {"type": "array", "items": {"type": "integer"}, "minItems": 1}
    ]
}


data_params_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "header": {
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "string", "enum": ["infer"]},
                        {"type": "null"}
                    ]
                },
                "delimiter": {"type": "string", "enum": [",", ";", "\t"]},
                "decimal": {"type": "string", "enum": [".", ","]},
                "na_policy": {"type": "string", "enum": ["abort", "remove", "ignore", "replace", "auto"]}
            },
            "additionalProperties": False
        },
        {
            "type": "array",
            "items": [
                {
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "string", "enum": ["infer"]},
                        {"type": "null"}
                    ]
                },
                {"type": "string", "enum": [",", ";", "\t"]},
                {"type": "string", "enum": [".", ","]},
                {"type": "string", "enum": ["abort", "remove", "ignore", "replace", "auto"]}
            ],
            "minItems": 0,
            "maxItems": 4
        }
    ]
}

filepath_schema = {
    "type": "string",
    "pattern": "^(/[a-zA-Z0-9_.-]+)+\\.(csv|csv\\.gz|npy)$",
    "$provider": "faker.nirs_file_path",
}

folderpath_schema = {
    "type": "string",
    "pattern": "^(/[a-zA-Z0-9_.-]+)+/$",
    "$provider": "faker.nirs_folder_path",
}
csvfile_schema = {
    "anyOf": [
        {"$ref": "#/definitions/filepath"},
        {
            "type": "object",
            "properties": {
                "path": {"$ref": "#/definitions/filepath"},
                "filter": {"$ref": "#/definitions/selector"},
                "params": {"$ref": "#/definitions/data_params"}
            },
            "required": ["path"]
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/filepath"},
                {
                    "type": "object",
                    "properties": {
                        "filter": {"$ref": "#/definitions/selector"},
                    },
                    "required": ["filter"]
                },
            ],
            "minItems": 2,
            "maxItems": 2
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/filepath"},
                {
                    "type": "object",
                    "properties": {
                        "filter": {"$ref": "#/definitions/selector"},
                    },
                    "required": ["filter"]
                },
                {"$ref": "#/definitions/data_params"}
            ],
            "minItems": 3,
            "maxItems": 3
        }
    ]
}

XYfile_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "X": {"$ref": "#/definitions/csvfile"},
                "Y": {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                },
                "params": {"$ref": "#/definitions/data_params"}
            },
            "required": ["X", "Y"]
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                }
            ],
            "minItems": 2,
            "maxItems": 2
        }
    ]
}


datafiles_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "train": {"$ref": "#/definitions/XYfile_schema"},
                "test": {"$ref": "#/definitions/XYfile_schema"},
                "valid": {"$ref": "#/definitions/XYfile_schema"},
                "params": {"$ref": "#/definitions/data_params"}
            },
            "required": ["train"]
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/XYfile_schema"},
                {"$ref": "#/definitions/XYfile_schema"},
                {"$ref": "#/definitions/XYfile_schema"},
            ],
            "minItems": 1,
            "maxItems": 3
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                }
            ],
            "minItems": 2,
            "maxItems": 2
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                },
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                }
            ],
            "minItems": 4,
            "maxItems": 4
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                },
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                },
                {"$ref": "#/definitions/csvfile"},
                {
                    "anyOf": [
                        {"$ref": "#/definitions/csvfile"},
                        {"$ref": "#/definitions/selector"}
                    ]
                }
            ],
            "minItems": 6,
            "maxItems": 6
        }
    ]
}

datafolder_schema = {
    "anyOf": [
        {"$ref": "#/definitions/folderpath"},
        {
            "type": "object",
            "properties": {
                "path": {"$ref": "#/definitions/folderpath"},
                "params": {"$ref": "#/definitions/data_params"}
            },
            "required": ["path"]
        },
        {
            "type": "array",
            "items": [
                {"$ref": "#/definitions/folderpath"},
                {"$ref": "#/definitions/data_params"}
            ],
            "minItems": 2,
            "maxItems": 2
        }
    ]
}

data_schema = {
    "type": "array",
    "items": {
        "anyOf": [
            datafolder_schema,
            datafiles_schema
        ]
    }
}

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "data": data_schema,
    },
    "required": ["data"],
    "definitions": {
        "folderpath": folderpath_schema,
        "filepath": filepath_schema,
        "csvfile": csvfile_schema,
        "XYfile_schema": XYfile_schema,
        "selector": range_selector_schema,
        "data_params": data_params_schema
    }
}


def validate_data_config(config_str) -> bool:
    """
    Validates a data configuration string against the nirs4all data schema.

    Parameters
    ----------
    config_str : str
        The JSON configuration string.

    Returns
    -------
    bool
        True if the configuration is valid, False otherwise.

    Raises
    ------
    Prints an error message if the configuration is invalid or if an error occurs during parsing.
    """
    try:
        validate(instance=json.loads(config_str), schema=schema)
        return True
    except ValidationError as e:
        print(f"The configuration is an invalid nirs4all data configuration: {e.message}")

    return False
