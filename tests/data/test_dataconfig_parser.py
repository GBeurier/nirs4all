# tests/test_data_config_parser.py

import pytest
from nirs4all.data.data_config_parser import parse_config


def test_parse_config_with_folder():
    folder_path = "/path/to/data"
    config = parse_config(folder_path)
    assert isinstance(config, dict)


def test_parse_config_invalid_input():
    config = parse_config(12345)
    assert config is None
