from pathlib import Path
from typing import Union

import yaml


def read_config(config_file: Union[str, Path]) -> dict:
    """
    Reads a config file and returns a dictionary
    :param config_file: Path to the config file (string or Path)
    :return: Dictionary of config values
    """
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    return config
