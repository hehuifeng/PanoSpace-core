"""
PanoSpace Common Utilities
==========================
Basic utilities for configuration handling, data downloading, and timing operations.
"""

import yaml
import requests
import time
from pathlib import Path
from contextlib import contextmanager


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary representation of the YAML file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def save_config(data: dict, path: str):
    """Save data to a YAML configuration file.

    Parameters
    ----------
    data : dict
        Data to save to YAML format.
    path : str
        Path where the YAML file will be saved.
    """
    with open(path, 'w') as file:
        yaml.safe_dump(data, file)


def download_data(url: str, dest: str):
    """Download data from a URL to a specified destination.

    Parameters
    ----------
    url : str
        URL to download the data from.
    dest : str
        Path to save the downloaded data.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


@contextmanager
def timer(name: str):
    """Context manager for timing code execution.

    Parameters
    ----------
    name : str
        Name or description for the timed block.
    """
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    print(f"[{name}] completed in {elapsed:.2f}s")
