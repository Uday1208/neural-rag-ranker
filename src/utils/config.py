# src/utils/config.py
"""
Configuration loading utilities for the Neural RAG Ranker project.

This module centralizes logic for reading YAML configuration files and
merging base settings with scenario-specific overrides so that all
training, evaluation, and serving scripts share a consistent view of
project configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file from disk into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary (empty dict if file is empty).
    """
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override keys into a base dictionary.

    Args:
        base: Original configuration dictionary to be updated.
        overrides: Dictionary containing override values.

    Returns:
        A new dictionary representing base updated with overrides.
    """
    result: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    specific_config_path: str | Path,
    base_config_path: str | Path = "config/base.yaml",
) -> Dict[str, Any]:
    """
    Load the base configuration and merge it with a specific configuration.

    Args:
        specific_config_path:
            Path to the YAML file containing scenario-specific settings.
        base_config_path:
            Path to the base YAML config to be loaded first.

    Returns:
        A configuration dictionary resulting from base + overrides.

    Raises:
        FileNotFoundError: If either configuration file cannot be found.
    """
    base_path = Path(base_config_path)
    specific_path = Path(specific_config_path)

    base_cfg = load_yaml_file(base_path)
    specific_cfg = load_yaml_file(specific_path)

    merged_cfg = _deep_update(base_cfg, specific_cfg)
    return merged_cfg
