"""Configuration loader for Sevak-Hybrid."""

import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config from project root or given path."""
    if not os.path.isabs(path):
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(project_root, path)
        if os.path.exists(alt):
            path = alt

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
