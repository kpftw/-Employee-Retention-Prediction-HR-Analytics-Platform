"""
Utility Module for Employee Retention Prediction & HR Analytics Platform.

Provides helper functions for directory creation, model persistence, and logging.
"""

import os
import joblib
from typing import Any

MODELS_DIR = "models"
REPORTS_DIR = "reports"

def ensure_directories_exist():
    """Creates required output directories if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "eda_plots"), exist_ok=True)

def save_artifact(obj: Any, filename: str):
    """
    Saves object to models/ directory using joblib.

    Args:
        obj (Any): Target object (model, scaler, dict).
        filename (str): Name of artifact file.
    """
    ensure_directories_exist()
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, filepath)
    print(f"[UTILS] Saved artifact -> {filepath}")

def load_artifact(filename: str) -> Any:
    """
    Loads object from models/ directory using joblib.

    Args:
        filename (str): Name of artifact file.

    Returns:
        Any: Loaded object.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)
