"""
Unit tests for data_loader module.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_ibm_hr_dataset, get_dataset_summary

def test_generate_ibm_hr_dataset():
    df = generate_ibm_hr_dataset(n_samples=100)
    assert len(df) == 100
    assert "Retention" in df.columns
    assert set(df["Retention"].unique()).issubset({"Yes", "No"})

def test_get_dataset_summary():
    df = generate_ibm_hr_dataset(n_samples=50)
    summary = get_dataset_summary(df)
    assert summary["num_rows"] == 50
    assert summary["num_cols"] == 35
