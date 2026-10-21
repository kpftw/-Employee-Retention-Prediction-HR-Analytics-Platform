"""
Unit tests for preprocessing and feature engineering modules.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_ibm_hr_dataset
from src.preprocessing import clean_data
from src.feature_engineering import add_engineered_features

def test_clean_data():
    df = generate_ibm_hr_dataset(n_samples=50)
    df_clean = clean_data(df)
    assert "EmployeeCount" not in df_clean.columns
    assert "StandardHours" not in df_clean.columns

def test_feature_engineering():
    df = generate_ibm_hr_dataset(n_samples=50)
    df_fe = add_engineered_features(clean_data(df))
    assert "Workload_Score" in df_fe.columns
    assert "Job_Stability_Score" in df_fe.columns
    assert "Age_Group" in df_fe.columns
