"""
Script to execute the full model training pipeline and serialize trained model artifacts.
"""

import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import run_full_training_pipeline

if __name__ == "__main__":
    print("[PIPELINE RUNNER] Starting full machine learning pipeline execution...")
    df_results, best_model, preprocessor = run_full_training_pipeline()
    print("[PIPELINE RUNNER] Pipeline execution complete! All artifacts generated in models/")
