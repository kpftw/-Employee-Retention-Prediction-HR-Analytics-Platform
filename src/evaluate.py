"""
Evaluation Module for Employee Retention Prediction & HR Analytics Platform.

Provides functions to compute classification metrics, confusion matrices, ROC curves,
Precision-Recall curves, and model comparison tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth binary labels (0/1).
        y_pred (np.ndarray): Predicted binary labels (0/1).
        y_prob (np.ndarray, optional): Predicted positive class probabilities.

    Returns:
        Dict[str, float]: Dictionary of Accuracy, Precision, Recall, F1 Score, and ROC AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5

    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1 Score": round(float(f1), 4),
        "ROC AUC": round(float(roc_auc), 4)
    }

def get_confusion_matrix_details(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Extracts True Negatives, False Positives, False Negatives, and True Positives.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        Dict[str, int]: Dictionary of TN, FP, FN, TP.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def generate_classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Generates a structured DataFrame from sklearn's classification report.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        pd.DataFrame: Classification report as tabular DataFrame.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=["Stay (No)", "Leave (Yes)"])
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    return df_report
