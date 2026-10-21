"""
Explainable AI (XAI) Module for Employee Retention Prediction & HR Analytics Platform.

Provides SHAP (SHapley Additive exPlanations) calculation routines for global 
feature importance and single-instance local predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import shap

def create_shap_explainer(model: Any, X_background: np.ndarray) -> shap.Explainer:
    """
    Creates an appropriate SHAP explainer instance for tree or linear models.

    Args:
        model (Any): Fitted scikit-learn/XGBoost/CatBoost model.
        X_background (np.ndarray): Background dataset sample for Kernel/Linear explainers.

    Returns:
        shap.Explainer: Fitted SHAP explainer.
    """
    try:
        # TreeExplainer for Tree-based models (RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost)
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fallback to Kernel or Linear Explainer
        sample_bg = shap.sample(X_background, min(50, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, sample_bg)
    
    return explainer

def compute_shap_values_for_sample(
    explainer: shap.Explainer, 
    X_sample: np.ndarray, 
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes local SHAP values for a single processed employee sample.

    Args:
        explainer (shap.Explainer): Fitted SHAP explainer.
        X_sample (np.ndarray): Single row processed numpy feature array (1, n_features).
        feature_names (List[str]): List of transformed feature names.

    Returns:
        pd.DataFrame: Table of feature names, feature values, SHAP values, and impact magnitude.
    """
    shap_vals = explainer.shap_values(X_sample)

    # Handle binary classifier output structure variants
    if isinstance(shap_vals, list):
        # List of arrays [class_0_shap, class_1_shap]
        vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        # (1, n_features, 2)
        vals = shap_vals[0, :, 1]
    elif len(shap_vals.shape) == 2:
        vals = shap_vals[0]
    else:
        vals = shap_vals

    df_shap = pd.DataFrame({
        "Feature": feature_names,
        "Value": X_sample[0],
        "SHAP_Value": vals,
        "Abs_SHAP": np.abs(vals)
    }).sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)

    return df_shap

def generate_natural_language_reasons(df_local_shap: pd.DataFrame, top_n: int = 4) -> List[str]:
    """
    Translates top local SHAP values into simple English explanations for HR managers.

    Args:
        df_local_shap (pd.DataFrame): Sorted local SHAP DataFrame.
        top_n (int): Number of top drivers to return.

    Returns:
        List[str]: Plain English explanations of risk factors.
    """
    top_drivers = df_local_shap.head(top_n)
    reasons = []

    for _, row in top_drivers.iterrows():
        feat = row["Feature"]
        val = row["Value"]
        shap_val = row["SHAP_Value"]

        direction = "increases" if shap_val > 0 else "reduces"
        
        # Clean up technical OHE feature names for business presentation
        clean_feat = feat.replace("cat__", "").replace("num__", "").replace("_", " ")

        reasons.append(
            f"**{clean_feat}** {direction} flight risk by {abs(shap_val):.1%} "
            f"(observed value: {val:.2f})"
        )

    return reasons
