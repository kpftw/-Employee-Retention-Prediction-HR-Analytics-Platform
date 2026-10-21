"""
Feature Selection Module for Employee Retention Prediction & HR Analytics Platform.

Provides techniques for identifying and selecting optimal subset of features using 
Correlation Filtering, Variance Thresholding, SelectKBest, RFECV, Tree Importance, 
Permutation Importance, and PCA.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score

def remove_high_correlation_features(X: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identifies and removes collinear numerical features exceeding Pearson correlation threshold.

    Args:
        X (pd.DataFrame): Numerical features DataFrame.
        threshold (float): Max correlation cutoff (default 0.85).

    Returns:
        Tuple[pd.DataFrame, List[str]]: Filtered DataFrame and list of dropped column names.
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    X_filtered = X.drop(columns=to_drop)
    return X_filtered, to_drop

def apply_variance_threshold(X: np.ndarray, feature_names: List[str], threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
    """
    Removes low-variance features below variance threshold.

    Args:
        X (np.ndarray): Feature matrix.
        feature_names (List[str]): List of feature names.
        threshold (float): Minimum variance cutoff.

    Returns:
        Tuple[np.ndarray, List[str]]: Transformed matrix and remaining feature names.
    """
    vt = VarianceThreshold(threshold=threshold)
    X_high_var = vt.fit_transform(X)
    support = vt.get_support()
    selected_names = [name for name, supp in zip(feature_names, support) if supp]
    return X_high_var, selected_names

def apply_select_k_best(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 20) -> Tuple[np.ndarray, List[str]]:
    """
    Selects top K highest ANOVA F-score features.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_names (List[str]): Feature column names.
        k (int): Number of top features to select.

    Returns:
        Tuple[np.ndarray, List[str]]: Selected matrix and feature names.
    """
    k_val = min(k, X.shape[1])
    skb = SelectKBest(score_func=f_classif, k=k_val)
    X_selected = skb.fit_transform(X, y)
    support = skb.get_support()
    selected_names = [name for name, supp in zip(feature_names, support) if supp]
    return X_selected, selected_names

def get_tree_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Computes tree-based Gini feature importance using Random Forest.

    Args:
        X (np.ndarray): Transformed features matrix.
        y (np.ndarray): Target binary vector.
        feature_names (List[str]): Transformed feature names.

    Returns:
        pd.DataFrame: Sorted DataFrame of feature importances.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp

def apply_rfecv_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        feature_names (List[str]): Feature names.

    Returns:
        Tuple[np.ndarray, List[str]]: Optimal feature matrix and selected feature names.
    """
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfecv = RFECV(estimator=rf, step=1, cv=3, scoring="f1", n_jobs=-1)
    X_rfecv = rfecv.fit_transform(X, y)
    selected_names = [name for name, supp in zip(feature_names, rfecv.support_) if supp]
    return X_rfecv, selected_names

def apply_pca_reduction(X: np.ndarray, n_components: float = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Applies Principal Component Analysis (PCA) to retain desired variance ratio.

    Args:
        X (np.ndarray): Feature matrix.
        n_components (float): Explained variance threshold (e.g., 0.95).

    Returns:
        Tuple[np.ndarray, PCA]: Reduced feature matrix and fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

if __name__ == "__main__":
    from src.data_loader import load_or_generate_data
    from src.preprocessing import clean_data, prepare_train_test_data, HRDataPreprocessor
    from src.feature_engineering import add_engineered_features

    df = load_or_generate_data()
    df_clean = clean_data(df)
    df_fe = add_engineered_features(df_clean)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_fe)

    preprocessor = HRDataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    y_train_enc = preprocessor.fit_transform_target(y_train)

    df_imp = get_tree_feature_importance(X_train_proc, y_train_enc, preprocessor.feature_names)
    print(f"[FEATURE SELECTION] Computed importances for {len(df_imp)} features. Top feature: {df_imp.iloc[0]['Feature']}")
