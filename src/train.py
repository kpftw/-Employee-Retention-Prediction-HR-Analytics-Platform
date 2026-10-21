"""
Training Pipeline for Employee Retention Prediction & HR Analytics Platform.

Trains and evaluates 12 Machine Learning classifiers:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Gradient Boosting
8. AdaBoost
9. Extra Trees
10. XGBoost
11. LightGBM
12. CatBoost

Handles SMOTE oversampling, Cross-Validation, Hyperparameter Tuning, and 
Artifact Serialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from src.data_loader import load_or_generate_data
from src.preprocessing import clean_data, prepare_train_test_data, HRDataPreprocessor
from src.feature_engineering import add_engineered_features
from src.evaluate import evaluate_model_performance
from src.utils import save_artifact

def get_classifiers() -> Dict[str, Any]:
    """Returns a dictionary of 12 un-tuned classifier instances."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

def train_and_evaluate_all_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    use_smote: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains all 12 classifiers, performs K-Fold CV, and returns evaluation metrics summary.

    Args:
        X_train (np.ndarray): Processed train features.
        y_train (np.ndarray): Encoded train target.
        X_test (np.ndarray): Processed test features.
        y_test (np.ndarray): Encoded test target.
        use_smote (bool): Whether to balance training set using SMOTE.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Summary dataframe and dictionary of fitted models.
    """
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    classifiers = get_classifiers()
    results = []
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in classifiers.items():
        # Cross validation scores
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="f1", n_jobs=-1)
        
        # Fit model
        model.fit(X_train_res, y_train_res)
        fitted_models[name] = model

        # Test evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_model_performance(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics["CV Mean F1"] = round(float(np.mean(cv_scores)), 4)
        metrics["CV Std F1"] = round(float(np.std(cv_scores)), 4)

        results.append(metrics)

    df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    return df_results, fitted_models

def tune_best_model(X_train: np.ndarray, y_train: np.ndarray, model_name: str = "Random Forest") -> Any:
    """
    Performs hyperparameter tuning via GridSearchCV on selected top model.

    Args:
        X_train (np.ndarray): Resampled train features.
        y_train (np.ndarray): Resampled train target.
        model_name (str): Model name key.

    Returns:
        Any: Best tuned model estimator.
    """
    if model_name == "Random Forest":
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "criterion": ["gini", "entropy"]
        }
    elif model_name == "XGBoost":
        base_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    else:
        base_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None]
        }

    grid = GridSearchCV(base_model, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"[TUNING] Best Parameters for {model_name}: {grid.best_params_}")
    return grid.best_estimator_

def run_full_training_pipeline() -> Tuple[pd.DataFrame, Any, HRDataPreprocessor]:
    """
    Executes complete end-to-end training pipeline and persists model artifacts.

    Returns:
        Tuple[pd.DataFrame, Any, HRDataPreprocessor]: Results table, Best model, and Preprocessor.
    """
    # 1. Load Data
    df = load_or_generate_data()
    df_clean = clean_data(df)
    df_fe = add_engineered_features(df_clean)

    # 2. Split Data
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_fe, target_col="Retention")

    # 3. Preprocess
    preprocessor = HRDataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    y_train_enc = preprocessor.fit_transform_target(y_train)
    y_test_enc = preprocessor.transform_target(y_test)

    # 4. Train & Compare Models
    df_results, fitted_models = train_and_evaluate_all_models(
        X_train_proc, y_train_enc, X_test_proc, y_test_enc, use_smote=True
    )

    best_model_name = df_results.iloc[0]["Model"]
    print(f"[TRAINING] Best performing model: {best_model_name}")

    # 5. Tune Best Model
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train_enc)
    best_tuned_model = tune_best_model(X_train_res, y_train_res, model_name=best_model_name)

    # 6. Save Artifacts
    save_artifact(best_tuned_model, "best_model.pkl")
    save_artifact(preprocessor, "preprocessor.pkl")
    save_artifact(preprocessor.feature_names, "feature_names.pkl")
    save_artifact(df_results, "model_comparison_results.pkl")

    return df_results, best_tuned_model, preprocessor

if __name__ == "__main__":
    df_results, best_model, preprocessor = run_full_training_pipeline()
    print("[TRAINING] Complete pipeline executed successfully!")
    print(df_results.head(5).to_string())
