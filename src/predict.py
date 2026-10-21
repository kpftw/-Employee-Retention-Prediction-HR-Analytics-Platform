"""
Prediction Module for Employee Retention Prediction & HR Analytics Platform.

Exposes end-to-end single and batch prediction interfaces with risk probabilities, 
confidence scores, and SHAP explainability insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.utils import load_artifact
from src.preprocessing import clean_data, HRDataPreprocessor
from src.feature_engineering import add_engineered_features
from src.explain import create_shap_explainer, compute_shap_values_for_sample, generate_natural_language_reasons

class EmployeeRetentionPredictor:
    """
    Production-ready Employee Retention Predictor.
    Loads best trained model and preprocessor artifacts to evaluate single employee 
    profiles or batch CSV uploads.
    """
    def __init__(self):
        self.model = load_artifact("best_model.pkl")
        self.preprocessor: HRDataPreprocessor = load_artifact("preprocessor.pkl")
        self.feature_names = load_artifact("feature_names.pkl")
        self._explainer = None

    @property
    def explainer(self):
        """Lazy loads SHAP explainer."""
        if self._explainer is None:
            # Generate background sample matrix for SHAP explainer
            dummy_bg = np.zeros((10, len(self.feature_names)))
            self._explainer = create_shap_explainer(self.model, dummy_bg)
        return self._explainer

    def predict_single(self, employee_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts a dictionary of single employee attributes and returns full prediction payload.

        Args:
            employee_dict (Dict[str, Any]): Employee attributes.

        Returns:
            Dict[str, Any]: Prediction payload with Probability, Decision, Confidence Score, 
                            and Plain-English SHAP reasons.
        """
        df_input = pd.DataFrame([employee_dict])
        
        # 1. Clean & Feature Engineer
        df_clean = clean_data(df_input)
        df_fe = add_engineered_features(df_clean)

        # 2. Transform Features
        X_proc = self.preprocessor.transform(df_fe)

        # 3. Model Inference
        prob_leave = float(self.model.predict_proba(X_proc)[0, 1])
        pred_label = "Likely to Leave" if prob_leave >= 0.5 else "Likely to Stay"
        
        # Confidence Score: Distance from 0.5 decision boundary
        confidence_score = round(abs(prob_leave - 0.5) * 2 * 100, 1)

        # 4. SHAP Local Explanation
        df_local_shap = compute_shap_values_for_sample(self.explainer, X_proc, self.feature_names)
        reasons = generate_natural_language_reasons(df_local_shap, top_n=4)

        return {
            "prediction": pred_label,
            "retention_probability": round(prob_leave, 4),
            "attrition_probability": round(1.0 - prob_leave, 4),
            "confidence_score_pct": confidence_score,
            "risk_category": "High Risk" if prob_leave >= 0.65 else ("Medium Risk" if prob_leave >= 0.35 else "Low Risk"),
            "top_reasons": reasons,
            "shap_breakdown": df_local_shap.head(10).to_dict(orient="records")
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Processes batch DataFrame and appends prediction outcomes and risk scores.

        Args:
            df_raw (pd.DataFrame): Raw batch DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame enriched with prediction columns.
        """
        df_out = df_raw.copy()
        df_clean = clean_data(df_raw)
        df_fe = add_engineered_features(df_clean)

        X_proc = self.preprocessor.transform(df_fe)
        probs = self.model.predict_proba(X_proc)[:, 1]

        df_out["Retention_Probability"] = np.round(probs, 4)
        df_out["Predicted_Retention"] = np.where(probs >= 0.5, "Yes", "No")
        df_out["Risk_Category"] = np.where(
            probs >= 0.65, "High Risk",
            np.where(probs >= 0.35, "Medium Risk", "Low Risk")
        )
        return df_out

if __name__ == "__main__":
    from src.train import run_full_training_pipeline
    run_full_training_pipeline()

    predictor = EmployeeRetentionPredictor()
    sample_employee = {
        "Age": 28,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 500,
        "Department": "Sales",
        "DistanceFromHome": 25,
        "Education": 2,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 45,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2500,
        "MonthlyRate": 12000,
        "NumCompaniesWorked": 5,
        "OverTime": "Yes",
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 4,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 1
    }
    result = predictor.predict_single(sample_employee)
    print(f"[PREDICTION TEST] Result: {result['prediction']} (Prob: {result['retention_probability']})")
