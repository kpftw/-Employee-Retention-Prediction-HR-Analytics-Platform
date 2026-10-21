"""
Unit tests for predict module and pipeline artifacts.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import run_full_training_pipeline
from src.predict import EmployeeRetentionPredictor

def test_prediction_pipeline():
    # Execute quick training to ensure model artifacts exist
    run_full_training_pipeline()
    predictor = EmployeeRetentionPredictor()

    sample = {
        "Age": 30, "Gender": "Male", "Department": "Sales", "JobRole": "Sales Executive",
        "Education": 3, "EducationField": "Life Sciences", "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800, "HourlyRate": 60, "MonthlyRate": 15000, "MonthlyIncome": 4000,
        "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2, "DistanceFromHome": 10, "JobSatisfaction": 3,
        "EnvironmentSatisfaction": 3, "PerformanceRating": 3, "WorkLifeBalance": 3,
        "OverTime": "No", "MaritalStatus": "Married", "NumCompaniesWorked": 2,
        "PercentSalaryHike": 14, "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TotalWorkingYears": 8, "TrainingTimesLastYear": 3, "JobInvolvement": 3, "JobLevel": 2
    }

    result = predictor.predict_single(sample)
    assert result["prediction"] in ["Likely to Leave", "Likely to Stay"]
    assert 0.0 <= result["retention_probability"] <= 1.0
    assert len(result["top_reasons"]) > 0
