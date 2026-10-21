"""
FastAPI Microservice for Employee Retention Prediction & HR Analytics Platform.

Exposes RESTful endpoints for single employee prediction, batch prediction, 
and health check monitoring.
"""

import sys
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import EmployeeRetentionPredictor

app = FastAPI(
    title="Employee Retention Prediction API",
    description="Enterprise Machine Learning REST Microservice for HR Analytics",
    version="1.0.0"
)

# Global predictor instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = EmployeeRetentionPredictor()
    except Exception:
        from src.train import run_full_training_pipeline
        run_full_training_pipeline()
        predictor = EmployeeRetentionPredictor()

class SingleEmployeeRequest(BaseModel):
    Age: int = Field(..., example=30)
    Gender: str = Field(..., example="Male")
    Department: str = Field(..., example="Research & Development")
    JobRole: str = Field(..., example="Research Scientist")
    Education: int = Field(..., example=3)
    EducationField: str = Field(..., example="Life Sciences")
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., example=800)
    HourlyRate: int = Field(..., example=60)
    MonthlyRate: int = Field(..., example=15000)
    MonthlyIncome: int = Field(..., example=4000)
    YearsAtCompany: int = Field(..., example=5)
    YearsInCurrentRole: int = Field(..., example=3)
    YearsSinceLastPromotion: int = Field(..., example=1)
    YearsWithCurrManager: int = Field(..., example=2)
    DistanceFromHome: int = Field(..., example=10)
    JobSatisfaction: int = Field(..., example=3)
    EnvironmentSatisfaction: int = Field(..., example=3)
    PerformanceRating: int = Field(..., example=3)
    WorkLifeBalance: int = Field(..., example=3)
    OverTime: str = Field(..., example="No")
    MaritalStatus: str = Field(..., example="Married")
    NumCompaniesWorked: int = Field(..., example=2)
    PercentSalaryHike: int = Field(..., example=14)
    RelationshipSatisfaction: int = Field(..., example=3)
    StockOptionLevel: int = Field(..., example=1)
    TotalWorkingYears: int = Field(..., example=8)
    TrainingTimesLastYear: int = Field(..., example=3)
    JobInvolvement: int = Field(..., example=3)
    JobLevel: int = Field(..., example=2)

@app.get("/")
def read_root():
    return {
        "service": "Employee Retention Prediction API",
        "status": "Online",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
def predict_single(employee: SingleEmployeeRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor engine not initialized.")
    payload = employee.dict()
    result = predictor.predict_single(payload)
    return result

@app.post("/batch_predict")
def predict_batch(employees: List[SingleEmployeeRequest]):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor engine not initialized.")
    payloads = [e.dict() for e in employees]
    import pandas as pd
    df_in = pd.DataFrame(payloads)
    df_out = predictor.predict_batch(df_in)
    return df_out[["Predicted_Retention", "Retention_Probability", "Risk_Category"]].to_dict(orient="records")
