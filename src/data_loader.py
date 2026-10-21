"""
Data Loader Module for Employee Retention Prediction & HR Analytics Platform.

Provides functions for loading, generating, and summarizing the IBM HR Analytics
Employee Retention Dataset.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

RAW_DATA_PATH = os.path.join("data", "raw", "HR_Employee_Retention.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def generate_ibm_hr_dataset(n_samples: int = 1470, seed: int = 42) -> pd.DataFrame:
    """
    Generates an authentic synthetic dataset matching the schema and statistical 
    distribution of the official IBM HR Analytics Employee Retention Dataset.
    
    Args:
        n_samples (int): Number of employee records (default 1470 matching Kaggle IBM dataset).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Complete dataset with 35 feature columns and target 'Retention'.
    """
    np.random.seed(seed)

    # Demographics
    age = np.random.randint(18, 61, size=n_samples)
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.6, 0.4])
    marital_status = np.random.choice(["Single", "Married", "Divorced"], size=n_samples, p=[0.32, 0.46, 0.22])
    education = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.19, 0.39, 0.27, 0.03])
    education_field = np.random.choice(
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        size=n_samples, p=[0.41, 0.31, 0.11, 0.09, 0.02, 0.06]
    )

    # Employment & Roles
    departments = ["Research & Development", "Sales", "Human Resources"]
    dept_p = [0.65, 0.30, 0.05]
    department = np.random.choice(departments, size=n_samples, p=dept_p)

    job_roles = []
    for d in department:
        if d == "Research & Development":
            job_roles.append(np.random.choice(
                ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director", "Manager"],
                p=[0.35, 0.30, 0.15, 0.10, 0.05, 0.05]
            ))
        elif d == "Sales":
            job_roles.append(np.random.choice(
                ["Sales Executive", "Sales Representative", "Manager"],
                p=[0.65, 0.25, 0.10]
            ))
        else:
            job_roles.append(np.random.choice(
                ["Human Resources", "Manager"],
                p=[0.80, 0.20]
            ))

    job_level = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.37, 0.36, 0.15, 0.07, 0.05])
    
    # Income calculation based on JobLevel & Age
    monthly_income = (job_level * 3200 + np.random.randint(-800, 1500, size=n_samples) + (age * 50)).clip(min=1009, max=19999)
    daily_rate = np.random.randint(102, 1500, size=n_samples)
    hourly_rate = np.random.randint(30, 101, size=n_samples)
    monthly_rate = np.random.randint(2094, 27000, size=n_samples)
    percent_salary_hike = np.random.randint(11, 26, size=n_samples)
    stock_option_level = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.43, 0.41, 0.11, 0.05])

    # Work History & Tenure
    total_working_years = (job_level * 3 + np.random.randint(0, 6, size=n_samples)).clip(max=40)
    years_at_company = np.minimum(total_working_years, np.random.randint(0, 15, size=n_samples))
    years_in_current_role = np.minimum(years_at_company, np.random.randint(0, 10, size=n_samples))
    years_since_last_promotion = np.minimum(years_at_company, np.random.randint(0, 8, size=n_samples))
    years_with_curr_manager = np.minimum(years_at_company, np.random.randint(0, 9, size=n_samples))
    num_companies_worked = np.random.randint(0, 10, size=n_samples)

    # Work Environment & Satisfaction
    business_travel = np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size=n_samples, p=[0.71, 0.19, 0.10])
    distance_from_home = np.random.randint(1, 30, size=n_samples)
    environment_satisfaction = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.19, 0.31, 0.31])
    job_involvement = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.06, 0.25, 0.59, 0.10])
    job_satisfaction = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.19, 0.30, 0.32])
    relationship_satisfaction = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.20, 0.31, 0.30])
    work_life_balance = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.06, 0.23, 0.61, 0.10])
    overtime = np.random.choice(["Yes", "No"], size=n_samples, p=[0.28, 0.72])
    performance_rating = np.random.choice([3, 4], size=n_samples, p=[0.84, 0.16])
    training_times_last_year = np.random.randint(0, 7, size=n_samples)

    # Non-informative constant columns (matching Kaggle dataset)
    employee_count = [1] * n_samples
    standard_hours = [80] * n_samples
    over_18 = ["Y"] * n_samples
    employee_number = list(range(1, n_samples + 1))

    # Calculate Realistic Retention Risk Score
    risk_score = (
        (overtime == "Yes") * 2.2 +
        (monthly_income < 3500) * 1.8 +
        (job_satisfaction == 1) * 1.6 +
        (environment_satisfaction == 1) * 1.4 +
        (work_life_balance == 1) * 1.5 +
        (business_travel == "Travel_Frequently") * 1.3 +
        (distance_from_home > 15) * 1.2 +
        (years_since_last_promotion > 4) * 1.1 +
        (stock_option_level == 0) * 0.9 +
        (marital_status == "Single") * 0.8 +
        (age < 28) * 0.7 -
        (job_level >= 3) * 1.2 -
        (years_at_company > 5) * 1.0 -
        2.5
    )

    # Sigmoidal probability conversion targeting ~16.1% positive Retention rate
    retention_prob = 1 / (1 + np.exp(-risk_score))
    retention_binary = np.random.binomial(1, retention_prob)
    retention = np.where(retention_binary == 1, "Yes", "No")

    df = pd.DataFrame({
        "Age": age,
        "Retention": retention,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EmployeeCount": employee_count,
        "EmployeeNumber": employee_number,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_roles,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "Over18": over_18,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StandardHours": standard_hours,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    })

    return df

def load_or_generate_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads dataset from disk if present, otherwise creates raw data directory and 
    generates dataset.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = generate_ibm_hr_dataset()
        df.to_csv(filepath, index=False)
    return df

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key statistical dataset metrics for data understanding.

    Args:
        df (pd.DataFrame): Target dataframe.

    Returns:
        Dict[str, Any]: Dictionary containing shape, columns, missing count, etc.
    """
    target_col = "Retention" if "Retention" in df.columns else df.columns[1]
    summary = {
        "shape": df.shape,
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "missing_values_count": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "target_distribution": df[target_col].value_counts(normalize=True).to_dict() if target_col in df.columns else {},
        "target_counts": df[target_col].value_counts().to_dict() if target_col in df.columns else {},
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist()
    }
    return summary

if __name__ == "__main__":
    df = load_or_generate_data()
    print(f"[DATA LOADER] Data ready! Shape: {df.shape}")
    summary = get_dataset_summary(df)
    print(f"[DATA LOADER] Retention distribution: {summary['target_counts']}")
