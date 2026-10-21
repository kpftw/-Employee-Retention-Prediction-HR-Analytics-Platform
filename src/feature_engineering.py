"""
Feature Engineering Module for Employee Retention Prediction & HR Analytics Platform.

Creates business-domain features derived from raw HR attributes to boost classifier 
predictive power and provide interpretable HR analytics.
"""

import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-driven feature engineering transformations to the input HR DataFrame.

    Engineered Features:
    --------------------
    1. Age_Group: Categorizes employees by life/career stages (Under 30, 30-39, 40-49, 50+).
       * Rationale: Younger employees exhibit significantly higher retention due to career exploration.

    2. Income_Per_Experience: MonthlyIncome / (TotalWorkingYears + 1).
       * Rationale: Measures compensation fairness relative to industry experience. Low pay relative to 
         experience causes dissatisfaction.

    3. Years_Since_Promotion_Ratio: YearsSinceLastPromotion / (YearsAtCompany + 1).
       * Rationale: Indicates career stagnation within the company. Higher ratio signals lack of growth.

    4. Distance_Category: Categorizes commute into Near (<=5km), Moderate (6-15km), Far (>15km).
       * Rationale: Long commutes increase fatigue and work-life balance issues, triggering turnover.

    5. Experience_Level: Experience tier (Entry <=3 yrs, Mid 4-8 yrs, Senior 9-15 yrs, Executive >15 yrs).
       * Rationale: Attrition strategies differ greatly between early career and executive talent.

    6. Promotion_Gap: YearsAtCompany - YearsSinceLastPromotion.
       * Rationale: Quantifies how many tenure years were spent in current/earlier roles before last promotion.

    7. Income_Growth_Indicator: (PercentSalaryHike * MonthlyIncome) / 100.
       * Rationale: Measures actual monetary value increase from recent performance hikes.

    8. Workload_Score: Combined risk index of OverTime (Yes=2), BusinessTravel (Freq=2, Rare=1), and Distance.
       * Rationale: Multi-factor stress metric capturing overtime demands, travel burden, and commute time.

    9. Job_Stability_Score: YearsAtCompany / (NumCompaniesWorked + 1).
       * Rationale: Indicates employee loyalty vs job-hopping tendencies. High stability indicates low flight risk.

    Args:
        df (pd.DataFrame): Raw or cleaned HR dataframe.

    Returns:
        pd.DataFrame: Dataframe augmented with engineered features.
    """
    df_fe = df.copy()

    # 1. Age Group
    df_fe["Age_Group"] = pd.cut(
        df_fe["Age"], 
        bins=[0, 29, 39, 49, 100], 
        labels=["Under_30", "30_39", "40_49", "50_Plus"]
    ).astype(str)

    # 2. Income Per Experience
    df_fe["Income_Per_Experience"] = (df_fe["MonthlyIncome"] / (df_fe["TotalWorkingYears"] + 1)).round(2)

    # 3. Years Since Promotion Ratio
    df_fe["Years_Since_Promotion_Ratio"] = (df_fe["YearsSinceLastPromotion"] / (df_fe["YearsAtCompany"] + 1)).round(3)

    # 4. Distance Category
    df_fe["Distance_Category"] = pd.cut(
        df_fe["DistanceFromHome"], 
        bins=[-1, 5, 15, 100], 
        labels=["Near", "Moderate", "Far"]
    ).astype(str)

    # 5. Experience Level
    df_fe["Experience_Level"] = pd.cut(
        df_fe["TotalWorkingYears"], 
        bins=[-1, 3, 8, 15, 100], 
        labels=["Entry", "Mid", "Senior", "Executive"]
    ).astype(str)

    # 6. Promotion Gap
    df_fe["Promotion_Gap"] = (df_fe["YearsAtCompany"] - df_fe["YearsSinceLastPromotion"]).clip(lower=0)

    # 7. Income Growth Indicator
    df_fe["Income_Growth_Indicator"] = ((df_fe["PercentSalaryHike"] * df_fe["MonthlyIncome"]) / 100.0).round(2)

    # 8. Workload Score
    overtime_val = (df_fe["OverTime"] == "Yes").astype(int) * 2.0
    travel_val = df_fe["BusinessTravel"].map({"Travel_Frequently": 2.0, "Travel_Rarely": 1.0, "Non-Travel": 0.0}).fillna(0)
    distance_val = (df_fe["DistanceFromHome"] > 15).astype(int) * 1.0
    df_fe["Workload_Score"] = (overtime_val + travel_val + distance_val).round(1)

    # 9. Job Stability Score
    df_fe["Job_Stability_Score"] = (df_fe["YearsAtCompany"] / (df_fe["NumCompaniesWorked"] + 1)).round(2)

    return df_fe

if __name__ == "__main__":
    from src.data_loader import load_or_generate_data
    from src.preprocessing import clean_data

    df = load_or_generate_data()
    df_clean = clean_data(df)
    df_fe = add_engineered_features(df_clean)
    print(f"[FEATURE ENGINEERING] Created {df_fe.shape[1] - df_clean.shape[1]} new features. Total columns: {df_fe.shape[1]}")
