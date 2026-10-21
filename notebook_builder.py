"""
Utility script to generate all 10 production Jupyter Notebooks (.ipynb)
for Employee Retention Prediction & HR Analytics Platform.
"""

import os
import json

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename: str, cells: list):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[NOTEBOOK BUILDER] Created {filepath}")

def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# 1. 01_Data_Collection.ipynb
create_notebook("01_Data_Collection.ipynb", [
    markdown_cell("# STEP 1: Data Collection\n\nThis notebook demonstrates loading the IBM HR Analytics Employee Retention Dataset and displaying initial structure metrics."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\n\ndf = load_or_generate_data()\nprint(f'Shape: {df.shape}')\nprint(f'Columns: {list(df.columns)}')"),
    code_cell("df.info()"),
    code_cell("df.head(5)")
])

# 2. 02_Data_Understanding.ipynb
create_notebook("02_Data_Understanding.ipynb", [
    markdown_cell("# STEP 2: Data Understanding\n\nDeep dive statistics, missing values, duplicates, and target class distribution."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data, get_dataset_summary\n\ndf = load_or_generate_data()\nsummary = get_dataset_summary(df)\nfor k, v in summary.items():\n    print(f'{k}: {v}')"),
    code_cell("df.describe().T"),
    code_cell("df['Retention'].value_counts(normalize=True)")
])

# 3. 03_EDA.ipynb
create_notebook("03_EDA.ipynb", [
    markdown_cell("# STEP 3: Exploratory Data Analysis (EDA)\n\nVisualizing features against employee retention rates."),
    code_cell("import sys\nsys.path.append('..')\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.data_loader import load_or_generate_data\n\ndf = load_or_generate_data()\nplt.style.use('seaborn-v0_8-darkgrid')"),
    code_cell("# Retention by Department\nplt.figure(figsize=(8, 4))\nsns.countplot(data=df, x='Department', hue='Retention', palette='viridis')\nplt.title('Department-wise Retention Distribution')\nplt.show()"),
    code_cell("# Monthly Income Boxplot by Retention\nplt.figure(figsize=(7, 4))\nsns.boxplot(data=df, x='Retention', y='MonthlyIncome', palette='Set2')\nplt.title('Monthly Income vs Retention')\nplt.show()"),
    code_cell("# OverTime vs Retention\nplt.figure(figsize=(6, 4))\nsns.countplot(data=df, x='OverTime', hue='Retention', palette='magma')\nplt.title('OverTime Work vs Retention')\nplt.show()")
])

# 4. 04_Data_Preprocessing.ipynb
create_notebook("04_Data_Preprocessing.ipynb", [
    markdown_cell("# STEP 4 & 5: Data Cleaning & Preprocessing\n\nHandling outliers, encoding, scaling, and train-test splits."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\nfrom src.preprocessing import clean_data, treat_outliers_iqr, prepare_train_test_data, HRDataPreprocessor\n\ndf = load_or_generate_data()\ndf_clean = clean_data(df)\ndf_iqr = treat_outliers_iqr(df_clean)\nX_train, X_test, y_train, y_test = prepare_train_test_data(df_iqr, target_col='Retention')\n\nprep = HRDataPreprocessor()\nX_train_p = prep.fit_transform(X_train)\ny_train_e = prep.fit_transform_target(y_train)\nprint(f'Processed train shape: {X_train_p.shape}')")
])

# 5. 05_Feature_Engineering.ipynb
create_notebook("05_Feature_Engineering.ipynb", [
    markdown_cell("# STEP 6: Feature Engineering\n\nCreating domain features: Age_Group, Workload_Score, Job_Stability_Score, etc."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\nfrom src.preprocessing import clean_data\nfrom src.feature_engineering import add_engineered_features\n\ndf = load_or_generate_data()\ndf_clean = clean_data(df)\ndf_fe = add_engineered_features(df_clean)\nprint('Engineered Columns:', list(df_fe.columns[-9:]))\ndf_fe[['Workload_Score', 'Job_Stability_Score', 'Income_Per_Experience']].head()")
])

# 6. 06_Feature_Selection.ipynb
create_notebook("06_Feature_Selection.ipynb", [
    markdown_cell("# STEP 7: Feature Selection\n\nEvaluating feature importances using Gini importance and correlation filtering."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\nfrom src.preprocessing import clean_data, prepare_train_test_data, HRDataPreprocessor\nfrom src.feature_engineering import add_engineered_features\nfrom src.feature_selection import get_tree_feature_importance\n\ndf = add_engineered_features(clean_data(load_or_generate_data()))\nX_tr, X_te, y_tr, y_te = prepare_train_test_data(df, target_col='Retention')\nprep = HRDataPreprocessor()\nX_tr_p = prep.fit_transform(X_tr)\ny_tr_e = prep.fit_transform_target(y_tr)\n\ndf_imp = get_tree_feature_importance(X_tr_p, y_tr_e, prep.feature_names)\ndf_imp.head(10)")
])

# 7. 07_Model_Training.ipynb
create_notebook("07_Model_Training.ipynb", [
    markdown_cell("# STEP 8 & 9: Model Training & SMOTE Class Balancing\n\nTraining 12 classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, etc."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.train import run_full_training_pipeline\n\ndf_results, best_model, prep = run_full_training_pipeline()\ndf_results")
])

# 8. 08_Hyperparameter_Tuning.ipynb
create_notebook("08_Hyperparameter_Tuning.ipynb", [
    markdown_cell("# STEP 10 & 11: Cross Validation & Hyperparameter Tuning\n\nGridSearchCV optimization on top performing algorithms."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\nfrom src.preprocessing import clean_data, prepare_train_test_data, HRDataPreprocessor\nfrom src.feature_engineering import add_engineered_features\nfrom src.train import tune_best_model\nfrom imblearn.over_sampling import SMOTE\n\ndf = add_engineered_features(clean_data(load_or_generate_data()))\nX_tr, X_te, y_tr, y_te = prepare_train_test_data(df, target_col='Retention')\nprep = HRDataPreprocessor()\nX_tr_p = prep.fit_transform(X_tr)\ny_tr_e = prep.fit_transform_target(y_tr)\n\nsmote = SMOTE(random_state=42)\nX_res, y_res = smote.fit_resample(X_tr_p, y_tr_e)\nbest_tuned = tune_best_model(X_res, y_res, 'Random Forest')\nprint(best_tuned)")
])

# 9. 09_Model_Evaluation.ipynb
create_notebook("09_Model_Evaluation.ipynb", [
    markdown_cell("# STEP 12: Model Evaluation & Metrics\n\nAccuracy, Precision, Recall, F1 Score, ROC AUC, and Confusion Matrix."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.data_loader import load_or_generate_data\nfrom src.preprocessing import clean_data, prepare_train_test_data, HRDataPreprocessor\nfrom src.feature_engineering import add_engineered_features\nfrom src.utils import load_artifact\nfrom src.evaluate import evaluate_model_performance, generate_classification_report_df\n\nmodel = load_artifact('best_model.pkl')\nprep = load_artifact('preprocessor.pkl')\n\ndf = add_engineered_features(clean_data(load_or_generate_data()))\nX_tr, X_te, y_tr, y_te = prepare_train_test_data(df, target_col='Retention')\nX_te_p = prep.transform(X_te)\ny_te_e = prep.transform_target(y_te)\n\ny_pred = model.predict(X_te_p)\ny_prob = model.predict_proba(X_te_p)[:, 1]\nmetrics = evaluate_model_performance(y_te_e, y_pred, y_prob)\nprint('Evaluation Metrics:', metrics)\ngenerate_classification_report_df(y_te_e, y_pred)")
])

# 10. 10_Model_Explainability.ipynb
create_notebook("10_Model_Explainability.ipynb", [
    markdown_cell("# STEP 13: Explainable AI (SHAP)\n\nGlobal feature importance and single employee prediction drivers."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.predict import EmployeeRetentionPredictor\n\npredictor = EmployeeRetentionPredictor()\nsample_employee = {\n    'Age': 29, 'BusinessTravel': 'Travel_Frequently', 'DailyRate': 400, 'Department': 'Sales',\n    'DistanceFromHome': 22, 'Education': 3, 'EducationField': 'Life Sciences',\n    'EnvironmentSatisfaction': 1, 'Gender': 'Female', 'HourlyRate': 50, 'JobInvolvement': 2,\n    'JobLevel': 1, 'JobRole': 'Sales Representative', 'JobSatisfaction': 1, 'MaritalStatus': 'Single',\n    'MonthlyIncome': 2800, 'MonthlyRate': 14000, 'NumCompaniesWorked': 6, 'OverTime': 'Yes',\n    'PercentSalaryHike': 11, 'PerformanceRating': 3, 'RelationshipSatisfaction': 2,\n    'StockOptionLevel': 0, 'TotalWorkingYears': 5, 'TrainingTimesLastYear': 2, 'WorkLifeBalance': 1,\n    'YearsAtCompany': 2, 'YearsInCurrentRole': 1, 'YearsSinceLastPromotion': 2, 'YearsWithCurrManager': 1\n}\n\nres = predictor.predict_single(sample_employee)\nprint(f'Prediction: {res[\"prediction\"]} (Prob: {res[\"retention_probability\"]})')\nprint('Top Reasons:')\nfor r in res['top_reasons']:\n    print(' -', r)")
])

print("[NOTEBOOK BUILDER] All 10 notebooks built successfully!")
