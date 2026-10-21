"""
Preprocessing Module for Employee Retention Prediction & HR Analytics Platform.

Provides functions and classes for data cleaning, outlier detection & treatment,
categorical encoding, feature scaling, and train-test splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Non-informative constant columns to drop
DROP_COLUMNS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame by dropping zero-variance/constant columns and 
    handling any duplicates or missing values.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_clean = df.copy()
    
    # Drop constant columns if present
    cols_to_drop = [c for c in DROP_COLUMNS if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    # Drop duplicate records if any
    df_clean = df_clean.drop_duplicates()

    # Impute missing values if any
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean

def treat_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detects and caps numerical outliers using Interquartile Range (IQR) bounds.

    Args:
        df (pd.DataFrame): Input dataframe.
        factor (float): IQR multiplier (default 1.5).

    Returns:
        pd.DataFrame: Dataframe with capped numerical outliers.
    """
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

    return df_out

class HRDataPreprocessor:
    """
    Custom scikit-learn compatible preprocessor for HR Analytics dataset.
    """
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.label_encoder = LabelEncoder()
        self.column_transformer = None
        self.feature_names = None
        self.cat_cols = None
        self.num_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fits encoders and scalers on features X."""
        X_df = X.copy()
        self.cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

        scaler = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", scaler, self.num_cols),
                ("cat", ohe, self.cat_cols)
            ],
            remainder="passthrough"
        )
        self.column_transformer.fit(X_df)

        # Retrieve output feature names after encoding
        cat_ohe_names = self.column_transformer.named_transformers_["cat"].get_feature_names_out(self.cat_cols)
        self.feature_names = self.num_cols + list(cat_ohe_names)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input features X into scaled and encoded numpy array."""
        X_df = X.copy()
        X_trans = self.column_transformer.transform(X_df)
        return X_trans

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fits on X and returns transformed numpy array."""
        return self.fit(X, y).transform(X)

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        """Encodes target 'Retention' ('Yes'->1, 'No'->0)."""
        return self.label_encoder.fit_transform(y)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transforms target values."""
        return self.label_encoder.transform(y)

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """Converts binary predictions (1/0) back to string ('Yes'/'No')."""
        return self.label_encoder.inverse_transform(y_pred)

def prepare_train_test_data(
    df: pd.DataFrame, 
    target_col: str = "Retention", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits DataFrame into stratified train and test subsets.

    Args:
        df (pd.DataFrame): Source dataframe.
        target_col (str): Target column name.
        test_size (float): Train-test ratio (default 0.2).
        random_state (int): Seed.

    Returns:
        Tuple[X_train, X_test, y_train, y_test]
    """
    if target_col not in df.columns and "Attrition" in df.columns:
        target_col = "Attrition"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from src.data_loader import load_or_generate_data
    df = load_or_generate_data()
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_clean)
    preprocessor = HRDataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    print(f"[PREPROCESSING] Prepared train shape: {X_train_proc.shape}, test size: {len(X_test)}")
