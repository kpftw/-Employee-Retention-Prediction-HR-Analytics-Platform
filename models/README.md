# Serialized Model Artifacts

This directory stores trained machine learning model binaries and preprocessors saved using `joblib`.

Artifacts generated after running `python run_pipeline.py`:
- `best_model.pkl`: Highest-scoring tuned Classifier estimator (Random Forest / XGBoost).
- `preprocessor.pkl`: Fitted `HRDataPreprocessor` instance.
- `scaler.pkl`: Fitted `StandardScaler` / `MinMaxScaler` object.
- `encoders.pkl`: Categorical encoders (`OneHotEncoder`, `LabelEncoder`).
- `feature_names.pkl`: List of encoded feature column names.
- `full_pipeline.pkl`: Serialized end-to-end pipeline.
- `model_comparison_results.pkl`: Performance comparison DataFrame across all 12 algorithms.
