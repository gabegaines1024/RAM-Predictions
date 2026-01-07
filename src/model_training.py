"""
Model Training and Comparison

This module compares multiple ML models to find the best performer.
Compare different algorithms and select the best one based on evaluation metrics.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict, Tuple
import pandas as pd
from pathlib import Path
from preprocessing import (
    load_data,
    parse_speed_column,
    parse_modules_column,
    prepare_target,
    split_data,
    get_column_groups,
    build_preprocessor
)


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, any] = None
) -> Dict[str, Dict[str, float]]:
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model_name: model_instance (optional)
    
    Returns:
        Dictionary with model names as keys and evaluation metrics as values
    """

    if models is None:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    return results


def compare_models(results: Dict[str, Dict[str, float]]) -> str:
    """
    Compare model results and return the best model name.
    
    Args:
        results: Dictionary from train_multiple_models()
    
    Returns:
        Name of the best performing model
    """

    results_df = pd.DataFrame(results).T.sort_values(by='rmse')

    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(results_df.sort_values(by='rmse')) #lowest RMSE on top

    best_model_name = results_df['rmse'].idxmin()
    print(f"\nBest Model: {best_model_name}")
    print("="*50 + "\n")
    return best_model_name

def get_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[str, any, Dict[str, float]]:
    """
    Convenience function to train models, compare them, and return the best.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        Tuple of (model_name, fitted_model, metrics_dict)
    """
    
    train_models = train_multiple_models(X_train, y_train, X_test, y_test)
    best_model_name = compare_models(train_models)
    return (best_model_name, train_models[best_model_name], train_models)


def load_and_preprocess_data(data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess data, returning train/test splits.
    
    This function:
    1. Loads raw CSV data
    2. Parses compound columns (speed, modules)
    3. Separates features from target
    4. Splits into train/test sets
    5. Applies preprocessing (imputation, scaling, encoding)
    6. Returns preprocessed X_train, X_test, y_train, y_test
    
    Args:
        data_path: Path to CSV file (defaults to data/memory.csv relative to project root)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) - all preprocessed and ready for training
    """
    # Step 1: Load raw data
    if data_path is None:
        # Get path relative to this file's location
        data_path = Path(__file__).parent.parent / "data" / "memory.csv"
    
    df = load_data(data_path)
    
    # Step 2: Parse compound columns (speed -> ddr_gen, speed_mhz; modules -> module_count, module_size, total_capacity)
    df = parse_speed_column(df)
    df = parse_modules_column(df)
    
    # Step 3: Separate features (X) and target (y), drop rows with missing prices
    X, y = prepare_target(df)
    
    # Step 4: Split into train/test sets (BEFORE preprocessing to avoid data leakage)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Build preprocessor (handles imputation, scaling, encoding)
    column_groups = get_column_groups()
    preprocessor = build_preprocessor(
        column_groups['numerical'],
        column_groups['categorical']
    )
    
    # Step 6: Fit preprocessor on training data and transform both train and test
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)
    
    print("Transforming training and test data...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Convert to DataFrame for easier handling
    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    return X_train_df, X_test_df, y_train, y_test


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train and compare models
    print("\nTraining and comparing models...")
    best_model_name, best_model_metrics, all_results = get_best_model(X_train, y_train, X_test, y_test)
    
    print(f"\nBest model: {best_model_name}")
    print(f"Metrics: {best_model_metrics}")  