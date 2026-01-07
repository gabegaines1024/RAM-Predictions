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
from preprocessing import build_model_pipeline


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
    
    TODO:
    1. Find the model with the lowest RMSE (or highest R²)
    2. Print comparison table showing all models side-by-side
    3. Return the name of the best performing model
    
    Args:
        results: Dictionary from train_multiple_models()
    
    Returns:
        Name of the best performing model
    """
    # TODO: Create a DataFrame from results for easy comparison
    # TODO: Sort by RMSE (ascending) or R² (descending)
    # TODO: Print formatted comparison table
    # TODO: Return best model name
    pass


def get_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[str, any, Dict[str, float]]:
    """
    Convenience function to train models, compare them, and return the best.
    
    TODO:
    1. Call train_multiple_models()
    2. Call compare_models() to find best model name
    3. Return tuple of (best_model_name, best_fitted_model, best_metrics)
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        Tuple of (model_name, fitted_model, metrics_dict)
    """
    # TODO: Train all models
    # TODO: Compare and find best
    # TODO: Return best model info
    pass


if __name__ == "__main__":
    # TODO: Import preprocessing functions
    # TODO: Load and preprocess data
    # TODO: Split into train/test
    # TODO: Train and compare models
    # TODO: Print best model results
    pass

