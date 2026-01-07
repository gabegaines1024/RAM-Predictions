"""
Hyperparameter Tuning

This module implements hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
Find the best hyperparameters for your model to improve performance.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
from typing import Dict, Any
import pandas as pd


def create_param_grid(model_name: str) -> Dict[str, list]:
    """
    Create parameter grid for different model types.
    
    TODO:
    1. For 'random_forest', return dict with:
       - 'n_estimators': [50, 100, 200, 300]
       - 'max_depth': [None, 10, 20, 30]
       - 'min_samples_split': [2, 5, 10]
       - 'min_samples_leaf': [1, 2, 4]
    
    2. For 'gradient_boosting', return dict with:
       - 'n_estimators': [50, 100, 200]
       - 'learning_rate': [0.01, 0.1, 0.2]
       - 'max_depth': [3, 5, 7]
    
    3. Add support for other models if needed
    
    Args:
        model_name: Name of the model type ('random_forest', 'gradient_boosting', etc.)
    
    Returns:
        Dictionary with parameter names as keys and lists of values to try
    """
    # TODO: Implement parameter grids for different models
    pass


def grid_search_tuning(
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = -1,
    verbose: int = 1
) -> GridSearchCV:
    """
    Perform grid search cross-validation to find best hyperparameters.
    
    TODO:
    1. Create a GridSearchCV object with:
       - estimator=model
       - param_grid=param_grid
       - cv=cv (number of folds)
       - scoring=scoring (or use make_scorer for RMSE)
       - n_jobs=n_jobs (parallel processing)
       - verbose=verbose
    
    2. Fit the GridSearchCV on X_train, y_train
    
    3. Print best parameters and best score
    
    4. Return the fitted GridSearchCV object
    
    Args:
        model: Base model instance (unfitted)
        param_grid: Dictionary of parameters to search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric (default: 'neg_mean_squared_error')
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level
    
    Returns:
        Fitted GridSearchCV object
    """
    # TODO: Create custom scorer for RMSE if needed (make_scorer)
    # TODO: Initialize GridSearchCV
    # TODO: Fit on training data
    # TODO: Print best parameters and score
    # TODO: Return fitted GridSearchCV
    pass


def randomized_search_tuning(
    model: Any,
    param_distributions: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> RandomizedSearchCV:
    """
    Perform randomized search cross-validation (faster than grid search).
    
    TODO:
    1. Create RandomizedSearchCV object similar to GridSearchCV
    2. Use n_iter to limit number of combinations tried (much faster)
    3. Fit on training data
    4. Print best parameters and score
    5. Return fitted RandomizedSearchCV
    
    Args:
        model: Base model instance
        param_distributions: Dictionary of parameters (can use distributions from scipy.stats)
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_state: Random seed for reproducibility
        verbose: Verbosity level
    
    Returns:
        Fitted RandomizedSearchCV object
    """
    # TODO: Initialize RandomizedSearchCV
    # TODO: Fit on training data
    # TODO: Print results
    # TODO: Return fitted RandomizedSearchCV
    pass


def evaluate_tuned_model(
    search_cv: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate the best model from hyperparameter search on test set.
    
    TODO:
    1. Get the best estimator from search_cv (search_cv.best_estimator_)
    2. Make predictions on X_test
    3. Calculate RMSE, MAE, R²
    4. Compare with baseline model if provided
    5. Return metrics dictionary
    
    Args:
        search_cv: Fitted GridSearchCV or RandomizedSearchCV object
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with evaluation metrics
    """
    # TODO: Get best model from search_cv
    # TODO: Make predictions
    # TODO: Calculate metrics
    # TODO: Return metrics
    pass


if __name__ == "__main__":
    # TODO: Import preprocessing functions
    # TODO: Load and preprocess data
    # TODO: Split into train/test
    # TODO: Create base model
    # TODO: Define parameter grid
    # TODO: Run hyperparameter tuning (GridSearch or RandomizedSearch)
    # TODO: Evaluate best model on test set
    pass

