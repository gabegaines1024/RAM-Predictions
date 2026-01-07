"""
Visualization and Analysis

This module creates visualizations to understand model performance,
data distributions, and relationships between features and price.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.pipeline import Pipeline


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
):
    """
    Create residual plots to check model assumptions.
    
    TODO:
    1. Calculate residuals (y_true - y_pred)
    2. Create figure with 2 subplots side-by-side:
       a) Residuals vs Predicted values (scatter plot)
       b) Residuals distribution (histogram or Q-Q plot)
    3. Add horizontal line at y=0 for first subplot
    4. Check for patterns (should be random scatter if model is good)
    5. Save if save_path provided
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Optional path to save plot
    """
    # TODO: Calculate residuals
    # TODO: Create subplots
    # TODO: Plot residuals vs predictions
    # TODO: Plot residual distribution
    # TODO: Add labels, titles, grid
    # TODO: Save if needed
    pass


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None
):
    """
    Create scatter plot of predicted vs actual values.
    
    TODO:
    1. Create scatter plot: y_pred on x-axis, y_true on y-axis
    2. Add diagonal line (perfect predictions)
    3. Calculate and display R² on plot
    4. Color points by density if many points (hexbin or 2D histogram)
    5. Add labels and title
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Optional path to save plot
    """
    # TODO: Create scatter plot
    # TODO: Add diagonal line
    # TODO: Add R² text
    # TODO: Add labels and styling
    # TODO: Save if needed
    pass


def plot_prediction_errors(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Visualize prediction errors across different price ranges.
    
    TODO:
    1. Calculate errors (y_true - y_pred)
    2. Create subplots:
       a) Error distribution histogram
       b) Absolute error vs actual price (scatter)
       c) Box plot of errors by price bins
    3. Identify if model performs better at certain price ranges
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Optional path to save plot
    """
    # TODO: Calculate errors
    # TODO: Create subplots
    # TODO: Plot error distributions
    # TODO: Analyze patterns
    # TODO: Save if needed
    pass


def plot_feature_vs_price(
    data: pd.DataFrame,
    feature: str,
    price_col: str = 'price',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot relationship between a feature and price.
    
    TODO:
    1. Create scatter plot: feature on x-axis, price on y-axis
    2. Add trend line (linear regression line)
    3. If categorical feature, create box plot instead
    4. Add correlation coefficient to title
    5. Save if path provided
    
    Args:
        data: DataFrame with features and price
        feature: Feature name to plot against price
        price_col: Name of price column
        figsize: Figure size
        save_path: Optional path to save plot
    """
    # TODO: Check if feature is numeric or categorical
    # TODO: Create appropriate plot type
    # TODO: Add trend line and stats
    # TODO: Save if needed
    pass


def create_model_comparison_plot(
    results: dict,
    metric: str = 'r2',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Create bar chart comparing models by a metric.
    
    TODO:
    1. Extract metric values for each model from results dict
    2. Create horizontal bar chart
    3. Sort by metric value (highest first)
    4. Add value labels on bars
    5. Add title and labels
    
    Args:
        results: Dictionary from model_training.train_multiple_models()
        metric: Metric to compare ('r2', 'rmse', 'mae')
        figsize: Figure size
        save_path: Optional path to save plot
    """
    # TODO: Extract metric values
    # TODO: Create bar plot
    # TODO: Sort and style
    # TODO: Save if needed
    pass


def plot_learning_curve(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot learning curve to check for overfitting/underfitting.
    
    TODO:
    1. Use sklearn.model_selection.learning_curve()
    2. Plot training score and validation score vs training set size
    3. Add shaded regions for std dev
    4. Identify if model needs more data (gaps) or is overfitting (training >> validation)
    5. Save if path provided
    
    Args:
        model: Model or pipeline to evaluate
        X: Features
        y: Target
        cv: Number of cross-validation folds
        train_sizes: Array of training set sizes to evaluate
        figsize: Figure size
        save_path: Optional path to save plot
    """
    from sklearn.model_selection import learning_curve
    # TODO: Get learning curve data
    # TODO: Plot training and validation scores
    # TODO: Add styling and labels
    # TODO: Save if needed
    pass


if __name__ == "__main__":
    # TODO: Import preprocessing and training functions
    # TODO: Load data and train model
    # TODO: Make predictions
    # TODO: Create various visualizations
    # TODO: Save plots to 'plots' or 'figures' directory
    pass

