"""
Feature Importance Analysis

This module analyzes which features are most important for predictions.
Helps understand what drives RAM prices and identify potential feature engineering opportunities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List
from sklearn.base import BaseEstimator


def get_feature_importance(model: BaseEstimator, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    TODO:
    1. Get feature_importances_ from model (if tree-based model)
    2. For linear models, use coefficient absolute values
    3. Create DataFrame with columns: ['feature', 'importance']
    4. Sort by importance (descending)
    5. Return DataFrame
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names (in order used by model)
    
    Returns:
        DataFrame with features and their importance scores, sorted by importance
    """

    feature_importances = []
    #check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        feature_importances.append(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        feature_importances.append(np.abs(model.coef_))
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """
    Create a bar plot of feature importances.
    
    TODO:
    1. Select top N features from importance_df
    2. Create horizontal bar plot (barh) with:
       - Features on y-axis
       - Importance scores on x-axis
       - Color by importance value
    3. Add title and labels
    4. Add grid for readability
    5. Save to file if save_path provided
    6. Show plot
    
    Args:
        importance_df: DataFrame from get_feature_importance()
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    
    feature_importances = importance_df.head(top_n)
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=feature_importances, color='skyblue')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Complete feature importance analysis.
    
    TODO:
    1. Get feature importance DataFrame
    2. Print top N features with their importance scores
    3. Create visualization
    4. Calculate cumulative importance (what % of importance do top N features explain?)
    5. Return importance DataFrame
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to analyze
    
    Returns:
        DataFrame with feature importances
    """

    importance_df = get_feature_importance(model=model, feature_names=feature_names)
    print(f"Top {top_n} features by importance: {importance_df.head(top_n)}")
    plot_feature_importance(importance_df=importance_df, top_n=top_n)
    return importance_df


def compare_feature_importance_across_models(
    models: Dict[str, BaseEstimator],
    feature_names: List[str],
    top_n: int = 10
):
    """
    Compare feature importance across multiple models.
    
    TODO:
    1. Get feature importance for each model
    2. Create comparison plot showing importance from each model side-by-side
    3. Identify features that are consistently important across models
    4. Save comparison plot
    
    Args:
        models: Dictionary of model_name: trained_model
        feature_names: List of feature names
        top_n: Number of top features to compare
    """
    # TODO: Get importance for each model
    # TODO: Merge into single DataFrame
    # TODO: Create grouped bar plot or heatmap
    # TODO: Identify consistent important features
    # TODO: Display results
    pass


if __name__ == "__main__":
    # TODO: Import preprocessing and training functions
    # TODO: Load data and train a model
    # TODO: Get feature names (might need to handle encoded categorical features)
    # TODO: Analyze feature importance
    # TODO: Plot results
    pass

