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

    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # For linear models, use absolute value of coefficients
    elif hasattr(model, 'coef_'):
        # For linear models, coef_ might be 1D or 2D depending on model
        coef = model.coef_
        if coef.ndim > 1:
            # If 2D (e.g., multi-output), take mean across outputs
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
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
    top_n: int = 10,
    save_path: str = None
):
    """
    Compare feature importance across multiple models.
    
    Args:
        models: Dictionary of model_name: trained_model
        feature_names: List of feature names
        top_n: Number of top features to compare
        save_path: Optional path to save the comparison plot
    """
    # Get feature importance for each model
    importance_dict = {}
    for model_name, model in models.items():
        importance_df = get_feature_importance(model, feature_names)
        importance_dict[model_name] = importance_df.set_index('feature')['importance']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(importance_dict)
    
    # Get top features (based on average importance across all models)
    comparison_df['avg_importance'] = comparison_df.mean(axis=1)
    top_features = comparison_df.nlargest(top_n, 'avg_importance').index
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting (only top features)
    plot_data = comparison_df.loc[top_features].drop('avg_importance', axis=1)
    
    # Create grouped bar chart
    x = np.arange(len(top_features))
    width = 0.25
    multiplier = 0
    
    for model_name in plot_data.columns:
        offset = width * multiplier
        ax.barh(x + offset, plot_data[model_name], width, label=model_name)
        multiplier += 1
    
    # Customize plot
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Feature Importance Comparison Across Models (Top {top_n})', fontsize=14)
    ax.set_yticks(x + width)
    ax.set_yticklabels(top_features)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nTop {top_n} features (by average importance across models):")
    print(comparison_df.loc[top_features].drop('avg_importance', axis=1).to_string())
    
    # Identify consistently important features
    print(f"\nFeatures consistently in top {top_n} across all models:")
    for feature in top_features:
        if all(comparison_df.loc[feature, model] > 0 for model in models.keys()):
            print(f"  - {feature}")


if __name__ == "__main__":
    """
    Main function to demonstrate feature importance analysis.
    
    This script:
    1. Loads and preprocesses data
    2. Trains multiple models
    3. Analyzes feature importance for each model
    4. Compares feature importance across models
    """
    from pathlib import Path
    from preprocessing import (
        load_data,
        parse_speed_column,
        parse_modules_column,
        prepare_target,
        split_data,
        get_column_groups,
        build_preprocessor,
        build_full_pipeline
    )
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    print("="*60)
    print("Feature Importance Analysis")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_path = Path(__file__).parent.parent / "data" / "memory.csv"
    df = load_data(data_path)
    df = parse_speed_column(df)
    df = parse_modules_column(df)
    X, y = prepare_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 2: Build preprocessor
    print("\n2. Building preprocessor...")
    column_groups = get_column_groups()
    preprocessor = build_preprocessor(
        column_groups['numerical'],
        column_groups['categorical']
    )
    preprocessor.fit(X_train)
    
    # Get feature names after preprocessing (handles one-hot encoding)
    feature_names = preprocessor.get_feature_names_out()
    print(f"   Number of features after preprocessing: {len(feature_names)}")
    
    # Step 3: Transform data
    print("\n3. Transforming data...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Step 4: Train multiple models
    print("\n4. Training models...")
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    for model_name, model in models.items():
        print(f"   Training {model_name}...")
        model.fit(X_train_transformed, y_train)
        trained_models[model_name] = model
    
    # Step 5: Analyze feature importance for each model
    print("\n5. Analyzing feature importance...")
    print("\n" + "-"*60)
    
    for model_name, model in trained_models.items():
        print(f"\n{model_name}:")
        print("-"*60)
        importance_df = get_feature_importance(model, feature_names)
        
        # Print top 10 features
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Calculate cumulative importance
        top_10_importance = importance_df.head(10)['importance'].sum()
        total_importance = importance_df['importance'].sum()
        cumulative_pct = (top_10_importance / total_importance) * 100
        print(f"\nTop 10 features explain {cumulative_pct:.1f}% of total importance")
    
    # Step 6: Visualize feature importance for best model (Random Forest)
    print("\n6. Creating visualization...")
    rf_model = trained_models['Random Forest']
    importance_df = get_feature_importance(rf_model, feature_names)
    
    # Create plot
    plot_feature_importance(
        importance_df=importance_df,
        top_n=15,
        figsize=(12, 8),
        save_path=Path(__file__).parent.parent / "plots" / "feature_importance.png"
    )
    
    # Step 7: Compare feature importance across models
    print("\n7. Comparing feature importance across models...")
    compare_feature_importance_across_models(
        models=trained_models,
        feature_names=feature_names,
        top_n=10
    )
    
    print("\n" + "="*60)
    print("Feature importance analysis complete!")
    print("="*60)

