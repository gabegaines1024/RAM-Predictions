"""
Prediction Functions

This module provides functions to make predictions on new data.
Use your trained model to predict RAM prices for new products.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
from sklearn.pipeline import Pipeline
from src.model_persistence import load_pipeline


def predict_price(
    pipeline: Pipeline,
    data: Union[pd.DataFrame, Dict[str, any]],
    return_confidence: bool = False
) -> Union[float, np.ndarray, pd.DataFrame]:
    """
    Predict RAM price(s) for new data.
    
    TODO:
    1. If data is a dictionary (single row), convert to DataFrame
    2. If data is DataFrame, use as-is
    3. Ensure data has all required features (might need to add missing columns)
    4. Use pipeline.predict() to get predictions
    5. If return_confidence is True:
       - For tree models, could use prediction intervals or feature importance
       - Or return prediction with uncertainty estimate
    6. Format output nicely (single value for single input, array for multiple)
    7. Return predictions
    
    Args:
        pipeline: Trained sklearn Pipeline (preprocessing + model)
        data: Input data - DataFrame or dict with feature names as keys
        return_confidence: Whether to return confidence/uncertainty estimates
    
    Returns:
        Predicted price(s) - float for single input, array/DataFrame for multiple
    """
    # TODO: Convert dict to DataFrame if needed
    # TODO: Ensure all required features present
    # TODO: Make predictions
    # TODO: Format output
    # TODO: Return predictions (with confidence if requested)
    pass


def predict_from_csv(
    pipeline_path: str,
    data_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Load model, make predictions on CSV data, and optionally save results.
    
    TODO:
    1. Load pipeline from pipeline_path
    2. Load new data from data_path (CSV)
    3. Make predictions using predict_price()
    4. Create DataFrame with original data + predictions column
    5. If output_path provided, save to CSV
    6. Return DataFrame with predictions
    
    Args:
        pipeline_path: Path to saved pipeline
        data_path: Path to CSV file with new data
        output_path: Optional path to save predictions CSV
    
    Returns:
        DataFrame with original data and 'predicted_price' column
    """
    # TODO: Load pipeline
    # TODO: Load new data
    # TODO: Make predictions
    # TODO: Add predictions as new column
    # TODO: Save if output_path provided
    # TODO: Return results DataFrame
    pass


def batch_predict(
    pipeline: Pipeline,
    data_list: List[Dict[str, any]]
) -> pd.DataFrame:
    """
    Make predictions for a list of data samples.
    
    TODO:
    1. Convert list of dicts to DataFrame
    2. Use predict_price() to get all predictions
    3. Create results DataFrame with input features + predictions
    4. Return results
    
    Args:
        pipeline: Trained pipeline
        data_list: List of dictionaries, each representing one sample
    
    Returns:
        DataFrame with input features and predictions
    """
    # TODO: Convert to DataFrame
    # TODO: Make predictions
    # TODO: Combine and return
    pass


def validate_input_data(
    data: pd.DataFrame,
    required_features: List[str]
) -> tuple[bool, List[str]]:
    """
    Validate that input data has all required features.
    
    TODO:
    1. Check if all required_features are present in data columns
    2. Return tuple of (is_valid, missing_features)
    3. Print helpful error message if features missing
    
    Args:
        data: Input DataFrame
        required_features: List of feature names that must be present
    
    Returns:
        Tuple of (is_valid: bool, missing_features: List[str])
    """
    # TODO: Check for missing features
    # TODO: Return validation result
    pass


def example_usage():
    """
    Example of how to use prediction functions.
    
    TODO: Create a working example showing:
    1. Loading a saved pipeline
    2. Creating sample data (dict or DataFrame)
    3. Making predictions
    4. Printing results
    """
    # TODO: Example code here
    pass


if __name__ == "__main__":
    # TODO: Run example_usage() or create your own test
    example_usage()

