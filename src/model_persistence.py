"""
Model Persistence

This module handles saving and loading trained models and pipelines.
Save your best model so you can use it later without retraining.
"""

import joblib
import pickle
from pathlib import Path
from typing import Optional, Any
from sklearn.pipeline import Pipeline
import json
from datetime import datetime


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[dict] = None,
    format: str = 'joblib'
):
    """
    Save a trained model to disk.
    
    TODO:
    1. Create parent directories if they don't exist
    2. If format is 'joblib':
       - Use joblib.dump(model, filepath)
       - Joblib is better for sklearn models (handles numpy arrays efficiently)
    3. If format is 'pickle':
       - Use pickle.dump(model, open(filepath, 'wb'))
    4. If metadata provided:
       - Save metadata as JSON file alongside model (same name with .json extension)
       - Include timestamp, model type, performance metrics, etc.
    5. Print confirmation message
    
    Args:
        model: Trained model or pipeline to save
        filepath: Path where to save the model
        metadata: Optional dictionary with model info (metrics, training date, etc.)
        format: 'joblib' or 'pickle' (default: 'joblib')
    """
    # TODO: Create directory if needed
    # TODO: Save model using specified format
    # TODO: Save metadata if provided
    # TODO: Print confirmation
    pass


def load_model(
    filepath: str,
    format: str = 'joblib',
    metadata_path: Optional[str] = None
) -> tuple[Any, Optional[dict]]:
    """
    Load a saved model from disk.
    
    TODO:
    1. Check if file exists
    2. Load model using specified format
    3. Load metadata JSON if metadata_path provided or if .json file exists alongside
    4. Return tuple of (model, metadata)
    
    Args:
        filepath: Path to saved model file
        format: 'joblib' or 'pickle' (should match what was used to save)
        metadata_path: Optional path to metadata JSON file
    
    Returns:
        Tuple of (loaded_model, metadata_dict)
    """
    # TODO: Check file exists
    # TODO: Load model
    # TODO: Load metadata if available
    # TODO: Return model and metadata
    pass


def save_pipeline(
    pipeline: Pipeline,
    filepath: str,
    metrics: Optional[dict] = None
):
    """
    Save a complete preprocessing + model pipeline with metadata.
    
    TODO:
    1. Create metadata dictionary with:
       - Training timestamp
       - Model type
       - Evaluation metrics (if provided)
       - Pipeline steps info
    2. Save pipeline using save_model()
    3. Save metadata alongside
    
    Args:
        pipeline: Complete sklearn Pipeline (preprocessing + model)
        filepath: Path to save pipeline
        metrics: Optional evaluation metrics dictionary
    """
    # TODO: Create metadata dict
    # TODO: Add metrics to metadata
    # TODO: Save pipeline and metadata
    pass


def load_pipeline(filepath: str) -> tuple[Pipeline, dict]:
    """
    Load a saved pipeline with its metadata.
    
    TODO:
    1. Load pipeline and metadata
    2. Print metadata info (metrics, training date)
    3. Return pipeline and metadata
    
    Args:
        filepath: Path to saved pipeline file
    
    Returns:
        Tuple of (pipeline, metadata)
    """
    # TODO: Load pipeline and metadata
    # TODO: Print metadata summary
    # TODO: Return pipeline and metadata
    pass


def create_model_registry(
    models_dir: str = 'models'
) -> pd.DataFrame:
    """
    Create a registry of all saved models with their metadata.
    
    TODO:
    1. Scan models directory for saved models
    2. Load metadata from each model
    3. Create DataFrame with columns:
       - model_name
       - filepath
       - training_date
       - model_type
       - rmse, mae, r2 (metrics)
    4. Sort by performance (e.g., R² descending)
    5. Return registry DataFrame
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        DataFrame with model registry information
    """
    import pandas as pd
    # TODO: Find all model files in directory
    # TODO: Load metadata from each
    # TODO: Create registry DataFrame
    # TODO: Return registry
    pass


if __name__ == "__main__":
    # TODO: Import training and preprocessing functions
    # TODO: Train a model/pipeline
    # TODO: Evaluate it
    # TODO: Save with metadata
    # TODO: Load it back and verify it works
    pass

