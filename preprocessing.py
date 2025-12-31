"""
Data Summary:
- Total rows: 5522
- Missing values: price (2783), speed (2), price_per_gb (2837), 
                  color (505), first_word_latency (95), cas_latency (33)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Purpose: Load raw CSV data into a DataFrame.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Raw DataFrame
    """
    df = pd.read_csv(filepath)
    return df


# =============================================================================
# STEP 2: PARSE COMPOUND COLUMNS
# =============================================================================
def parse_speed_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Split 'speed' column (e.g., "5,6000") into separate columns.
    
    Creates:
        - ddr_gen: DDR generation (4, 5, etc.)
        - speed_mhz: Memory speed in MHz
    
    Args:
        df: DataFrame with 'speed' column
    
    Returns:
        DataFrame with new columns added
    """
    # TODO: Use str.split(',', expand=True) and pd.to_numeric()
    pass


def parse_modules_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Split 'modules' column (e.g., "2,16") into separate columns.
    
    Creates:
        - module_count: Number of modules (sticks)
        - module_size: Size per module in GB
        - total_capacity: module_count * module_size
    
    Args:
        df: DataFrame with 'modules' column
    
    Returns:
        DataFrame with new columns added
    """
    # TODO: Use str.split(',', expand=True) and pd.to_numeric()
    pass


# =============================================================================
# STEP 3: HANDLE TARGET VARIABLE
# =============================================================================
def prepare_target(df: pd.DataFrame, target_col: str = 'price') -> tuple[pd.DataFrame, pd.Series]:
    """
    Purpose: Separate features from target and remove rows with missing target.
    
    Why: Can't train on rows where we don't know the price.
    
    Args:
        df: Full DataFrame
        target_col: Name of target column
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # TODO: Drop rows where target is NaN, then separate X and y
    pass


# =============================================================================
# STEP 4: TRAIN-TEST SPLIT
# =============================================================================
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Purpose: Create train/test split BEFORE any preprocessing to avoid data leakage.
    
    Why: Preprocessing (like computing median for imputation) should only use 
         training data, not test data.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Fraction for test set (default 20%)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Use train_test_split with random_state for reproducibility
    pass


# =============================================================================
# STEP 5: DEFINE COLUMN GROUPS
# =============================================================================
def get_column_groups() -> dict:
    """
    Purpose: Define which columns are numerical vs categorical.
    
    Returns:
        Dict with 'numerical' and 'categorical' keys
    """
    return {
        'numerical': [
            'speed_mhz',          # Memory speed
            'cas_latency',        # CAS latency
            'first_word_latency', # First word latency (ns)
            'total_capacity',     # Total GB
        ],
        'categorical': [
            'ddr_gen',   # DDR generation (4 or 5)
            'color',     # RAM color/style
        ],
        'drop': [
            'name',         # Too many unique values
            'speed',        # Already parsed
            'modules',      # Already parsed
            'price_per_gb', # Derived from target (data leakage!)
        ]
    }


# =============================================================================
# STEP 6: BUILD PREPROCESSING PIPELINES
# =============================================================================
def build_numerical_pipeline() -> Pipeline:
    """
    Purpose: Create pipeline for numerical features.
    
    Steps:
        1. SimpleImputer(strategy='median') - Fill missing with median
        2. StandardScaler() - Normalize to mean=0, std=1
    
    Why median? Robust to outliers (prices are often skewed).
    Why scale? Many ML algorithms perform better with normalized features.
    """
    # TODO: Return Pipeline with imputer and scaler
    pass


def build_categorical_pipeline() -> Pipeline:
    """
    Purpose: Create pipeline for categorical features.
    
    Steps:
        1. SimpleImputer(strategy='most_frequent') - Fill missing with mode
        2. OneHotEncoder(handle_unknown='ignore') - Convert to binary columns
    
    Why most_frequent? Reasonable guess for missing categories.
    Why handle_unknown='ignore'? Test set might have categories not in training.
    """
    # TODO: Return Pipeline with imputer and encoder
    pass


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Purpose: Combine numerical and categorical pipelines using ColumnTransformer.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
    
    Returns:
        ColumnTransformer that applies correct pipeline to each column group
    """
    # TODO: Use ColumnTransformer with the two pipelines
    pass


# =============================================================================
# STEP 7: CUSTOM TRANSFORMERS (OPTIONAL)
# =============================================================================
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Purpose: Apply log transformation to handle right-skewed features.
    
    Why: Price data is often right-skewed. Log transform makes it more normal,
         which can improve linear model performance.
    
    Usage: Add to numerical pipeline if features are heavily skewed.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # TODO: Return np.log1p(X) to handle zeros
        pass


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Purpose: Select specific columns from DataFrame.
    
    Why: Useful at the start of a pipeline to pick only the columns you need.
    """
    
    def __init__(self, columns: list):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # TODO: Return X[self.columns]
        pass


# =============================================================================
# STEP 8: FULL PIPELINE WITH MODEL
# =============================================================================
def build_full_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """
    Purpose: Chain preprocessing with model for clean fit/predict workflow.
    
    Benefits:
        - Single .fit() call does preprocessing + training
        - Single .predict() call does preprocessing + prediction
        - Easy to save/load the complete pipeline
        - No risk of forgetting to preprocess at prediction time
    
    Args:
        preprocessor: The ColumnTransformer from step 6
        model: Any sklearn-compatible model (e.g., LinearRegression())
    
    Returns:
        Complete Pipeline
    """
    # TODO: Return Pipeline([('preprocessor', preprocessor), ('model', model)])
    pass


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Main function to run the full preprocessing pipeline.
    
    Workflow:
        1. Load raw data
        2. Parse compound columns (speed, modules)
        3. Prepare target (drop rows with missing price)
        4. Train-test split (BEFORE preprocessing!)
        5. Build preprocessor
        6. Build full pipeline with model
        7. Fit on training data
        8. Evaluate on test data
    """
    # TODO: Implement the full workflow
    pass


if __name__ == "__main__":
    main()
