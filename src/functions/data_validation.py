# src/data_validation.py

import pandas as pd

def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    return df.isnull().sum()

import pandas as pd

def check_data_types(df: pd.DataFrame, expected_types: dict) -> bool:
    """
    Validate that each column in the DataFrame has the expected data type.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        expected_types (dict): A dictionary where keys are column names and values are the
                               expected data type as strings (e.g. 'int64', 'object', etc.)
    
    Returns:
        bool: True if all columns match the expected data type.
    
    Raises:
        KeyError: If any expected column is missing.
        TypeError: If a column's data type does not match the expected type.
    """
    for col, exp_type in expected_types.items():
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in DataFrame")
        actual_type = df[col].dtype.name  # Get the dtype name, e.g. 'int64'
        if actual_type != exp_type:
            raise TypeError(f"Column {col} has type {actual_type}, expected {exp_type}")
    return True


def check_unique_values(df, column):
    """Check for unique values in a specific column."""
    return df[column].is_unique

def check_value_ranges(df, column, min_value, max_value):
    """Check if values in a column fall within a specified range."""
    return df[column].between(min_value, max_value).all()

def check_duplicates(df, subset=None):
    """Check for duplicate rows (or based on a subset) in the DataFrame."""
    return df.duplicated(subset=subset).sum()

def check_column_names(df, expected_columns):
    """Check if the DataFrame contains the expected columns."""
    return set(expected_columns).issubset(df.columns)

def check_outliers(df, column, threshold):
    """Check for outliers in a column based on a threshold."""
    return (df[column] > threshold).sum()

def check_date_format(df, column, date_format):
    """Check if the date column matches the expected format."""
    return pd.to_datetime(df[column], format=date_format, errors='coerce').notnull().all()