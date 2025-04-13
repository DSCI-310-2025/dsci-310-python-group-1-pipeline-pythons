# src/data_validation.py

import pandas as pd

def check_missing_values(df):
    """
    Check for missing values in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to check.
        
    Returns:
    --------
    pd.Series
        Series with count of missing values per column.
    """
    return df.isnull().sum()


def check_data_types(df: pd.DataFrame, expected_types: dict) -> bool:
    """
    Validate that each column in the DataFrame has the expected data type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    expected_types : dict
        Dictionary where keys are column names and values are expected data type strings (e.g. 'int64', 'object').
        
    Returns:
    --------
    bool
        True if all data types match expected values.
        
    Raises:
    -------
    KeyError
        If a specified column is missing.
    TypeError
        If any column has an unexpected data type.
    """
    for col, exp_type in expected_types.items():
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in DataFrame")
        actual_type = df[col].dtype.name
        if actual_type != exp_type:
            raise TypeError(f"Column {col} has type {actual_type}, expected {exp_type}")
    return True


def check_zero_variance_columns(df):
    """
    Check for columns with zero variance in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list
        List of column names with zero variance.
    """
    return [col for col in df.columns if df[col].nunique() <= 1]


def check_value_ranges(df, column, min_value, max_value):
    """
    Check whether values in a column fall within a specified range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to validate.
    min_value : numeric
        Minimum acceptable value.
    max_value : numeric
        Maximum acceptable value.
        
    Returns:
    --------
    bool
        True if all values are within the range, False otherwise.
    """
    return df[column].between(min_value, max_value).all()


def check_duplicates(df, subset=None):
    """
    Check for duplicate rows in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    subset : list or None
        Optional list of columns to consider for identifying duplicates.
        
    Returns:
    --------
    int
        Number of duplicate rows found.
    """
    return df.duplicated(subset=subset).sum()


def check_column_names(df, expected_columns):
    """
    Check if the DataFrame contains the expected set of columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    expected_columns : list
        List of required column names.
        
    Returns:
    --------
    bool
        True if all expected columns are present, False otherwise.
    """
    return set(expected_columns).issubset(df.columns)


def check_outliers(df, column, threshold):
    """
    Check for outliers in a column based on a threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to check.
    threshold : numeric
        Threshold above which values are considered outliers.
        
    Returns:
    --------
    int
        Number of values exceeding the threshold.
    """
    return (df[column] > threshold).sum()


def check_class_balance(df, column, threshold=0.9):
    """
    Check if a single class dominates the target column beyond a threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to check for class balance.
    threshold : float
        Max allowed proportion for the majority class (default is 0.9).
        
    Returns:
    --------
    bool
        True if class balance is acceptable, False if too imbalanced.
    """
    proportions = df[column].value_counts(normalize=True)
    return proportions.max() <= threshold
