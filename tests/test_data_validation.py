import pandas as pd
import pytest
from src.functions.data_validation import (
    check_missing_values,
    check_data_types,
    check_zero_variance_columns,
    check_value_ranges,
    check_duplicates,
    check_column_names,
    check_outliers,
    check_class_balance
)

def test_check_missing_values():
    """Test that missing values are counted correctly."""
    df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
    missing = check_missing_values(df)
    assert missing['A'] == 1
    assert missing['B'] == 0

def test_check_data_types_valid():
    """Test that correct data types return True."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    expected = {'A': 'int64', 'B': 'object'}
    assert check_data_types(df, expected) == True

def test_check_data_types_invalid():
    """Test that mismatched data types raise TypeError."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    expected = {'A': 'int64', 'B': 'int64'}
    with pytest.raises(TypeError):
        check_data_types(df, expected)

def test_check_zero_variance_columns():
    """Test detection of columns with zero variance."""
    df = pd.DataFrame({
        'A': [1, 1, 1],
        'B': [2, 3, 4],
        'C': [5, 5, 5]
    })
    zero_var = check_zero_variance_columns(df)
    assert set(zero_var) == {'A', 'C'}

def test_check_value_ranges_valid():
    """Test that valid range check passes."""
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert check_value_ranges(df, 'A', 0, 5) == True

def test_check_value_ranges_invalid():
    """Test that range check fails when value out of bounds."""
    df = pd.DataFrame({'A': [1, 2, 10]})
    assert check_value_ranges(df, 'A', 0, 5) == False

def test_check_duplicates_valid():
    """Test that no duplicates returns 0."""
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert check_duplicates(df) == 0

def test_check_duplicates_invalid():
    """Test that duplicate count is correct."""
    df = pd.DataFrame({'A': [1, 2, 2]})
    assert check_duplicates(df, subset=['A']) == 1

def test_check_column_names_valid():
    """Test that all expected columns are present."""
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    assert check_column_names(df, ['A', 'B']) == True

def test_check_column_names_invalid():
    """Test that missing columns return False."""
    df = pd.DataFrame(columns=['A', 'C'])
    assert check_column_names(df, ['A', 'B']) == False

def test_check_outliers_valid():
    """Test that no values exceed threshold."""
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert check_outliers(df, 'A', 100) == 0

def test_check_outliers_invalid():
    """Test detection of values above threshold."""
    df = pd.DataFrame({'A': [1, 2, 300]})
    assert check_outliers(df, 'A', 100) == 1

def test_check_class_balance_pass():
    """Test that a reasonably balanced target passes."""
    df = pd.DataFrame({'target': [0, 1, 0, 1]})
    assert check_class_balance(df, 'target', threshold=0.9) == True

def test_check_class_balance_fail():
    """Test that imbalanced classes fail the threshold check."""
    df = pd.DataFrame({'target': [1, 1, 1, 1, 1, 0]})
    assert check_class_balance(df, 'target', threshold=0.7) == False
