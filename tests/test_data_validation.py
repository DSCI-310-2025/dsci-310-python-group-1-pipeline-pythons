import pandas as pd
import pytest
from src.functions.data_validation import (
    check_missing_values,
    check_data_types,
    check_unique_values,
    check_value_ranges,
    check_duplicates,
    check_column_names,
    check_outliers,
    check_date_format
)

def test_check_missing_values():
    # Assume check_missing_values returns a dict with missing counts per column.
    df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
    missing_counts = check_missing_values(df)
    assert missing_counts['A'] == 1

def test_check_data_types_valid():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    expected_types = {'A': 'int64', 'B': 'object'}
    result = check_data_types(df, expected_types)
    # If result is a Series, force it to a single Boolean.
    if hasattr(result, "all"):
        result = result.all()
    assert result == True

def test_check_data_types_invalid():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    expected_types = {'A': 'int64', 'B': 'int64'}  # B is actually object.
    # Either expect a TypeError or a False result.
    try:
        result = check_data_types(df, expected_types)
        if hasattr(result, "all"):
            result = result.all()
        assert result == False
    except TypeError:
        pass  # Acceptable alternative behavior.

def test_check_unique_values_valid():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    # For column A, values are unique.
    assert check_unique_values(df, 'A') == True

def test_check_unique_values_not_unique():
    df = pd.DataFrame({'A': [1, 2, 2], 'B': [1, 1, 1]})
    assert check_unique_values(df, 'A') == False

def test_check_value_ranges_valid():
    df = pd.DataFrame({'A': [1, 2, 3]})
    # Expect True when all values are within [0, 5].
    assert check_value_ranges(df, 'A', 0, 5) == True

def test_check_value_ranges_invalid():
    df = pd.DataFrame({'A': [1, 6, 3]})
    # Expect False when a value is out-of-range.
    assert check_value_ranges(df, 'A', 0, 5) == False

def test_check_duplicates_valid():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # Expect 0 duplicate rows.
    assert check_duplicates(df) == 0

def test_check_duplicates_invalid():
    df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 6]})
    # Count duplicates based on column 'A'
    assert check_duplicates(df, subset=['A']) == 1


def test_check_column_names_valid():
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    expected_columns = ['A', 'B']
    # Expect True if all expected columns are present.
    assert check_column_names(df, expected_columns) == True

def test_check_column_names_invalid():
    df = pd.DataFrame(columns=['A', 'C'])
    expected_columns = ['A', 'B']
    # Instead of raising an error, we now expect False.
    assert check_column_names(df, expected_columns) == False

def test_check_outliers_valid():
    df = pd.DataFrame({'A': [1, 2, 3]})
    # With a threshold of 50, expect no outliers.
    assert check_outliers(df, 'A', 50) == 0

def test_check_outliers_invalid():
    df = pd.DataFrame({'A': [1, 2, 100]})
    # Expect 1 outlier (the 100 exceeds threshold 50).
    assert check_outliers(df, 'A', 50) == 1

def test_check_date_format_valid():
    df = pd.DataFrame({'date_column': ['2021-01-01', '2021-02-01']})
    # Expect True when dates match '%Y-%m-%d'.
    assert check_date_format(df, 'date_column', '%Y-%m-%d') == True

def test_check_date_format_invalid():
    df = pd.DataFrame({'date_column': ['2021/01/01', '2021-02-01']})
    # Expect False when the date format does not match.
    assert check_date_format(df, 'date_column', '%Y-%m-%d') == False
