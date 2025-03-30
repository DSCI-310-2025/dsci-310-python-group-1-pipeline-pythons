import pandas as pd
from src.data_cleaning import apply_mappings
import pytest
from src.data_cleaning import load_and_prepare_raw_data

def test_apply_value_mappings_basic():
    df = pd.DataFrame({
        "Job": ["A171", "A172", "A173"],
        "Housing": ["A151", "A152", "A153"]
    })

    mappings = {
        "Job": {
            "A171": "Unemployed / Unskilled (Non-Resident)",
            "A172": "Unskilled (Resident)",
            "A173": "Skilled Employee / Official"
        },
        "Housing": {
            "A151": "Rent",
            "A152": "Own",
            "A153": "For Free"
        }
    }

    result = apply_mappings(df, mappings)

    assert result.loc[0, "Job"] == "Unemployed / Unskilled (Non-Resident)"
    assert result.loc[1, "Housing"] == "Own"

def test_apply_value_mappings_missing_column():
    df = pd.DataFrame({
        "Job": ["A171", "A172"]
    })
    mappings = {
        "MissingCol": {"X": "Y"}
    }
    result = apply_mappings(df, mappings)
    assert result.equals(df)

def test_apply_value_mappings_empty_df():
    df = pd.DataFrame()
    mappings = {
        "Any": {"A": "B"}
    }
    result = apply_mappings(df, mappings)
    assert result.empty

def test_load_and_prepare_raw_data_valid(tmp_path):
    # Create dummy file
    content = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A191 A201 1"
    file_path = tmp_path / "raw_data.txt"
    file_path.write_text(content)

    # Read with function
    df = load_and_prepare_raw_data(str(file_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 21)
    assert "CreditAmount" in df.columns

def test_load_and_prepare_raw_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_prepare_raw_data("non_existent_file.txt")

def test_load_and_prepare_raw_data_bad_format(tmp_path):
    # Create a poorly formatted file (wrong column count)
    content = "A11 6 A34 A43"
    file_path = tmp_path / "bad_data.txt"
    file_path.write_text(content)

    with pytest.raises(ValueError):
        load_and_prepare_raw_data(str(file_path))
