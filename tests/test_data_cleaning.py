import pandas as pd
from src.data_cleaning import apply_mappings

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
