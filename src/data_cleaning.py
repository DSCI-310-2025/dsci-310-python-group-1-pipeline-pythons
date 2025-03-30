import pandas as pd

def apply_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Applies categorical value mappings to a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - mappings (dict): A dictionary where keys are column names and values are mapping dicts.

    Returns:
    - pd.DataFrame: The transformed DataFrame with applied mappings.
    """
    df_copy = df.copy()
    for col, mapping in mappings.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(mapping)
    return df_copy
