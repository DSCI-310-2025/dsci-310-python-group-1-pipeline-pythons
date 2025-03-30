import pandas as pd
import os


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



def load_and_prepare_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw German credit data and returns a cleaned DataFrame with assigned column names.

    Parameters:
    ------------
    filepath : str
        The path to the raw data file.

    Returns:
    --------
    pd.DataFrame
        A cleaned DataFrame with proper column names.

    Raises:
    -------
    FileNotFoundError
        If the file does not exist at the specified path.
    pd.errors.ParserError
        If the file cannot be parsed by pandas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at path: {filepath}")

    try:
        df = pd.read_csv(filepath, sep=" ", header=None)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse file: {e}")

    # Define columns (example, replace with actual column names)
    column_names = [
            "Checking_Acc_Status", "Duration (in months)", "Credit_History", "Purpose",
            "Credit_Amount", "Savings_Acc", "Employment", "Installment_Rate",
            "Personal_Status", "Other_Debtors", "Residence_Since", "Property",
            "Age", "Other_Installment", "Housing", "Existing_Credits",
            "Job", "Num_People_Maintained", "Telephone", "Foreign_Worker", "Credit Standing"
        ]
    

    if len(df.columns) != len(column_names):
        raise ValueError("Number of columns in raw data does not match expected column names")

    df.columns = column_names

    return df
