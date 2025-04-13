#!/usr/bin/env python
"""
Preprocesses the German Credit Data dataset.

Usage:
    preprocess_data.py --input=<input_file> --output=<output_file>

Options:
    --input=<input_file>        Path to the raw data file
    --output=<output_file>      Path to save the preprocessed data
"""

import os
import click
import pandas as pd
import numpy as np
from functions.model_utils import load_and_prepare_raw_data
from creditriskutilities import apply_mappings
from functions.data_validation import (
    check_missing_values,
    check_data_types,
    check_zero_variance_columns,
    check_value_ranges,
    check_duplicates,
    check_column_names,
    check_outliers,
    check_class_balance
)

@click.command()
@click.option('--input', default="../data/raw/raw_data.csv", show_default=True, help="Path to raw input file")
@click.option('--output', default="../data/processed/german_processed.csv", show_default=True, help="Path to save processed data")
def preprocess_data(input, output):
    """
    Clean and preprocess the German Credit Data dataset.
    Performs data loading, value mapping, encoding, and validation.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)

    try:
        print(f"Loading data from {input}...")
        df = load_and_prepare_raw_data(input)

        # -------------------- Apply Mappings --------------------
        mappings = {
            "Checking_Acc_Status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM or Salary Assigned", "A14": "No Checking Account"},
            "Credit_History": {"A30": "No Credit Taken / All Paid", "A31": "All Paid (Same Bank)", "A32": "All Paid (Other Banks)",
                               "A33": "Past Delays in Payment", "A34": "Critical Account / Other Existing Credits"},
            "Purpose": {"A40": "New Car", "A41": "Used Car", "A42": "Furniture/Equipment", "A43": "Radio/TV", "A44": "Domestic Appliances",
                        "A45": "Repairs", "A46": "Education", "A47": "Vacation", "A48": "Retraining", "A49": "Business", "A410": "Others"},
            "Savings_Acc": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "No Savings Account"},
            "Employment": {"A71": "Unemployed", "A72": "< 1 Year", "A73": "1-4 Years", "A74": "4-7 Years", "A75": ">= 7 Years"},
            "Personal_Status": {"A91": "Male: Divorced/Separated", "A92": "Female: Divorced/Separated/Married", "A93": "Male: Single",
                                "A94": "Male: Married/Widowed", "A95": "Female: Single"},
            "Other_Debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
            "Property": {"A121": "Real Estate", "A122": "Building Society Savings / Life Insurance", "A123": "Car or Other Property", "A124": "No Property"},
            "Other_Installment": {"A141": "Bank", "A142": "Stores", "A143": "None"},
            "Housing": {"A151": "Rent", "A152": "Own", "A153": "For Free"},
            "Job": {"A171": "Unemployed / Unskilled (Non-Resident)", "A172": "Unskilled (Resident)", "A173": "Skilled Employee / Official", "A174": "Management / Self-Employed / Highly Qualified"},
            "Telephone": {"A191": "No Telephone", "A192": "Yes, Registered"},
            "Foreign_Worker": {"A201": "Yes", "A202": "No"}
        }

        df = apply_mappings(df, mappings)

        # -------------------- Encode Categorical Values --------------------
        df = pd.get_dummies(df, drop_first=True).astype(int)

        # Map Credit Standing to 0 = good, 1 = bad
        df['Credit Standing'] = df['Credit Standing'].map({1: 0, 2: 1})

        # Final check for any remaining object dtypes
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes

        # -------------------- VALIDATION CHECKS --------------------

        print("Performing validation checks...")

        # 1. Check column names
        expected_columns = [
            "Duration (in months)", "Credit_Amount", "Age", "Installment_Rate", "Residence_Since",
            "Existing_Credits", "Num_People_Maintained", "Credit Standing"
        ]
        if not check_column_names(df, expected_columns):
            raise ValueError("DataFrame is missing one or more expected columns.")

        # 2. Check for missing values
        missing = check_missing_values(df)
        if missing.sum() > 0:
            raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")

        # 3. Check data types
        expected_types = {
            'Credit_Amount': 'int64',
            'Duration (in months)': 'int64',
            'Age': 'int64',
            'Credit Standing': 'int64'
        }
        check_data_types(df, expected_types)

        # 4. Check value ranges
        if not check_value_ranges(df, 'Age', 18, 100):
            raise ValueError("Age values fall outside expected range [18, 100]")

       # 5. Check for zero-variance columns
        zero_var_cols = check_zero_variance_columns(df)
        if zero_var_cols:
            print(f"Warning: The following columns have zero variance and may be dropped: {zero_var_cols}")

        # 6. Check duplicates
        duplicate_count = check_duplicates(df)
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate rows.")

        # 7. Check outliers
        outliers = check_outliers(df, 'Credit_Amount', threshold=20000)
        if outliers > 0:
            print(f"Note: Detected {outliers} outliers in 'Credit_Amount' > 20,000.")

        # 8. Check class balance
        if not check_class_balance(df, 'Credit Standing', threshold=0.9):
            raise ValueError("Class imbalance detected in target column 'Credit Standing'")

        # -------------------- Save Final Output --------------------
        print(f"Saving preprocessed data to {output}...")
        df.to_csv(output, index=False)
        print("Data preprocessing completed successfully.")

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        exit(1)


if __name__ == "__main__":
    preprocess_data()
