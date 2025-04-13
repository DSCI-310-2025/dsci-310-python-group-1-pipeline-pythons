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
from creditriskutilities import apply_mappings, load_and_prepare_raw_data

@click.command()
@click.option('--input', default="../data/raw/raw_data.csv", show_default=True, help="Path to raw input file")
@click.option('--output', default="../data/processed/german_processed.csv", show_default=True, help="Path to save processed data")
def preprocess_data(input, output):
    """Clean and preprocess the German Credit Data dataset."""

    # Create output directory if needed
    os.makedirs(os.path.dirname(output), exist_ok=True)

    try:
        print(f"Loading data from {input}...")
        df = load_and_prepare_raw_data(input)

        # --- DATA VALIDATION CHECKS (inline) ---

        # 1. Check for correct column names
        expected_cols = [
            "Checking_Acc_Status", "Duration (in months)", "Credit_History", "Purpose",
            "Credit_Amount", "Savings_Acc", "Employment", "Installment_Rate",
            "Personal_Status", "Other_Debtors", "Residence_Since", "Property",
            "Age", "Other_Installment", "Housing", "Existing_Credits",
            "Job", "Num_People_Maintained", "Telephone", "Foreign_Worker", "Credit Standing"
        ]
        if not set(expected_cols).issubset(df.columns):
            raise ValueError("Column mismatch: One or more expected columns are missing.")

        # 2. Check data types
        expected_types = {
            "Duration (in months)": "int64",
            "Credit_Amount": "int64",
            "Age": "int64",
            "Credit Standing": "int64"
        }
        for col, expected_type in expected_types.items():
            actual_type = df[col].dtype.name
            if actual_type != expected_type:
                raise TypeError(f"Incorrect type for column '{col}': expected {expected_type}, got {actual_type}")

        # 3. Check for missing values
        if df.isnull().sum().sum() > 0:
            raise ValueError("Missing values detected in dataset.")

        # 4. Check for duplicate rows
        if df.duplicated().sum() > 0:
            raise ValueError("Duplicate rows detected in dataset.")

        # 5. Check for outliers in credit amount
        if (df['Credit_Amount'] > 100000).sum() > 0:
            raise ValueError("Unrealistically high values found in 'Credit_Amount'.")

        # 6. Check for invalid category levels (zero-variance categorical columns)
        zero_var_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() <= 1]
        if zero_var_cols:
            raise ValueError(f"Categorical columns with only one level found: {zero_var_cols}")

        # 7. Check target class balance (not overly dominated)
        class_ratio = df['Credit Standing'].value_counts(normalize=True).max()
        if class_ratio > 0.9:
            raise ValueError("Class imbalance detected: one class exceeds 90% of total.")

        # 8. Check correlation anomalies between target and features
        df_encoded = pd.get_dummies(df.drop(columns=["Credit Standing"]), drop_first=True)
        df_encoded['Credit Standing'] = df['Credit Standing']
        correlations = df_encoded.corr()['Credit Standing'].drop('Credit Standing')
        if correlations.abs().max() < 0.01:
            raise ValueError("All features have near-zero correlation with target. Unexpected behavior.")

        # --- APPLY MAPPINGS & TRANSFORMATIONS ---
        mappings = {
            "Checking_Acc_Status": {
                "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM or Salary Assigned", "A14": "No Checking Account"
            },
            "Credit_History": {
                "A30": "No Credit Taken / All Paid", "A31": "All Paid (Same Bank)",
                "A32": "All Paid (Other Banks)", "A33": "Past Delays in Payment",
                "A34": "Critical Account / Other Existing Credits"
            },
            "Purpose": {
                "A40": "New Car", "A41": "Used Car", "A42": "Furniture/Equipment", "A43": "Radio/TV",
                "A44": "Domestic Appliances", "A45": "Repairs", "A46": "Education", "A47": "Vacation",
                "A48": "Retraining", "A49": "Business", "A410": "Others"
            },
            "Savings_Acc": {
                "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "No Savings Account"
            },
            "Employment": {
                "A71": "Unemployed", "A72": "< 1 Year", "A73": "1-4 Years", "A74": "4-7 Years", "A75": ">= 7 Years"
            },
            "Personal_Status": {
                "A91": "Male: Divorced/Separated", "A92": "Female: Divorced/Separated/Married", "A93": "Male: Single",
                "A94": "Male: Married/Widowed", "A95": "Female: Single"
            },
            "Other_Debtors": {
                "A101": "None", "A102": "Co-applicant", "A103": "Guarantor"
            },
            "Property": {
                "A121": "Real Estate", "A122": "Savings/Insurance", "A123": "Car/Other", "A124": "No Property"
            },
            "Other_Installment": {
                "A141": "Bank", "A142": "Stores", "A143": "None"
            },
            "Housing": {
                "A151": "Rent", "A152": "Own", "A153": "For Free"
            },
            "Job": {
                "A171": "Unemployed / Unskilled (Non-Resident)", "A172": "Unskilled (Resident)",
                "A173": "Skilled", "A174": "High-Skill/Management"
            },
            "Telephone": {
                "A191": "No Telephone", "A192": "Yes, Registered"
            },
            "Foreign_Worker": {
                "A201": "Yes", "A202": "No"
            }
        }

        df = apply_mappings(df, mappings)

        # One-hot encode categorical features
        df = pd.get_dummies(df, drop_first=True).astype(int)

        # Re-map target: 1 = good → 0; 2 = bad → 1
        df['Credit Standing'] = df['Credit Standing'].map({1: 0, 2: 1})

        # Double-check that all object columns are now encoded
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes

        print(f"Saving preprocessed data to {output}...")
        df.to_csv(output, index=False)
        print("Data preprocessing completed successfully.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        exit(1)

if __name__ == "__main__":
    preprocess_data()
