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
from data_cleaning import apply_mappings, load_and_prepare_raw_data


@click.command()
@click.option('--input', default="../data/raw/raw_data.csv", show_default=True, help="Path to raw input file")
@click.option('--output', default="../data/processed/german_processed.csv", show_default=True, help="Path to save processed data")
def preprocess_data(input, output):
    """Clean and preprocess the German Credit Data dataset."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    try:
        # Load and prepare the data using the abstracted function
        print(f"Loading data from {input}...")
        df = load_and_prepare_raw_data(input)
        
        # Define mappings for categorical variables
        mappings = {
            "Checking_Acc_Status": {
                "A11": "< 0 DM",
                "A12": "0-200 DM",
                "A13": ">= 200 DM or Salary Assigned",
                "A14": "No Checking Account"
            },
            "Credit_History": {
                "A30": "No Credit Taken / All Paid",
                "A31": "All Paid (Same Bank)",
                "A32": "All Paid (Other Banks)",
                "A33": "Past Delays in Payment",
                "A34": "Critical Account / Other Existing Credits"
            },
            "Purpose": {
                "A40": "New Car",
                "A41": "Used Car",
                "A42": "Furniture/Equipment",
                "A43": "Radio/TV",
                "A44": "Domestic Appliances",
                "A45": "Repairs",
                "A46": "Education",
                "A47": "Vacation",
                "A48": "Retraining",
                "A49": "Business",
                "A410": "Others"
            },
            "Savings_Acc": {
                "A61": "< 100 DM",
                "A62": "100-500 DM",
                "A63": "500-1000 DM",
                "A64": ">= 1000 DM",
                "A65": "No Savings Account"
            },
            "Employment": {
                "A71": "Unemployed",
                "A72": "< 1 Year",
                "A73": "1-4 Years",
                "A74": "4-7 Years",
                "A75": ">= 7 Years"
            },
            "Personal_Status": {
                "A91": "Male: Divorced/Separated",
                "A92": "Female: Divorced/Separated/Married",
                "A93": "Male: Single",
                "A94": "Male: Married/Widowed",
                "A95": "Female: Single"
            },
            "Other_Debtors": {
                "A101": "None",
                "A102": "Co-applicant",
                "A103": "Guarantor"
            },
            "Property": {
                "A121": "Real Estate",
                "A122": "Building Society Savings / Life Insurance",
                "A123": "Car or Other Property",
                "A124": "No Property"
            },
            "Other_Installment": {
                "A141": "Bank",
                "A142": "Stores",
                "A143": "None"
            },
            "Housing": {
                "A151": "Rent",
                "A152": "Own",
                "A153": "For Free"
            },
            "Job": {
                "A171": "Unemployed / Unskilled (Non-Resident)",
                "A172": "Unskilled (Resident)",
                "A173": "Skilled Employee / Official",
                "A174": "Management / Self-Employed / Highly Qualified"
            },
            "Telephone": {
                "A191": "No Telephone",
                "A192": "Yes, Registered"
            },
            "Foreign_Worker": {
                "A201": "Yes",
                "A202": "No"
            }
        }
        
        # Apply mappings to categorical columns
        df = apply_mappings(df, mappings)
        
        # Convert categorical data to numerical format using one-hot encoding
        df = pd.get_dummies(df, drop_first=True).astype(int)
        
        # Map the 'Credit Standing' column from 1 (good) and 2 (bad) to 0 (good) and 1 (bad)
        df['Credit Standing'] = df['Credit Standing'].map({1: 0, 2: 1})
        
        # Ensure all categorical columns are encoded as 0s and 1s
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes
        
        # Save preprocessed data
        print(f"Saving preprocessed data to {output}...")
        df.to_csv(output, index=False)
        
        print(f"Data preprocessing completed successfully.")
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        exit(1)

if __name__ == "__main__":
    preprocess_data() 