#!/usr/bin/env python
"""
Utility functions for loading data and producing visualizations for EDA.
"""

import os
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_corr_barplot(df: pd.DataFrame, target: str, save_path: str):
    corr = df.corr()[target].sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x=corr.values, y=corr.index)
    plt.title(f'Feature Correlation with {target}', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    for i, val in enumerate(corr.values):
        color = '#e74c3c' if val > 0 else '#3498db'
        bars.patches[i].set_color(color)
        plt.text(val + 0.01 if val >= 0 else val - 0.06, i, f'{val:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_credit_standing_distribution(df: pd.DataFrame, save_path: str):
    custom_palette = ['#3498db', '#e74c3c']
    counts = df['Credit Standing'].value_counts()

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=custom_palette)
    plt.title('Credit Standing Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Credit Standing (0=Good, 1=Bad)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Good (0)', 'Bad (1)'])

    total = len(df)
    for i, p in enumerate(ax.patches):
        pct = 100 * p.get_height() / total
        ax.annotate(f'{int(p.get_height())}\n({pct:.1f}%)',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distributions(df: pd.DataFrame, features: list, target: str, save_path: str):
    """
    Plots histograms (with an overlaid kernel density estimate) for each feature in 'features',
    colored by a target variable, and saves the plot to the specified path.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list): List of feature column names to be plotted.
        target (str): The column name for the target variable to use as hue in the plots.
        save_path (str): The file path where the resulting plot image will be saved.
    
    Raises:
        ValueError: If the DataFrame is empty.
        KeyError: If the target column or any feature is missing in the DataFrame.
        TypeError: If any feature column is not numeric (since histograms require numeric data).
    
    Returns:
        None. The plot is saved to the provided 'save_path'.
    """
    
    # Validation: Ensure the DataFrame is not empty.
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot generate plots on an empty DataFrame.")
    
    # Validation: Check that the target column exists in the DataFrame.
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    
    # Validation: For each feature, check the column exists and is numeric.
    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Feature column '{feature}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature column '{feature}' must be numeric to plot a histogram.")
    
    # Create subplots: one row per feature.
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4 * len(features)))
    
    # If there's only one feature, ensure 'axes' is a list (subplots returns an Axes object if single subplot)
    if len(features) == 1:
        axes = [axes]
    
    # Generate the plots for each feature.
    for i, feature in enumerate(features):
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            kde=True,  # Overlay a kernel density estimate.
            palette=['#3498db', '#e74c3c'],
            alpha=0.6,
            bins=30,
            ax=axes[i],
            hue_order=[0, 1]
        )
        axes[i].set_title(f"Distribution of {feature} by {target}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Frequency")
        
        # (Optional) Here you could add additional annotations, such as mean lines, if needed.
    
    # Adjust layout, save, and close the figure.
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()