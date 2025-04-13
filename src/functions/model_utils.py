#!/usr/bin/env python
"""
Utility functions for loading data and producing visualizations for EDA.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw German credit data and returns a cleaned DataFrame with assigned column names.

    Parameters
    ----------
    filepath : str
        The path to the raw data file.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with proper column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.ParserError
        If the file cannot be parsed.
    ValueError
        If the number of columns in the data does not match the expected count.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at path: {filepath}")

    try:
        df = pd.read_csv(filepath, sep=" ", header=None)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse file: {e}")

    # Define expected column names (update as needed)
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
    """
    Plots a bar chart of correlation coefficients with respect to the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target : str
        The target column to compute correlations for.
    save_path : str
        File path to save the plot image.
    
    Raises
    ------
    KeyError
        If target column is not found in df.
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot plot correlations.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")

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
    """
    Plots the distribution of the 'Credit Standing' variable as a bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    save_path : str
        File path to save the plot image.
    
    Raises
    ------
    KeyError
        If 'Credit Standing' is not in the DataFrame.
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot plot credit standing distribution.")
    if 'Credit Standing' not in df.columns:
        raise KeyError("Column 'Credit Standing' not found in DataFrame")
        
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


def plot_feature_distributions(df: pd.DataFrame, features: list, target: str, save_path: str):
    """
    Plots histograms (with an overlaid kernel density estimate) for each feature in 'features',
    colored by a target variable, and saves the plot to the specified path.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    features : list
        List of feature column names to be plotted.
    target : str
        The column name for the target variable to use as hue.
    save_path : str
        The file path where the plot image will be saved.
    
    Raises
    ------
    ValueError
        If the DataFrame is empty.
    KeyError
        If the target column or any feature is missing.
    TypeError
        If any feature column is not numeric.
    
    Returns
    -------
    None
        The function saves the plot to 'save_path'.
    """
    # Input validation
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot generate plots on an empty DataFrame.")
    
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    
    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Feature column '{feature}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature column '{feature}' must be numeric to plot a histogram.")
    
    # Utility: Create subplots.
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    # Utility: Plot a single feature's distribution.
    def plot_single_feature(ax, feature):
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            kde=True,
            palette=['#3498db', '#e74c3c'],
            alpha=0.6,
            bins=30,
            ax=ax,
            hue_order=[0, 1]
        )
        ax.set_title(f"Distribution of {feature} by {target}", fontsize=14, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    
    # Decompose: Iterate through features and plot each.
    for i, feature in enumerate(features):
        plot_single_feature(axes[i], feature)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
