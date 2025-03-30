#!/usr/bin/env python
"""
Performs exploratory data analysis on the German Credit Data dataset.

Usage:
    exploratory_analysis.py --input=<input_file> --output_dir=<output_dir>

Options:
    --input=<input_file>        Path to the preprocessed data file
    --output_dir=<output_dir>   Directory to save the EDA visualizations
"""

import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization import (
    create_output_dir,
    plot_corr_barplot,
    plot_credit_standing_distribution,
    plot_feature_distributions
)

@click.command()
@click.option('--input', default="../data/processed/german_processed.csv", show_default=True, help="Path to processed input file")
@click.option('--output_dir', default="../results/eda", show_default=True, help="Directory to save EDA results")
def exploratory_analysis(input, output_dir):
    """Perform exploratory data analysis on the German Credit Data dataset."""
    
    # Create directory if it doesn't exist
    create_output_dir(output_dir)
    
    try:
        # Load the preprocessed data
        print(f"Loading preprocessed data from {input}...")
        df = pd.read_csv(input)
        
        # Set a more appealing visual style
        sns.set_style("whitegrid")
        # abstracted the seaborn styling into dictionary
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'axes.edgecolor': '#333333',
            'axes.linewidth': 0.8,
            'xtick.color': '#333333',
            'ytick.color': '#333333'
        })
        # Custom color palette
        custom_palette = ['#3498db', '#e74c3c']
        
        # 1. Correlation analysis
        plot_corr_barplot(df, target='Credit Standing',
                          save_path=os.path.join(output_dir, 'correlation_analysis.png'))

        # 2. Credit Standing Distribution
        plot_credit_standing_distribution(df,
                                          save_path=os.path.join(output_dir, 'credit_standing_distribution.png'))

        
        # 3. Feature distributions
        top_features = ['Credit_Amount', 'Age', 'Duration (in months)']
        
        plot_feature_distributions(df, top_features, target='Credit Standing',
                                   save_path=os.path.join(output_dir, 'feature_distributions.png'))
        
        # Save summary statistics
        summary_stats = df.describe().T
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
        
        # Calculate and save group statistics
        group_stats = df.groupby('Credit Standing')[top_features].agg(['mean', 'median', 'std'])
        group_stats.to_csv(os.path.join(output_dir, 'group_statistics.csv'))
        
        print(f"Exploratory data analysis completed successfully. Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error performing exploratory analysis: {e}")
        exit(1)

if __name__ == "__main__":
    exploratory_analysis() 