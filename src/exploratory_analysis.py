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

@click.command()
@click.option('--input', default="../data/processed/german_processed.csv", show_default=True, help="Path to processed input file")
@click.option('--output_dir', default="../results/eda", show_default=True, help="Directory to save EDA results")
def exploratory_analysis(input, output_dir):
    """Perform exploratory data analysis on the German Credit Data dataset."""
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the preprocessed data
        print(f"Loading preprocessed data from {input}...")
        df = pd.read_csv(input)
        
        # Set a more appealing visual style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        
        # Custom color palette
        custom_palette = ['#3498db', '#e74c3c']
        
        # 1. Correlation analysis
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        correlation_with_target = df[numerical_cols].corrwith(df['Credit Standing']).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        bars = sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index)
        
        plt.title('Feature Correlation with Credit Standing', fontsize=16, fontweight='bold')
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Color bars based on correlation value
        for i, bar in enumerate(bars.patches):
            if correlation_with_target.values[i] > 0:
                bar.set_facecolor('#e74c3c')  # Red for positive correlation
            else:
                bar.set_facecolor('#3498db')  # Blue for negative correlation
        
        # Add correlation values as text
        for i, v in enumerate(correlation_with_target.values):
            plt.text(v + 0.01 if v >= 0 else v - 0.06, i, f'{v:.2f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300)
        plt.close()
        
        # 2. Credit Standing Distribution
        plt.figure(figsize=(8, 8))
        target_counts = df['Credit Standing'].value_counts()
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette=custom_palette)
        plt.title('Credit Standing Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Credit Standing (0=Good, 1=Bad)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks([0, 1], ['Good (0)', 'Bad (1)'], fontsize=10)
        
        # Add percentage labels
        total = len(df)
        for i, p in enumerate(ax.patches):
            percentage = 100 * p.get_height() / total
            ax.annotate(f'{int(p.get_height())}\n({percentage:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'credit_standing_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Feature distributions
        top_features = ['Credit_Amount', 'Age', 'Duration (in months)']
        
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 4*len(top_features)))
        
        for i, feature in enumerate(top_features):
            # Create KDE plot with histograms
            sns.histplot(data=df, x=feature, hue='Credit Standing', kde=True, 
                         palette=custom_palette, alpha=0.6, bins=30, ax=axes[i],
                         hue_order=[0, 1])  # Explicitly set order: 0=Good, 1=Bad
            
            # Customize the plot
            axes[i].set_title(f'Distribution of {feature} by Credit Standing', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(feature, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            
            # Add mean lines with non-overlapping labels
            for j, (credit_status, color, label) in enumerate(zip([0, 1], custom_palette, ['Good Credit', 'Bad Credit'])):
                subset = df[df['Credit Standing'] == credit_status]
                mean_val = subset[feature].mean()
                
                axes[i].axvline(x=mean_val, color=color, linestyle='--', linewidth=2)
                
                # Position labels at different heights to avoid overlap
                y_pos = axes[i].get_ylim()[1] * (0.9 - j*0.15)  # Stagger vertically
                
                # Add background to text for better readability
                axes[i].text(mean_val, y_pos, 
                             f'{label} Mean: {mean_val:.2f}', 
                             color=color, fontweight='bold', ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.5'))
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
        
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