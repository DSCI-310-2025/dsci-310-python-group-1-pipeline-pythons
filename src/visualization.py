import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def plot_feature_distributions(df: pd.DataFrame, features: list, target: str, save_path: str):
    custom_palette = ['#3498db', '#e74c3c']
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4*len(features)))

    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue=target, kde=True,
                     palette=custom_palette, alpha=0.6, bins=30,
                     ax=axes[i], hue_order=[0, 1])
        axes[i].set_title(f'Distribution of {feature} by {target}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

        for j, (val, color, label) in enumerate(zip([0, 1], custom_palette, ['Good Credit', 'Bad Credit'])):
            mean_val = df[df[target] == val][feature].mean()
            axes[i].axvline(x=mean_val, color=color, linestyle='--', linewidth=2)
            y = axes[i].get_ylim()[1] * (0.9 - j*0.15)
            axes[i].text(mean_val, y, f'{label} Mean: {mean_val:.2f}',
                         ha='center', va='top', color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
