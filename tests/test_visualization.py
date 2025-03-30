import os
import pandas as pd
import pytest

from src.visualization import (
    create_output_dir,
    plot_corr_barplot,
    plot_credit_standing_distribution,
    plot_feature_distributions
)


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'Credit_Amount': [1000, 2000, 1500, 3000],
        'Age': [25, 45, 35, 50],
        'Duration (in months)': [12, 24, 18, 30],
        'Credit Standing': [0, 1, 0, 1]
    })

def test_create_output_dir(tmp_path):
    test_path = tmp_path / "subdir"
    create_output_dir(str(test_path))
    assert test_path.exists()

def test_plot_corr_barplot(mock_df, tmp_path):
    save_path = tmp_path / "corr_plot.png"
    plot_corr_barplot(mock_df, target='Credit Standing', save_path=str(save_path))
    assert save_path.exists()

def test_plot_credit_standing_distribution(mock_df, tmp_path):
    save_path = tmp_path / "credit_standing_distribution.png"
    plot_credit_standing_distribution(mock_df, save_path=str(save_path))
    assert save_path.exists()

def test_plot_feature_distributions(mock_df, tmp_path):
    save_path = tmp_path / "feature_distributions.png"
    features = ['Credit_Amount', 'Age', 'Duration (in months)']
    plot_feature_distributions(mock_df, features, target='Credit Standing', save_path=str(save_path))
    assert save_path.exists()

def test_plot_feature_distributions_missing_column(mock_df, tmp_path):
    # Test that it fails gracefully if a feature column is missing
    bad_features = ['Missing_Column']
    save_path = tmp_path / "bad_plot.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(mock_df, bad_features, target='Credit Standing', save_path=str(save_path))
