import os
import pandas as pd
import pytest
from src.functions.model_utils import (
    load_and_prepare_raw_data,
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

def test_load_and_prepare_raw_data_valid(tmp_path):
    # Create a dummy file with one row containing 21 space-separated tokens.
    content = (
        "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 "
        "A173 1 A191 A201 1"
    )
    file_path = tmp_path / "raw_data.txt"
    file_path.write_text(content)
    df = load_and_prepare_raw_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 21)
    # Verify that the column "Credit_Amount" (with underscore) exists.
    assert "Credit_Amount" in df.columns

def test_load_and_prepare_raw_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_prepare_raw_data("non_existent_file.txt")

def test_load_and_prepare_raw_data_bad_format(tmp_path):
    # Create a file with too few tokens.
    content = "A11 6 A34 A43"
    file_path = tmp_path / "bad_data.txt"
    file_path.write_text(content)
    with pytest.raises(ValueError):
        load_and_prepare_raw_data(str(file_path))

def test_plot_corr_barplot(mock_df, tmp_path):
    save_path = tmp_path / "corr_plot.png"
    plot_corr_barplot(mock_df, target='Credit Standing', save_path=str(save_path))
    assert os.path.exists(save_path)

def test_plot_credit_standing_distribution(mock_df, tmp_path):
    save_path = tmp_path / "credit_standing_distribution.png"
    plot_credit_standing_distribution(mock_df, save_path=str(save_path))
    assert os.path.exists(save_path)

def test_plot_feature_distributions(mock_df, tmp_path):
    save_path = tmp_path / "feature_distributions.png"
    features = ['Credit_Amount', 'Age', 'Duration (in months)']
    plot_feature_distributions(mock_df, features, target='Credit Standing', save_path=str(save_path))
    assert os.path.exists(save_path)

def test_plot_feature_distributions_missing_column(mock_df, tmp_path):
    # Test that the plotting function raises a KeyError if a required feature is missing.
    bad_features = ['Missing_Column']
    save_path = tmp_path / "bad_plot.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(mock_df,
            bad_features,
            target='Credit Standing',
            save_path=str(save_path)
            )

def test_plot_feature_distributions_missing_target(mock_df, tmp_path):
    """
    Test that the function raises a KeyError if the target column is not in the DataFrame.
    """
    save_path = tmp_path / "bad_target.png"
    with pytest.raises(KeyError):
        # Here, 'Nonexistent_Target' is not a column in mock_df.
        plot_feature_distributions(
            mock_df,
            ['Credit_Amount'],
            target='Nonexistent_Target',
            save_path=str(save_path)
        )

def test_plot_feature_distributions_empty_df(tmp_path):
    """
    Test that the function raises a ValueError when given an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    save_path = tmp_path / "empty_plot.png"
    with pytest.raises(ValueError):
        plot_feature_distributions(
            empty_df,
            ['Credit_Amount'],
            target='Credit Standing',
            save_path=str(save_path)
        )
