import os
import pandas as pd
import pytest
from src.functions.model_utils import (
    load_and_prepare_raw_data,
    plot_corr_barplot,
    plot_credit_standing_distribution,
    plot_feature_distributions
)

# --- Fixtures ---

@pytest.fixture
def mock_df():
    """Returns a minimal valid DataFrame for plotting tests."""
    return pd.DataFrame({
        'Credit_Amount': [1000, 2000],
        'Age': [30, 40],
        'Duration (in months)': [12, 24],
        'Credit Standing': [0, 1]
    })

# --- Tests for load_and_prepare_raw_data ---

def test_load_and_prepare_raw_data_valid(tmp_path):
    """Test that valid raw data is correctly loaded and column names are assigned."""
    content = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A191 A201 1"
    file_path = tmp_path / "data.txt"
    file_path.write_text(content)
    df = load_and_prepare_raw_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 21)
    assert "Credit_Amount" in df.columns

def test_load_and_prepare_raw_data_missing_file():
    """Test that a FileNotFoundError is raised when the file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_and_prepare_raw_data("non_existent.csv")

def test_load_and_prepare_raw_data_wrong_format(tmp_path):
    """Test that ValueError is raised when column count doesn't match expected."""
    content = "A11 6 A34"
    path = tmp_path / "bad.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_and_prepare_raw_data(str(path))

# --- Tests for plot_corr_barplot ---

def test_plot_corr_barplot_valid(mock_df, tmp_path):
    """Test correlation bar plot is saved successfully with valid input."""
    path = tmp_path / "corr_plot.png"
    plot_corr_barplot(mock_df, 'Credit Standing', str(path))
    assert os.path.exists(path)

def test_plot_corr_barplot_missing_target(mock_df, tmp_path):
    """Test that KeyError is raised if the target column is missing."""
    df = mock_df.drop(columns=['Credit Standing'])
    path = tmp_path / "missing.png"
    with pytest.raises(KeyError):
        plot_corr_barplot(df, 'Credit Standing', str(path))

# --- Tests for plot_credit_standing_distribution ---

def test_plot_credit_distribution_valid(mock_df, tmp_path):
    """Test credit standing distribution plot is saved correctly."""
    path = tmp_path / "dist.png"
    plot_credit_standing_distribution(mock_df, str(path))
    assert os.path.exists(path)

def test_plot_credit_distribution_missing_column(tmp_path):
    """Test that KeyError is raised if 'Credit Standing' is missing."""
    df = pd.DataFrame({'SomeCol': [1, 2]})
    path = tmp_path / "missing.png"
    with pytest.raises(KeyError):
        plot_credit_standing_distribution(df, str(path))

# --- Tests for plot_feature_distributions ---

def test_plot_feature_distributions_valid(mock_df, tmp_path):
    """Test feature distributions are plotted and saved for numeric features."""
    path = tmp_path / "features.png"
    features = ['Credit_Amount', 'Age']
    plot_feature_distributions(mock_df, features, 'Credit Standing', str(path))
    assert os.path.exists(path)

def test_plot_feature_distributions_missing_feature(mock_df, tmp_path):
    """Test KeyError is raised if a listed feature is missing in the DataFrame."""
    path = tmp_path / "bad_feature.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(mock_df, ['Missing_Col'], 'Credit Standing', str(path))

def test_plot_feature_distributions_missing_target(mock_df, tmp_path):
    """Test KeyError is raised if the target column is not in the DataFrame."""
    df = mock_df.drop(columns=['Credit Standing'])
    path = tmp_path / "bad_target.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))

def test_plot_feature_distributions_empty_df(tmp_path):
    """Test ValueError is raised for empty DataFrame input."""
    df = pd.DataFrame()
    path = tmp_path / "empty.png"
    with pytest.raises(ValueError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))

def test_plot_feature_distributions_non_numeric(tmp_path):
    """Test TypeError is raised for non-numeric feature columns."""
    df = pd.DataFrame({
        'Credit_Amount': ['low', 'medium'],
        'Age': [30, 40],
        'Credit Standing': [0, 1]
    })
    path = tmp_path / "non_numeric.png"
    with pytest.raises(TypeError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))
