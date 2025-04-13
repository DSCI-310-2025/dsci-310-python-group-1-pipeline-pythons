# dsci-310-python-group-1-pipeline-pythons

### Project Title: 
Credit Risk Prediction

### List of Contributors/Authors: 
Zhanerke Zhumash, Stallon Pinto, Ayush Joshi

### Summary of the Project 
This project aims to develop a predictive model to assess the credit risk of loan applicants using the Statlog (German Credit Data) dataset. The primary objective is to determine whether an applicant is a good or bad credit risk based on various demographic and financial attributes. We explored a few models and landed on a Random Forest classifier as it had the highest accuracy and was also able to minimize false negatives, which is crucial for reducing the risk of approving applicants with poor creditworthiness.

___

### Instructions on How to Run the Data Analysis
To run the data analysis, you can use the provided Docker setup. Follow these steps:

1. **Clone the GitHub Repo**:
    ```bash
    git clone https://github.com/DSCI-310-2025/dsci-310-python-group-1-pipeline-pythons.git
    cd dsci-310-python-group-1-pipeline-pythons
    ```

2. **Build the Docker Image**:
   ```bash
   docker build -t <docker-username>/dsci310-project:latest .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -p 8888:8888 <docker-username>/dsci310-project:latest 
   ```

This will start a Jupyter Notebook server accessible at `http://localhost:8888`. Make sure to replace `<docker-username>` with your Docker Hub username.

___

### A List of the Dependencies Needed to Run Your Analysis
The analysis requires the following Python packages, which are installed in the Docker image:
- pandas==2.2.3
- matplotlib==3.10.1
- seaborn==0.13.2
- scipy==1.11.3
- numpy==1.26.4
- scikit-learn==1.3.0
- click==8.1.7
- requests==2.32.3
- pytest==8.3.5
- pyarrow (recommended for future compatibility with pandas)
- creditriskutilities==1.0.1

___

### The Names of the Licenses Contained in LICENSE.md:
- MIT License
- Creative Commons License 

## Usage of Makefile
To run the entire pipeline, use:
```bash
make 
```
This will:
1. Preprocess raw data
2. Perform exploratory analysis
3. Generate reports

### Cleaning Up
To remove generated files, run:
```bash
make clean
```

## Directory Structure
```
dsci-310-python-group-1-pipeline-pythons/
├── data/                             # Directory for datasets
│   ├── raw/                          # Directory for raw input data
│   │   └── raw_data.csv              # Input raw dataset
│   └── processed/                    # Directory for processed data
│       └── german_processed.csv       # Preprocessed dataset
├── results/                          # Directory for results and outputs
│   ├── model_results.txt              # Analysis results summary
│   ├── eda/                           # Directory for EDA visualizations
│   │   ├── credit_standing_distribution.png  # Visualization of credit standing distribution
│   │   ├── correlation_analysis.png    # Visualization of correlation analysis
│   │   └── feature_distributions.png   # Visualization of feature distributions
│   └── models/                        # Directory for model outputs
│       ├── baseline_confusion_matrix.png  # Confusion matrix for baseline model
│       ├── feature_importance.png      # Feature importance plot
│       ├── knn_confusion_matrix.png    # Confusion matrix for KNN model
│       ├── model_comparison.png         # Comparison of different models
│       └── randomforest_confusion_matrix.png  # Confusion matrix for Random Forest model
├── reports/                           # Directory for reports
│   └── credit_risk_analysis.qmd       # Quarto report source
├── src/                               # Directory for source code
│   ├── functions/                     # Directory for utility functions
│   │   ├── __init__.py                # Makes functions a package
│   │   ├── data_validation.py          # Data validation functions
│   │   └── model_utils.py              # Utility functions for loading data and visualizations
│   ├── exploratory_analysis.py         # Script for exploratory data analysis
│   ├── preprocess_data.py              # Script for preprocessing data
│   ├── model_training.py               # Script for training models
│   └── download_data.py                # Script for downloading data
└── tests/                             # Directory for test cases
    ├── conftest.py                     # Configuration for pytest
    ├── test_data_validation.py          # Tests for data validation functions
    └── test_project_utils.py            # Tests for utility functions
```

### Notes on Reproducibility
- Ensure that you have Docker installed on your machine to run the provided Docker setup.
- The project is designed to be reproducible using the Docker container, which encapsulates all dependencies and configurations.

### Additional Information
- For any issues or contributions, please refer to the project's GitHub repository.
- Ensure to check the `LICENSE.md` file for licensing information.

## Package Information
This project includes a Python package that has been published on TestPyPI. You can install it using the following command:

```
pip install --index-url https://test.pypi.org/simple/ creditriskutilities
```

To make this simpler, we have already added this into the dockerfile for ease in reproducibility
