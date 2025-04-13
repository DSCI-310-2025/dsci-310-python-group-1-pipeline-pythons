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
├── data/
│   ├── raw/
│   │   └── raw_data.csv          # Input raw dataset
│   └── processed/
│       └── german_processed.csv   # Preprocessed dataset
├── output/
│   └── model_results.txt          # Analysis results
├── reports/
│   └── report.qmd                 # Quarto report source
├── src/
│   ├── functions/
│   │   ├── __init__.py            # Makes functions a package
│   │   ├── data_validation.py      # Data validation functions
│   │   └── model_utils.py          # Utility functions for loading data and visualizations
│   ├── exploratory_analysis.py     # Script for exploratory data analysis
│   ├── preprocess_data.py          # Script for preprocessing data
│   └── credit_risk_analysis.ipynb  # Jupyter Notebook for credit risk analysis
└── tests/
    ├── test_data_validation.py      # Tests for data validation functions
    └── test_project_utils.py        # Tests for utility functions
```

### Notes on Reproducibility
- Ensure that you have Docker installed on your machine to run the provided Docker setup.
- The project is designed to be reproducible using the Docker container, which encapsulates all dependencies and configurations.

### Additional Information
- For any issues or contributions, please refer to the project's GitHub repository.
- Ensure to check the `LICENSE.md` file for licensing information.