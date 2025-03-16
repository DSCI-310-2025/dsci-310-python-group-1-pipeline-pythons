# dsci-310-python-group-1-pipeline-pythons

### Project title: 
Credit Risk Prediction

### List of contributors/authors: 
Zhanerke Zhumash, Stallon Pinto, Ayush Joshi


### Summary of the project 
This project aims to develop a predictive model to assess the credit risk of loan applicants using the Statlog (German Credit Data) dataset. The primary objective is to determine whether an applicant is a good or bad credit risk based on various demographic and financial attributes. We explored a few models and landed on a Random Forest classifier as it had the highest accuracy and was also able to minimize false negatives, which is crucial for reducing the risk of approving applicants with poor creditworthiness.

___

### Instructions on how to run the data analysis
To run the data analysis, you can use the provided Docker setup. Follow these steps:

1. **Clone the github repo**:
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

This will start a Jupyter Notebook server accessible at `http://localhost:8888`. Make sure to replace <docker-username> with your dockerhub username.
___
### A list of the dependencies needed to run your analysis
The analysis requires the following Python packages, which are installed in the Docker image:
- pandas==2.2.3
- matplotlib==3.10.1
- seaborn==0.13.2
- scipy==1.11.3
- numpy==1.26.4
- scikit-learn==1.3.0
___
###  The names of the licenses contained in LICENSE.md:
- MIT License
- Creative Commons License 

## Usage of Makefile
To run the entire pipeline, use:
```bash sh
make 
```
This will:
1. Preprocess raw data
2. Perform exploratory analysis
3. Generate reports

### Cleaning Up
To remove generated files, run:
```sh
make clean
```

## File Structure
- `data/raw/raw_data.csv` - Input raw dataset
- `data/processed/german_processed.csv` - Preprocessed dataset
- `output/model_results.txt` - Analysis results
- `reports/report.qmd` - Quarto report source
