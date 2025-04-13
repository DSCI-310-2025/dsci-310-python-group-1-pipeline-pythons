# author: Ayush Joshi, Stallon Pinto, & Zhanerke Zhumash
# date: 2025-03-15

SHELL := /bin/bash
.ONESHELL:

# Use the conda python environment if available, otherwise use python3.
PYTHON := $(shell command -v conda >/dev/null 2>&1 && echo "conda run -n base python" || echo "python3")
SRC_DIR := src
DATA_DIR := data
OUTPUT_DIR := results

# Define source files and output files
RAW_DATA := $(DATA_DIR)/raw/raw_data.csv
PROCESSED_DATA := $(DATA_DIR)/processed/german_processed.csv
MODEL_OUTPUT := $(OUTPUT_DIR)/model_results.txt
REPORT_HTML := reports/analysis_report.html
REPORT_PDF := reports/analysis_report.pdf

.PHONY: all clean

# Run the entire analysis pipeline
all: $(MODEL_OUTPUT) $(REPORT_HTML) $(REPORT_PDF)

# Preprocess data
$(PROCESSED_DATA): $(RAW_DATA) $(SRC_DIR)/preprocess_data.py
	@echo "Preprocessing data..."
		set -e
		$(PYTHON) $(SRC_DIR)/preprocess_data.py --input $(RAW_DATA) --output $(PROCESSED_DATA)

# Run analysis
$(MODEL_OUTPUT): $(PROCESSED_DATA) $(SRC_DIR)/exploratory_analysis.py $(SRC_DIR)/model_training.py
	@echo "Running analysis..."
		mkdir -p $(OUTPUT_DIR)/eda $(OUTPUT_DIR)/models
		set -e
		$(PYTHON) $(SRC_DIR)/exploratory_analysis.py --input $(PROCESSED_DATA) --output_dir $(OUTPUT_DIR)/eda
		$(PYTHON) $(SRC_DIR)/model_training.py --input $(PROCESSED_DATA) --output_dir $(OUTPUT_DIR)/models
		touch $(MODEL_OUTPUT)

# Generate HTML report
$(REPORT_HTML): $(MODEL_OUTPUT) reports/credit_risk_analysis.qmd
	@echo "Generating HTML report..."
		set -e
		quarto render reports/credit_risk_analysis.qmd --to html

# Generate PDF report
$(REPORT_PDF): $(MODEL_OUTPUT) reports/credit_risk_analysis.qmd
	@echo "Generating PDF report..."
		set -e
		quarto render reports/credit_risk_analysis.qmd --to pdf

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
		rm -rf $(PROCESSED_DATA) $(MODEL_OUTPUT) $(REPORT_HTML) $(REPORT_PDF) reports/report_files
