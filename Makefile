# author: Jordan Bourak & Tiffany Timbers
# date: 2021-11-22

# Define variables
PYTHON := python3
SRC_DIR := src
DATA_DIR := data
OUTPUT_DIR := output

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
	$(PYTHON) $(SRC_DIR)/preprocess.py --input $(RAW_DATA) --output $(PROCESSED_DATA)

# Run analysis
$(MODEL_OUTPUT): $(PROCESSED_DATA) $(SRC_DIR)/exploratory_analysis.py
	@echo "Running analysis..."
	$(PYTHON) $(SRC_DIR)/exploratory_analysis.py --input $(PROCESSED_DATA) --output $(MODEL_OUTPUT)

# Generate reports
$(REPORT_HTML): $(MODEL_OUTPUT) reports/report.qmd
	@echo "Generating HTML report..."
	quarto render reports/report.qmd --to html

$(REPORT_PDF): $(MODEL_OUTPUT) reports/report.qmd
	@echo "Generating PDF report..."
	quarto render reports/report.qmd --to pdf

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(PROCESSED_DATA) $(MODEL_OUTPUT) $(REPORT_HTML) $(REPORT_PDF) reports/report_files
