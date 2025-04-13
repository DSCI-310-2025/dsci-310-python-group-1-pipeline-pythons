#!/usr/bin/env python
"""
Downloads the German Credit Data dataset from UCI repository.

Usage:
    download_data.py --url=<url> --output=<output_file>

Options:
    --url=<url>                 URL to download the data from
    --output=<output_file>      Path to save the downloaded data
"""

import os
import click
import requests

@click.command()
@click.option('--url', default="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", show_default=True, help="URL to download data from")
@click.option('--output', default="../data/raw/raw_data.csv", show_default=True, help="Path to save the downloaded data")
def download_data(url, output):
    """Download the German Credit Data dataset and save it locally."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    try:
        # Download data
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save raw data
        with open(output, 'wb') as f:
            f.write(response.content)
        
        print(f"Data successfully downloaded and saved to {output}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        exit(1)

if __name__ == "__main__":
    download_data() 