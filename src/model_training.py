#!/usr/bin/env python
"""
Trains and evaluates machine learning models for credit risk prediction.

Usage:
    model_training.py --input=<input_file> --output_dir=<output_dir>

Options:
    --input=<input_file>        Path to the preprocessed data file
    --output_dir=<output_dir>   Directory to save the model results
"""

import os
import click
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import utility functions
from model_utils import evaluate_model, plot_feature_importance, compare_models

@click.command()
@click.option('--input', default="../data/processed/german_processed.csv", show_default=True, help="Path to processed input file")
@click.option('--output_dir', default="../results/models", show_default=True, help="Directory to save model results")
def train_models(input, output_dir):
    """Train and evaluate machine learning models for credit risk prediction."""
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the preprocessed data
        print(f"Loading preprocessed data from {input}...")
        df = pd.read_csv(input)
        
        # Separate features and target
        X = df.drop('Credit Standing', axis=1)
        y = df['Credit Standing']
        
        # Create train-test split (75% train, 25% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Scale features for KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for future use
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # 1. Baseline Model (Majority Class Classifier)
        print("Training baseline model...")
        baseline_model = DummyClassifier(strategy='most_frequent')
        baseline_model.fit(X_train, y_train)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline", output_dir)
        
        # Save the baseline model
        with open(os.path.join(output_dir, 'baseline_model.pkl'), 'wb') as f:
            pickle.dump(baseline_model, f)
        
        # 2. KNN Classifier with Grid Search
        print("Training KNN model with grid search...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest KNN parameters: {grid_search.best_params_}")
        best_knn = grid_search.best_estimator_
        knn_metrics = evaluate_model(best_knn, X_test, y_test, "KNN", output_dir, X_test_scaled=X_test_scaled)
        
        # Save the KNN model
        with open(os.path.join(output_dir, 'knn_model.pkl'), 'wb') as f:
            pickle.dump(best_knn, f)
        
        # 3. Random Forest Classifier with Grid Search
        print("Training Random Forest model with grid search...")
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        
        print(f"\nBest Random Forest parameters: {grid_search_rf.best_params_}")
        best_rf = grid_search_rf.best_estimator_
        rf_metrics = evaluate_model(best_rf, X_test, y_test, "RandomForest", output_dir)
        
        # Save the Random Forest model
        with open(os.path.join(output_dir, 'random_forest_model.pkl'), 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Feature importance for Random Forest
        if hasattr(best_rf, 'feature_importances_'):
            plot_feature_importance(best_rf, X.columns, output_dir)
        
        # Compare model performance
        compare_models([baseline_metrics, knn_metrics, rf_metrics], output_dir)
        
        print(f"Model training and evaluation completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error training models: {e}")
        exit(1)

if __name__ == "__main__":
    train_models()