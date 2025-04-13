#!/usr/bin/env python
"""
Trains and evaluates machine learning models for credit risk prediction.

Usage:
    model_training.py --input=<input_file> --output_dir=<output_dir>
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

# Import evaluation utilities
from creditriskutilities import evaluate_model, plot_feature_importance, compare_models

@click.command()
@click.option('--input', default="../data/processed/german_processed.csv", show_default=True, help="Path to processed input file")
@click.option('--output_dir', default="../results/models", show_default=True, help="Directory to save model results")
def train_models(input, output_dir):
    """Train and evaluate machine learning models for credit risk prediction."""

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading preprocessed data from {input}...")
    df = pd.read_csv(input)

    # --- Step 1: Split Data ---
    X = df.drop('Credit Standing', axis=1)
    y = df['Credit Standing']

    # Ensure test set is unseen — split before any transformations
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Step 2: Scale Features for KNN (NO LEAKAGE: fit only on training data) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # --- Step 3: Baseline Model ---
    print("Training baseline model...")
    try:
        baseline_model = DummyClassifier(strategy='most_frequent')
        baseline_model.fit(X_train, y_train)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline", output_dir)

        with open(os.path.join(output_dir, 'baseline_model.pkl'), 'wb') as f:
            pickle.dump(baseline_model, f)

    except Exception as e:
        print(f"Error in baseline model: {e}")

    # --- Step 4: KNN with Grid Search ---
    print("Training KNN model with grid search...")
    try:
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        knn = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn, knn_param_grid, cv=5, scoring='f1', n_jobs=-1)
        knn_grid.fit(X_train_scaled, y_train)

        print(f"\nBest KNN Parameters: {knn_grid.best_params_}")
        best_knn = knn_grid.best_estimator_
        knn_metrics = evaluate_model(best_knn, X_test, y_test, "KNN", output_dir, X_test_scaled=X_test_scaled)

        with open(os.path.join(output_dir, 'knn_model.pkl'), 'wb') as f:
            pickle.dump(best_knn, f)

    except Exception as e:
        print(f"Error in KNN training: {e}")

    # --- Step 5: Random Forest with Grid Search ---
    print("Training Random Forest model with grid search...")
    try:
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 10],
            'class_weight': [None, 'balanced']
        }

        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train, y_train)

        print(f"\nBest Random Forest Parameters: {rf_grid.best_params_}")
        best_rf = rf_grid.best_estimator_
        rf_metrics = evaluate_model(best_rf, X_test, y_test, "RandomForest", output_dir)

        with open(os.path.join(output_dir, 'random_forest_model.pkl'), 'wb') as f:
            pickle.dump(best_rf, f)

        if hasattr(best_rf, 'feature_importances_'):
            plot_feature_importance(best_rf, X.columns, output_dir)

    except Exception as e:
        print(f"Error in Random Forest training: {e}")

    # --- Step 6: Compare Models ---
    try:
        compare_models([baseline_metrics, knn_metrics, rf_metrics], output_dir)
        print(f"\n✅ Model training complete. Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error comparing models: {e}")

if __name__ == "__main__":
    train_models()
