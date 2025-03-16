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
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
        
        # Function to evaluate and visualize model performance
        def evaluate_model(model, X_test, y_test, model_name, scaled=False):
            # Use scaled data if required
            X_test_eval = X_test_scaled if scaled else X_test
            
            # Make predictions
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Print metrics
            print(f"\n{model_name} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Save classification report
            report = classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_classification_report.csv'))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Good Credit', 'Bad Credit'],
                        yticklabels=['Good Credit', 'Bad Credit'])
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'), dpi=300)
            plt.close()
            
            return {
                'model_name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        
        # 1. Baseline Model (Majority Class Classifier)
        print("Training baseline model...")
        baseline_model = DummyClassifier(strategy='most_frequent')
        baseline_model.fit(X_train, y_train)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline")
        
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
        knn_metrics = evaluate_model(best_knn, X_test, y_test, "KNN", scaled=True)
        
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
        rf_metrics = evaluate_model(best_rf, X_test, y_test, "RandomForest")
        
        # Save the Random Forest model
        with open(os.path.join(output_dir, 'random_forest_model.pkl'), 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Feature importance for Random Forest
        if hasattr(best_rf, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Save feature importance
            feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title('Top 15 Features by Importance (Random Forest)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
        
        # Compare model performance
        models_comparison = pd.DataFrame([baseline_metrics, knn_metrics, rf_metrics])
        models_comparison = models_comparison.set_index('model_name')
        
        # Save model comparison
        models_comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
        
        # Plot model comparison
        plt.figure(figsize=(12, 6))
        models_comparison[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', colormap='viridis')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
        
        print(f"Model training and evaluation completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error training models: {e}")
        exit(1)

if __name__ == "__main__":
    train_models() 