"""
Model evaluation and testing module.
Provides comprehensive evaluation metrics and visualizations.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model

from config import INDEX_TO_CLASS, LESION_CLASSES, PATHS, ensure_directories
from data_loader import load_dataset


class ModelEvaluator:
    """Comprehensive model evaluation and metrics."""
    
    def __init__(self, model_path: str):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        ensure_directories()
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Create results directory
        self.results_dir = os.path.join(PATHS['results_dir'], self.model_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.class_names = [LESION_CLASSES[INDEX_TO_CLASS[i]] 
                           for i in range(len(INDEX_TO_CLASS))]
        
        print(f"Model loaded successfully!")
        print(f"Results will be saved to: {self.results_dir}")
    
    def load_test_data(self):
        """Load test dataset."""
        print("\nLoading test data...")
        _, _, _, _, self.X_test, self.y_test, _ = load_dataset()
        print(f"Test samples: {len(self.X_test)}")
    
    def predict(self):
        """Generate predictions on test set."""
        print("\nGenerating predictions...")
        self.y_pred_proba = self.model.predict(self.X_test, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        self.y_true = np.argmax(self.y_test, axis=1)
        print("Predictions complete!")
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        print("\nCalculating metrics...")
        
        metrics = {
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            'precision_macro': float(precision_score(self.y_true, self.y_pred, average='macro')),
            'precision_weighted': float(precision_score(self.y_true, self.y_pred, average='weighted')),
            'recall_macro': float(recall_score(self.y_true, self.y_pred, average='macro')),
            'recall_weighted': float(recall_score(self.y_true, self.y_pred, average='weighted')),
            'f1_macro': float(f1_score(self.y_true, self.y_pred, average='macro')),
            'f1_weighted': float(f1_score(self.y_true, self.y_pred, average='weighted')),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        # Save metrics
        metrics_path = os.path.join(self.results_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_path}")
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """Print metrics in readable format."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Precision (Macro): {metrics['precision_macro']*100:.2f}%")
        print(f"Recall (Macro): {metrics['recall_macro']*100:.2f}%")
        print(f"F1 Score (Macro): {metrics['f1_macro']*100:.2f}%")
        
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']*100:.2f}%")
            print(f"  Recall: {class_metrics['recall']*100:.2f}%")
            print(f"  F1 Score: {class_metrics['f1_score']*100:.2f}%")
    
    def generate_classification_report(self):
        """Generate and save detailed classification report."""
        print("\nGenerating classification report...")
        
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        report_path = os.path.join(self.results_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"Classification report saved to {report_path}")
        print("\n" + report)
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
    
    def plot_class_distribution(self):
        """Plot distribution of predictions vs true labels."""
        print("\nGenerating class distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # True labels distribution
        unique_true, counts_true = np.unique(self.y_true, return_counts=True)
        ax1.bar(range(len(unique_true)), counts_true, color='skyblue', edgecolor='black')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(self.y_pred, return_counts=True)
        ax2.bar(range(len(unique_pred)), counts_pred, color='lightcoral', edgecolor='black')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        dist_path = os.path.join(self.results_dir, 'class_distribution.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {dist_path}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*60)
        print("STARTING MODEL EVALUATION")
        print("="*60)
        
        self.load_test_data()
        self.predict()
        metrics = self.calculate_metrics()
        self.print_metrics(metrics)
        self.generate_classification_report()
        self.plot_confusion_matrix()
        self.plot_class_distribution()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"All results saved to: {self.results_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained skin cancer detection model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    evaluator = ModelEvaluator(model_path=args.model)
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

