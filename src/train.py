"""
Training script for skin cancer detection models.
Handles model training with proper logging, checkpointing, and metrics tracking.
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

from config import (
    TRAINING_CONFIG,
    CLR_CONFIG,
    PATHS,
    ensure_directories
)
from data_loader import load_dataset
from models import get_model, CyclicLR


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model_type: str = 'sequential'):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train ('sequential' or 'resnet')
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.data_loader = None
        
        ensure_directories()
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{model_type}_{timestamp}"
        self.run_dir = os.path.join(PATHS['logs_dir'], self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Training run: {self.run_name}")
    
    def load_data(self):
        """Load and prepare dataset."""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        (self.X_train, self.y_train, 
         self.X_val, self.y_val, 
         self.X_test, self.y_test, 
         self.data_loader) = load_dataset()
        
        print("\nDataset loaded successfully!")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
    
    def build_model(self):
        """Build and compile model."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        self.model = get_model(self.model_type)
        
        # Save model architecture
        model_json = self.model.to_json()
        json_path = os.path.join(self.run_dir, 'model_architecture.json')
        with open(json_path, 'w') as f:
            json.dump(json.loads(model_json), f, indent=2)
        
        # Save model summary
        summary_path = os.path.join(self.run_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"\nModel architecture saved to {json_path}")
    
    def get_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            PATHS['checkpoints_dir'], 
            f'{self.model_type}_best.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV logger
        csv_path = os.path.join(self.run_dir, 'training_log.csv')
        csv_logger = CSVLogger(csv_path)
        callbacks.append(csv_logger)
        
        # TensorBoard
        tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        if self.model_type == 'sequential':
            # Learning rate reduction for sequential model
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                verbose=1,
                factor=TRAINING_CONFIG['reduce_lr_factor'],
                min_lr=TRAINING_CONFIG['min_lr']
            )
            callbacks.append(reduce_lr)
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_CONFIG['early_stopping_patience'],
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        else:  # resnet with CLR
            steps_per_epoch = len(self.X_train) // TRAINING_CONFIG['batch_size']
            step_size = CLR_CONFIG['step_size_multiplier'] * steps_per_epoch
            
            clr = CyclicLR(
                base_lr=CLR_CONFIG['base_lr'],
                max_lr=CLR_CONFIG['max_lr'],
                step_size=step_size,
                mode=CLR_CONFIG['mode']
            )
            callbacks.append(clr)
        
        return callbacks
    
    def train(self, epochs: int = None, batch_size: int = None):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
        """
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        if batch_size is None:
            batch_size = TRAINING_CONFIG['batch_size']
        
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        callbacks = self.get_callbacks()
        
        # Get data augmentation generator
        datagen = self.data_loader.get_data_generator()
        datagen.fit(self.X_train)
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.X_train) // batch_size
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0
        )
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Save metrics
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'max_val_accuracy': float(max(self.history.history['val_accuracy'])),
            'max_train_accuracy': float(max(self.history.history['accuracy'])),
        }
        
        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_path}")
        
        return test_loss, test_accuracy
    
    def plot_history(self):
        """Plot and save training history."""
        print("\nPlotting training history...")
        
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(range(len(acc)), acc, label='Training Accuracy', linewidth=2)
        ax1.plot(range(len(val_acc)), val_acc, label='Validation Accuracy', linewidth=2)
        ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(range(len(loss)), loss, label='Training Loss', linewidth=2)
        ax2.plot(range(len(val_loss)), val_loss, label='Validation Loss', linewidth=2)
        ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.run_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")
    
    def save_final_model(self):
        """Save the final trained model."""
        model_path = os.path.join(PATHS['models_dir'], f'{self.model_type}_final.h5')
        self.model.save(model_path)
        print(f"\nFinal model saved to {model_path}")
    
    def run_full_training(self, epochs: int = None, batch_size: int = None):
        """
        Run complete training pipeline.
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
        """
        self.load_data()
        self.build_model()
        self.train(epochs, batch_size)
        self.evaluate()
        self.plot_history()
        self.save_final_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {self.run_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train skin cancer detection model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sequential',
        choices=['sequential', 'resnet'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    
    trainer = ModelTrainer(model_type=args.model)
    trainer.run_full_training(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

