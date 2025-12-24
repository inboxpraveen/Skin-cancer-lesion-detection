"""
Data loading and preprocessing module.
Handles dataset loading, preprocessing, and data augmentation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATASET_CONFIG, 
    LESION_CLASSES, 
    TRAINING_CONFIG,
    AUGMENTATION_CONFIG
)


class DataLoader:
    """Handles loading and preprocessing of skin lesion dataset."""
    
    def __init__(self, metadata_path: str, images_dir: str):
        """
        Initialize the data loader.
        
        Args:
            metadata_path: Path to the metadata CSV file
            images_dir: Directory containing the images
        """
        self.metadata_path = metadata_path
        self.images_dir = images_dir
        self.image_size = DATASET_CONFIG['image_size']
        self.num_classes = DATASET_CONFIG['num_classes']
        
        self.metadata_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata."""
        print("Loading metadata...")
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Handle missing values in age
        age_mean = self.metadata_df['age'].mean()
        self.metadata_df['age'].fillna(age_mean, inplace=True)
        
        # Create cell type mapping
        self.metadata_df['cell_type'] = self.metadata_df['dx'].map(LESION_CLASSES)
        self.metadata_df['cell_type_idx'] = pd.Categorical(
            self.metadata_df['cell_type']
        ).codes
        
        # Map image paths
        image_paths = self._get_image_paths()
        self.metadata_df['path'] = self.metadata_df['image_id'].map(image_paths.get)
        
        print(f"Loaded {len(self.metadata_df)} samples")
        return self.metadata_df
    
    def _get_image_paths(self) -> dict:
        """Create a mapping of image IDs to file paths."""
        image_paths = {}
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_id = os.path.splitext(file)[0]
                    image_paths[image_id] = os.path.join(root, file)
        return image_paths
    
    def load_images(self) -> None:
        """Load and preprocess all images."""
        print("Loading images...")
        images = []
        for path in self.metadata_df['path']:
            try:
                img = Image.open(path).resize(
                    (self.image_size[1], self.image_size[0])  # (width, height)
                )
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Add a blank image as placeholder
                images.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
        
        self.metadata_df['image'] = images
        print(f"Loaded {len(images)} images")
    
    def prepare_data(self) -> Tuple:
        """
        Prepare train, validation, and test datasets.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        if 'image' not in self.metadata_df.columns:
            self.load_images()
        
        print("Preparing datasets...")
        
        # Extract features and targets
        X = np.array(self.metadata_df['image'].tolist())
        y = self.metadata_df['cell_type_idx'].values
        
        # Split into train+val and test
        test_split = TRAINING_CONFIG['test_split']
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_split, 
            random_state=42,
            stratify=y
        )
        
        # Split train into train and validation
        val_split = TRAINING_CONFIG['validation_split']
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_split,
            random_state=42,
            stratify=y_train_val
        )
        
        # Normalize images
        train_mean = X_train.mean()
        train_std = X_train.std()
        
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_val = to_categorical(y_val, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Store normalization parameters
        self.train_mean = train_mean
        self.train_std = train_std
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_data_generator(self) -> ImageDataGenerator:
        """
        Create a data augmentation generator.
        
        Returns:
            Configured ImageDataGenerator
        """
        return ImageDataGenerator(**AUGMENTATION_CONFIG)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Resize image
        img = Image.fromarray(image)
        img = img.resize((self.image_size[1], self.image_size[0]))
        img_array = np.array(img)
        
        # Normalize using training statistics
        if hasattr(self, 'train_mean') and hasattr(self, 'train_std'):
            img_array = (img_array - self.train_mean) / self.train_std
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_class_distribution(self) -> pd.Series:
        """Get the distribution of classes in the dataset."""
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded.")
        return self.metadata_df['dx'].value_counts()


def load_dataset() -> Tuple:
    """
    Convenience function to load the complete dataset.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, data_loader)
    """
    data_loader = DataLoader(
        metadata_path=DATASET_CONFIG['metadata_file'],
        images_dir=DATASET_CONFIG['images_dir']
    )
    
    data_loader.load_metadata()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, data_loader

