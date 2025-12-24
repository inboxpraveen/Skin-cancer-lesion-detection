"""
Configuration module for skin cancer detection system.
Contains all hyperparameters, paths, and model configurations.
"""

import os
from typing import Dict, Tuple

# Dataset configuration
DATASET_CONFIG = {
    'metadata_file': 'data/HAM10000_metadata.csv',
    'images_dir': 'data/images',
    'image_size': (90, 120),  # (height, width)
    'channels': 3,
    'num_classes': 7,
}

# Class mappings
LESION_CLASSES = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

CLASS_TO_INDEX = {
    'Actinic keratoses': 0,
    'Basal cell carcinoma': 1,
    'Benign keratosis-like lesions': 2,
    'Dermatofibroma': 3,
    'Melanoma': 4,
    'Melanocytic nevi': 5,
    'Vascular lesions': 6
}

INDEX_TO_CLASS = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 30,
    'validation_split': 0.15,
    'test_split': 0.01,
    'learning_rate': 0.001,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 4,
    'reduce_lr_factor': 0.0001,
    'min_lr': 0.000001,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': False,
}

# Model configuration
MODEL_CONFIG = {
    'sequential': {
        'name': 'sequential_model',
        'dropout_rate': 0.20,
        'initial_filters': 32,
    },
    'resnet': {
        'name': 'resnet_model',
        'initial_filters': 16,
        'l2_regularization': 0.001,
        'dropout_rate': None,
        'blocks': 4,
        'layers_per_block': 2,
    }
}

# CLR configuration for ResNet
CLR_CONFIG = {
    'base_lr': 0.001,
    'max_lr': 0.1,
    'step_size_multiplier': 2,  # Multiplied by steps_per_epoch
    'mode': 'triangular',
}

# SGD configuration for ResNet
SGD_CONFIG = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'nesterov': True,
}

# Paths
PATHS = {
    'models_dir': 'models',
    'logs_dir': 'logs',
    'results_dir': 'results',
    'checkpoints_dir': 'models/checkpoints',
}

# Camera service configuration
CAMERA_CONFIG = {
    'camera_index': 0,
    'window_name': 'Skin Cancer Detection',
    'confidence_threshold': 0.5,
    'frame_width': 640,
    'frame_height': 480,
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'model_path': 'models/best_model.h5',
}


def get_image_shape() -> Tuple[int, int, int]:
    """Returns the input image shape for the model."""
    return (
        DATASET_CONFIG['image_size'][0],
        DATASET_CONFIG['image_size'][1],
        DATASET_CONFIG['channels']
    )


def ensure_directories():
    """Creates necessary directories if they don't exist."""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    os.makedirs(DATASET_CONFIG['images_dir'], exist_ok=True)

