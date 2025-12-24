"""
Inference module for skin cancer detection.
Provides prediction functionality for single images and batch processing.
"""

import os
import argparse
import numpy as np
from PIL import Image
from typing import Tuple, Dict
from tensorflow.keras.models import load_model

from config import (
    INDEX_TO_CLASS,
    LESION_CLASSES,
    DATASET_CONFIG
)


class SkinCancerPredictor:
    """Handles inference for skin cancer detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        self.image_size = DATASET_CONFIG['image_size']
        
        print("Model loaded successfully!")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Resize to model input size
        img = img.resize((self.image_size[1], self.image_size[0]))  # (width, height)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize (using approximate training statistics)
        # In production, these should be saved during training
        img_array = (img_array - 160.0) / 46.7
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(
        self, 
        image: np.ndarray, 
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict lesion class for input image.
        
        Args:
            image: Input image as numpy array
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class information
        class_code = INDEX_TO_CLASS[predicted_class_idx]
        class_name = LESION_CLASSES[class_code]
        
        result = {
            'class_code': class_code,
            'class_name': class_name,
            'class_index': int(predicted_class_idx),
            'confidence': confidence
        }
        
        if return_confidence:
            # Add all class probabilities
            all_probabilities = {}
            for idx, prob in enumerate(predictions[0]):
                code = INDEX_TO_CLASS[idx]
                name = LESION_CLASSES[code]
                all_probabilities[name] = float(prob)
            result['all_probabilities'] = all_probabilities
        
        return result
    
    def predict_from_file(self, image_path: str) -> Dict:
        """
        Predict lesion class from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Ensure RGB format
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        return self.predict(image_array)
    
    def predict_batch(self, images: list) -> list:
        """
        Predict lesion classes for multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def get_top_k_predictions(
        self, 
        image: np.ndarray, 
        k: int = 3
    ) -> list:
        """
        Get top K predictions with confidence scores.
        
        Args:
            image: Input image as numpy array
            k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence)
        """
        result = self.predict(image, return_confidence=True)
        probabilities = result['all_probabilities']
        
        # Sort by confidence
        sorted_predictions = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_predictions[:k]


def print_prediction(result: Dict):
    """Print prediction in formatted way."""
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Predicted Class: {result['class_name']}")
    print(f"Class Code: {result['class_code']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    
    if 'all_probabilities' in result:
        print("\n" + "-"*60)
        print("ALL CLASS PROBABILITIES")
        print("-"*60)
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, prob in sorted_probs:
            print(f"{class_name:40s}: {prob*100:6.2f}%")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Run inference on skin lesion images'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Show top K predictions'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SkinCancerPredictor(model_path=args.model)
    
    # Run prediction
    print(f"\nPredicting for image: {args.image}")
    result = predictor.predict_from_file(args.image)
    
    # Print results
    print_prediction(result)
    
    # Show top K predictions
    if args.top_k > 1:
        print(f"\n\nTop {args.top_k} Predictions:")
        print("-"*60)
        image = np.array(Image.open(args.image))
        top_k = predictor.get_top_k_predictions(image, k=args.top_k)
        for i, (class_name, confidence) in enumerate(top_k, 1):
            print(f"{i}. {class_name:40s}: {confidence*100:.2f}%")


if __name__ == '__main__':
    main()

