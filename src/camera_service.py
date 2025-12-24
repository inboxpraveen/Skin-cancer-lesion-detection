"""
Real-time camera service for skin cancer detection.
Provides live camera feed with real-time predictions.
"""

import os
import argparse
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from config import CAMERA_CONFIG
from inference import SkinCancerPredictor


class CameraService:
    """Real-time camera-based skin cancer detection service."""
    
    def __init__(self, model_path: str, camera_index: int = 0):
        """
        Initialize camera service.
        
        Args:
            model_path: Path to trained model
            camera_index: Camera device index (0 for default camera)
        """
        print("Initializing Camera Service...")
        
        # Initialize predictor
        self.predictor = SkinCancerPredictor(model_path)
        
        # Initialize camera
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        
        # Configuration
        self.confidence_threshold = CAMERA_CONFIG['confidence_threshold']
        self.window_name = CAMERA_CONFIG['window_name']
        
        # Create screenshots directory
        self.screenshots_dir = 'screenshots'
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        print("Camera Service initialized!")
    
    def start_camera(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
        
        print(f"Camera {self.camera_index} started successfully!")
        return True
    
    def stop_camera(self):
        """Stop camera capture and release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")
    
    def draw_prediction(
        self, 
        frame: np.ndarray, 
        prediction: dict,
        bbox: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Draw prediction information on frame.
        
        Args:
            frame: Video frame
            prediction: Prediction dictionary from predictor
            bbox: Optional bounding box (x, y, w, h) for ROI
            
        Returns:
            Frame with prediction overlay
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_height = 150
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Extract prediction info
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        # Color based on confidence
        if confidence >= 0.8:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw main prediction
        text = f"Prediction: {class_name}"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        conf_text = f"Confidence: {confidence*100:.2f}%"
        cv2.putText(frame, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw top 3 predictions
        if 'all_probabilities' in prediction:
            sorted_probs = sorted(
                prediction['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            y_offset = 90
            cv2.putText(frame, "Top Predictions:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (name, prob) in enumerate(sorted_probs):
                y_offset += 20
                short_name = name[:25] + "..." if len(name) > 25 else name
                prob_text = f"{i+1}. {short_name}: {prob*100:.1f}%"
                cv2.putText(frame, prob_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw bounding box if provided
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save screenshot",
            "Press 'r' to toggle ROI"
        ]
        y_pos = height - 70
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
        
        return frame
    
    def save_screenshot(self, frame: np.ndarray, prediction: dict):
        """
        Save screenshot with prediction.
        
        Args:
            frame: Video frame to save
            prediction: Prediction information
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        class_code = prediction['class_code']
        confidence = prediction['confidence']
        
        filename = f"{timestamp}_{class_code}_{confidence:.2f}.jpg"
        filepath = os.path.join(self.screenshots_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Screenshot saved: {filepath}")
    
    def run(self):
        """
        Run the camera service with real-time predictions.
        """
        if not self.start_camera():
            return
        
        self.running = True
        print("\n" + "="*60)
        print("CAMERA SERVICE RUNNING")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("="*60 + "\n")
        
        frame_count = 0
        predict_every_n_frames = 5  # Predict every N frames for performance
        last_prediction = None
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Cannot read frame from camera")
                    break
                
                # Make prediction every N frames
                if frame_count % predict_every_n_frames == 0:
                    # Convert BGR to RGB for prediction
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        last_prediction = self.predictor.predict(frame_rgb)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        last_prediction = None
                
                # Draw prediction if available
                if last_prediction is not None:
                    frame = self.draw_prediction(frame, last_prediction)
                else:
                    # Show "Processing..." message
                    cv2.putText(frame, "Processing...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    self.running = False
                elif key == ord('s') and last_prediction is not None:
                    self.save_screenshot(frame, last_prediction)
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.stop_camera()


class APIService:
    """Flask API service for skin cancer detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize API service.
        
        Args:
            model_path: Path to trained model
        """
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        self.predictor = SkinCancerPredictor(model_path)
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Register routes
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/predict', methods=['POST'])(self.predict_endpoint)
        self.app.route('/predict_base64', methods=['POST'])(self.predict_base64_endpoint)
    
    def health_check(self):
        """Health check endpoint."""
        from flask import jsonify
        return jsonify({'status': 'healthy', 'service': 'skin_cancer_detection'})
    
    def predict_endpoint(self):
        """Prediction endpoint for uploaded images."""
        from flask import request, jsonify
        import io
        from PIL import Image
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        try:
            # Read and process image
            image = Image.open(io.BytesIO(file.read()))
            image_array = np.array(image)
            
            # Ensure RGB
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Get prediction
            result = self.predictor.predict(image_array)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def predict_base64_endpoint(self):
        """Prediction endpoint for base64 encoded images."""
        from flask import request, jsonify
        import base64
        import io
        from PIL import Image
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Ensure RGB
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Get prediction
            result = self.predictor.predict(image_array)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the API service."""
        print("\n" + "="*60)
        print("STARTING API SERVICE")
        print("="*60)
        print(f"Host: {host}")
        print(f"Port: {port}")
        print("="*60 + "\n")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Skin Cancer Detection Camera Service'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='camera',
        choices=['camera', 'api'],
        help='Service mode: camera or api'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API host (for api mode)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='API port (for api mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'camera':
        service = CameraService(model_path=args.model, camera_index=args.camera)
        service.run()
    else:
        service = APIService(model_path=args.model)
        service.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()

