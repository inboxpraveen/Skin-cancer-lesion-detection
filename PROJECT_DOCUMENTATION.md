# Project Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module Documentation](#module-documentation)
3. [Data Pipeline](#data-pipeline)
4. [Model Details](#model-details)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Deployment Options](#deployment-options)
8. [API Specification](#api-specification)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## System Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (CLI Tools, Camera Service, REST API)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Application Layer                          │
│  (Training, Evaluation, Inference Logic)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                     Core Layer                               │
│  (Models, Data Processing, Configuration)                   │
└─────────────────────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  Infrastructure Layer                        │
│  (TensorFlow/Keras, OpenCV, Flask)                          │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Modularity**: Each component has a single, well-defined responsibility
- **Configurability**: All hyperparameters and settings are centralized in config.py
- **Extensibility**: New models and features can be added with minimal changes
- **Reproducibility**: Training runs are logged with timestamps and configurations
- **Production-Ready**: Includes error handling, logging, and proper resource management

## Module Documentation

### config.py

Central configuration module containing all system parameters.

**Key Components:**

- `DATASET_CONFIG`: Image size, paths, number of classes
- `TRAINING_CONFIG`: Batch size, epochs, learning rates, early stopping
- `AUGMENTATION_CONFIG`: Data augmentation parameters
- `MODEL_CONFIG`: Architecture-specific configurations
- `CAMERA_CONFIG`: Real-time service settings
- `API_CONFIG`: Flask API parameters

**Functions:**

- `get_image_shape()`: Returns tuple of (height, width, channels)
- `ensure_directories()`: Creates required directories if missing

### data_loader.py

Handles all data loading, preprocessing, and augmentation operations.

**Class: DataLoader**

Main class for dataset operations.

*Methods:*

- `load_metadata()`: Loads CSV metadata and handles missing values
- `load_images()`: Reads and resizes all images from disk
- `prepare_data()`: Splits into train/val/test sets and normalizes
- `get_data_generator()`: Creates Keras ImageDataGenerator for augmentation
- `preprocess_image()`: Preprocesses single image for inference
- `get_class_distribution()`: Returns class frequency statistics

**Data Preprocessing Steps:**

1. Load metadata from CSV file
2. Handle missing values (fill age with mean)
3. Map image IDs to file paths
4. Load and resize images to (90, 120, 3)
5. Split into train (84%), validation (15%), test (1%)
6. Normalize using training set mean and standard deviation
7. Convert labels to one-hot encoded format

**Data Augmentation:**

Applied during training using ImageDataGenerator:
- Random rotation (up to 10 degrees)
- Random zoom (up to 10%)
- Width and height shifts (up to 10%)
- Horizontal flips

### models.py

Contains all model architecture definitions.

**Class: CyclicLR**

Custom Keras callback implementing cyclic learning rate policy.

*Parameters:*
- `base_lr`: Minimum learning rate
- `max_lr`: Maximum learning rate
- `step_size`: Half-cycle length in iterations
- `mode`: 'triangular', 'triangular2', or 'exp_range'

*Theory:*
Cyclic learning rate allows the learning rate to oscillate between bounds, helping escape local minima and potentially speeding up convergence.

**Sequential CNN Architecture**

```
Input (90, 120, 3)
    │
    ├── Conv2D(32, 3x3) + BatchNorm + ReLU
    ├── Conv2D(64, 3x3) + BatchNorm + ReLU
    ├── Conv2D(64, 3x3) + BatchNorm + ReLU
    ├── MaxPool(2x2) + Dropout(0.2)
    │
    ├── Conv2D(64, 3x3) + BatchNorm + ReLU
    ├── Conv2D(128, 3x3) + BatchNorm + ReLU
    ├── Conv2D(128, 3x3) + BatchNorm + ReLU
    ├── MaxPool(2x2) + Dropout(0.2)
    │
    ├── Conv2D(128, 3x3) + BatchNorm + ReLU
    ├── Conv2D(256, 3x3) + BatchNorm + ReLU
    ├── Conv2D(256, 3x3) + BatchNorm + ReLU
    ├── MaxPool(2x2) + Dropout(0.2)
    │
    ├── Conv2D(7, 1x1) + BatchNorm + ReLU
    ├── Conv2D(7, 6x9)
    ├── Flatten
    └── Softmax
    │
Output (7 classes)
```

**Key Design Decisions:**

- **No Dense Layers**: Uses 1x1 and kernel-sized convolutions to preserve spatial information and reduce parameters
- **Batch Normalization**: After each convolution for faster training and better generalization
- **Progressive Filter Growth**: 32 -> 64 -> 128 -> 256 filters
- **Dropout**: 20% dropout after each pooling block to prevent overfitting

**Custom ResNet Architecture**

```
Input (flexible size, default 90x120x3)
    │
    ├── Conv2D(32, 3x3) + BatchNorm + ReLU
    │
    ├── ResBlock 1 (32 -> 64 filters) + MaxPool
    ├── ResBlock 2 (64 -> 128 filters) + MaxPool
    ├── ResBlock 3 (128 -> 256 filters) + MaxPool
    ├── ResBlock 4 (256 -> 512 filters, no pool)
    │
    ├── Conv2D(7, 1x1) [channel reduction]
    ├── GlobalAveragePooling2D
    └── Softmax
    │
Output (7 classes)
```

**ResBlock Structure:**

```
Input
    │
    ├────────────────────────┐
    │                        │
    ├── Conv2D + BN + ReLU  │
    ├── Conv2D + BN + ReLU  │
    │                        │
    └── Concatenate ─────────┘
         │
         ├── Conv2D(1x1) [transition]
         ├── BatchNorm + ReLU
         └── MaxPool (except final block)
```

**ResNet Features:**

- **Skip Connections**: Concatenate input with processed output
- **L2 Regularization**: 0.001 on convolutional layers
- **Flexible Input**: Can handle variable input sizes
- **Transition Layers**: 1x1 convolutions to reduce channels after concatenation

### train.py

Complete training pipeline with logging and checkpointing.

**Class: ModelTrainer**

Orchestrates the entire training process.

*Methods:*

- `load_data()`: Loads and prepares dataset
- `build_model()`: Creates model and saves architecture
- `get_callbacks()`: Configures training callbacks
- `train()`: Runs training loop with data augmentation
- `evaluate()`: Tests model on held-out test set
- `plot_history()`: Generates training history visualizations
- `save_final_model()`: Saves final trained model
- `run_full_training()`: Executes complete pipeline

**Training Callbacks:**

1. **ModelCheckpoint**: Saves best model based on validation accuracy
2. **CSVLogger**: Logs metrics to CSV file for analysis
3. **TensorBoard**: Creates TensorBoard logs for visualization
4. **ReduceLROnPlateau** (Sequential): Reduces LR when validation loss plateaus
5. **EarlyStopping** (Sequential): Stops training if no improvement
6. **CyclicLR** (ResNet): Implements cyclic learning rate policy

**Training Outputs:**

```
logs/<model>_<timestamp>/
    ├── model_architecture.json
    ├── model_summary.txt
    ├── training_log.csv
    ├── training_history.png
    ├── metrics.json
    └── tensorboard/
```

### evaluate.py

Comprehensive model evaluation and metrics generation.

**Class: ModelEvaluator**

Provides detailed performance analysis.

*Methods:*

- `load_test_data()`: Loads test dataset
- `predict()`: Generates predictions on test set
- `calculate_metrics()`: Computes accuracy, precision, recall, F1
- `print_metrics()`: Displays metrics in formatted output
- `generate_classification_report()`: Creates sklearn classification report
- `plot_confusion_matrix()`: Generates confusion matrix visualizations
- `plot_class_distribution()`: Compares true vs predicted distributions
- `run_full_evaluation()`: Executes complete evaluation pipeline

**Metrics Computed:**

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Per-Class Metrics**: Above metrics for each lesion type
- **Confusion Matrix**: Detailed prediction breakdown

**Evaluation Outputs:**

```
results/<model_name>/
    ├── evaluation_metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── class_distribution.png
```

### inference.py

Single image and batch prediction functionality.

**Class: SkinCancerPredictor**

Main inference interface.

*Methods:*

- `preprocess_image()`: Prepares image for model input
- `predict()`: Predicts class for single image
- `predict_from_file()`: Loads and predicts from file path
- `predict_batch()`: Processes multiple images
- `get_top_k_predictions()`: Returns top K predictions with confidence

**Prediction Output Format:**

```python
{
    'class_code': 'nv',
    'class_name': 'Melanocytic nevi',
    'class_index': 5,
    'confidence': 0.856,
    'all_probabilities': {
        'Melanocytic nevi': 0.856,
        'Melanoma': 0.089,
        ...
    }
}
```

### camera_service.py

Real-time camera detection and REST API service.

**Class: CameraService**

Webcam-based real-time detection.

*Methods:*

- `start_camera()`: Initializes camera capture
- `stop_camera()`: Releases camera resources
- `draw_prediction()`: Overlays prediction info on frame
- `save_screenshot()`: Saves frame with prediction
- `run()`: Main camera loop

**Features:**

- Real-time predictions at ~5-10 FPS
- Visual overlay with class name and confidence
- Top-3 predictions display
- Screenshot saving capability
- Keyboard controls for interaction

**Class: APIService**

Flask-based REST API.

*Endpoints:*

1. `GET /health`: Health check
2. `POST /predict`: Upload image file for prediction
3. `POST /predict_base64`: Send base64-encoded image

*Features:*

- CORS enabled for cross-origin requests
- JSON response format
- Error handling and validation
- Support for multiple image formats

## Data Pipeline

### Data Flow

```
Raw Images + Metadata
    │
    ├── Load metadata CSV
    ├── Map image IDs to file paths
    ├── Load and resize images (90x120)
    │
    ├── Train/Val/Test Split (84%/15%/1%)
    │
    ├── Compute training statistics
    ├── Normalize using train mean/std
    │
    ├── Convert labels to one-hot
    │
    └── Apply data augmentation (training only)
         │
         └── Feed to model
```

### Normalization Strategy

Images are normalized using training set statistics:

```python
normalized_image = (image - train_mean) / train_std
```

Where:
- `train_mean ≈ 160.0` (approximate pixel intensity mean)
- `train_std ≈ 46.7` (approximate pixel intensity std)

This centers the data around zero and scales to unit variance, improving training stability.

## Model Details

### Sequential CNN (Baseline)

**Total Parameters**: ~1.2M

**Performance**: ~75% test accuracy

**Receptive Field Calculation**:
- After 9 Conv layers (3x3) and 3 MaxPool layers (2x2)
- Final receptive field: ~74x74 pixels
- Sufficient to capture lesion features in 90x120 images

**Optimizer**: Adam
- Adaptive learning rates per parameter
- Default learning rate: 0.001
- Beta1: 0.9, Beta2: 0.999

**Loss Function**: Categorical Cross-Entropy
```
L = -Σ y_true * log(y_pred)
```

**Use Case**: Good baseline, fast training, educational purposes

### Custom ResNet (Baseline)

**Total Parameters**: ~2.5M

**Performance**: ~71% test accuracy

**Skip Connection Benefits**:
1. Gradient flow: Enables training of deeper networks
2. Feature reuse: Lower-level features accessible to higher layers
3. Identity mapping: Network can learn to skip layers if needed

**Optimizer**: SGD with Nesterov Momentum
- Learning rate: 0.001 - 0.1 (cyclic)
- Momentum: 0.9
- Nesterov accelerated gradient for faster convergence

**L2 Regularization**:
- Applied to all convolutional layers
- Weight penalty: 0.001
- Helps prevent overfitting

**Use Case**: Demonstrates residual learning, cyclic learning rates

### Extending to Advanced Architectures

**Important Note**: The provided baseline models achieve 71-75% accuracy, which is reasonable for a starting point. However, this architecture is designed to be **easily extensible** to achieve significantly higher accuracy (85-90%+) by integrating modern pre-trained models.

**Why Current Models Show Limited Accuracy**:
1. Trained from scratch on a relatively small dataset (10k images)
2. Simple architectures without transfer learning
3. Limited model capacity compared to state-of-the-art architectures
4. Small input image size (90x120) vs standard (224x224)

**How to Achieve Higher Accuracy**:

The modular design allows easy integration of:
- **Transfer Learning Models**: Leverage pre-trained weights from ImageNet
- **Modern Architectures**: EfficientNet, Vision Transformers, ConvNeXt
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Advanced Training Techniques**: Mixup, CutMix, test-time augmentation

See "Extending the System" section for detailed implementation examples and expected performance improvements.

## Training Process

### Sequential CNN Training

1. **Initialization**: Random weight initialization using Glorot uniform
2. **Data Augmentation**: Applied on-the-fly during training
3. **Batch Size**: 64 images per batch
4. **Epochs**: 30 (with early stopping)
5. **Learning Rate**: Starts at 0.001, reduced by 0.0001x when plateau
6. **Validation**: Monitored every epoch on validation set

**Training Progression**:
- Epoch 1-5: Rapid improvement (60-70% accuracy)
- Epoch 6-15: Steady improvement (70-74% accuracy)
- Epoch 16-30: Fine-tuning (74-75% accuracy)

### ResNet Training

1. **Initialization**: Glorot uniform for weights
2. **Data Augmentation**: Rotation, zoom, shifts
3. **Batch Size**: 64 images per batch
4. **Epochs**: 16
5. **Learning Rate**: Cyclic 0.001 - 0.1 (triangular policy)
6. **Validation**: Monitored every epoch

**Cyclic LR Schedule**:
```
Iterations 0-212: Ramp up 0.001 -> 0.1
Iterations 212-424: Ramp down 0.1 -> 0.001
(Repeats for remaining iterations)
```

## Evaluation Metrics

### Classification Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**:
```
Precision = TP / (TP + FP)
```
Answers: "Of all positive predictions, how many were correct?"

**Recall**:
```
Recall = TP / (TP + FN)
```
Answers: "Of all actual positives, how many did we find?"

**F1 Score**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean balances precision and recall.

### Macro vs Weighted Averaging

**Macro Average**:
- Treats all classes equally
- Good for balanced datasets
- Highlights performance on minority classes

**Weighted Average**:
- Weights by class frequency
- Better for imbalanced datasets
- Reflects overall performance

### Confusion Matrix Interpretation

```
                Predicted
              0   1   2   3   4   5   6
Actual    0 [TN  FP  FP  FP  FP  FP  FP]
          1 [FN  TP  FP  FP  FP  FP  FP]
          ...
```

- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- **Row sum**: Total samples for that class
- **Column sum**: Total predictions for that class

## Deployment Options

### Option 1: Local Camera Service

**Use Case**: Testing and demonstration

**Deployment**:
```bash
python src/camera_service.py --model models/sequential_best.h5 --mode camera
```

**Requirements**:
- Webcam or USB camera
- Display for visualization
- ~50MB RAM for model
- ~2GB GPU memory (optional)

### Option 2: REST API

**Use Case**: Integration with web/mobile applications

**Deployment**:
```bash
python src/camera_service.py --model models/sequential_best.h5 --mode api --port 5000
```

**Docker Deployment**:
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/camera_service.py", "--model", "models/sequential_best.h5", "--mode", "api"]
```

**Scaling Considerations**:
- Use gunicorn for production WSGI server
- Load balancer for multiple instances
- GPU support for faster inference
- Caching for repeated predictions

### Option 3: Batch Processing

**Use Case**: Processing large datasets

**Implementation**:
```python
from src.inference import SkinCancerPredictor

predictor = SkinCancerPredictor('models/sequential_best.h5')
results = predictor.predict_batch(image_list)
```

## API Specification

### Endpoint: POST /predict

**Request**:
```http
POST /predict HTTP/1.1
Host: localhost:5000
Content-Type: multipart/form-data

image: <binary file data>
```

**Response**:
```json
{
  "class_code": "mel",
  "class_name": "Melanoma",
  "class_index": 4,
  "confidence": 0.923,
  "all_probabilities": {
    "Melanoma": 0.923,
    "Melanocytic nevi": 0.042,
    "Benign keratosis-like lesions": 0.018,
    "Basal cell carcinoma": 0.009,
    "Actinic keratoses": 0.005,
    "Vascular lesions": 0.002,
    "Dermatofibroma": 0.001
  }
}
```

**cURL Example**:
```bash
curl -X POST -F "image=@lesion.jpg" http://localhost:5000/predict
```

**Python Example**:
```python
import requests

with open('lesion.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    
result = response.json()
print(f"Predicted: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Error Responses

**400 Bad Request**:
```json
{
  "error": "No image provided"
}
```

**500 Internal Server Error**:
```json
{
  "error": "Prediction failed: <error message>"
}
```

## Performance Optimization

### Training Optimization

1. **Mixed Precision Training**: Use float16 for faster computation
2. **Data Pipeline**: Use `tf.data` API with prefetching
3. **Batch Size**: Increase if GPU memory allows
4. **Multi-GPU**: Use `tf.distribute.MirroredStrategy`

### Inference Optimization

1. **Model Quantization**: Convert to INT8 for faster inference
2. **TensorFlow Lite**: For mobile deployment
3. **Batch Predictions**: Process multiple images together
4. **Model Pruning**: Remove unnecessary weights

### Example: TensorFlow Lite Conversion

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Troubleshooting

### Issue: Out of Memory During Training

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `--batch-size 32`
2. Use mixed precision training
3. Reduce image size in config
4. Use gradient accumulation

### Issue: Model Not Learning

**Symptoms**: Loss not decreasing, accuracy stuck

**Solutions**:
1. Check data preprocessing (normalization)
2. Verify labels are correct
3. Reduce learning rate
4. Add more data augmentation
5. Check for data leakage

### Issue: Overfitting

**Symptoms**: High train accuracy, low validation accuracy

**Solutions**:
1. Increase dropout rate
2. Add more data augmentation
3. Reduce model complexity
4. Use stronger L2 regularization
5. Early stopping

### Issue: Camera Service Not Starting

**Symptoms**: Cannot open camera error

**Solutions**:
1. Check camera permissions
2. Try different camera index: `--camera 1`
3. Ensure camera not in use by other applications
4. Install OpenCV properly: `pip install opencv-python`

### Issue: Poor Predictions on Real Images

**Symptoms**: Model works on test set but fails on new images

**Solutions**:
1. Ensure proper lighting conditions
2. Use similar image quality as training data
3. Crop to lesion region
4. Apply same preprocessing as training
5. Consider domain adaptation techniques

## Additional Resources

### Extending the System

**Adding a New Model**:

The current implementation provides baseline models (~71-75% accuracy), but the architecture is designed to easily integrate more advanced models for significantly higher accuracy (85-90%+).

**Example 1: Transfer Learning with EfficientNet**

1. Create model function in `models.py`:
```python
def build_efficientnet_model():
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, Model
    
    # Load pre-trained EfficientNet
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=get_image_shape()
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(7, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model
```

2. Add configuration:
```python
MODEL_CONFIG['efficientnet'] = {
    'name': 'efficientnet_model',
    'base_learning_rate': 0.0001,
    'fine_tune_learning_rate': 0.00001,
    'dropout_rate': 0.5,
    'fine_tune_at_layer': 100,  # Unfreeze from this layer
}
```

3. Update `get_model()` function

4. Implement two-stage training (optional but recommended):
```python
# Stage 1: Train only the top layers
model = build_efficientnet_model()
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Stage 2: Fine-tune entire network
model.layers[0].trainable = True  # Unfreeze base model
model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
```

**Example 2: Vision Transformer (ViT)**

```python
def build_vit_model():
    from tensorflow.keras.applications import ViTB16
    
    base_model = ViTB16(
        weights='imagenet21k',
        include_top=False,
        input_shape=get_image_shape()
    )
    
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(7, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model
```

**Example 3: Custom Ensemble**

```python
def build_ensemble_model():
    # Create multiple models
    efficientnet = build_efficientnet_model()
    resnet50 = build_resnet50_model()
    densenet = build_densenet_model()
    
    # Average predictions
    ensemble_input = layers.Input(shape=get_image_shape())
    
    pred1 = efficientnet(ensemble_input)
    pred2 = resnet50(ensemble_input)
    pred3 = densenet(ensemble_input)
    
    # Average or weighted average
    ensemble_output = layers.Average()([pred1, pred2, pred3])
    
    model = Model(inputs=ensemble_input, outputs=ensemble_output)
    return model
```

**Recommended Models for Higher Accuracy:**

| Model | Expected Accuracy | Training Time | Complexity |
|-------|------------------|---------------|------------|
| Sequential CNN (baseline) | 75% | 30-40 min | Low |
| Custom ResNet (baseline) | 71% | 25-35 min | Medium |
| EfficientNetB0 | 85-88% | 45-60 min | Medium |
| EfficientNetB3 | 87-90% | 60-90 min | High |
| ResNet50 | 83-86% | 50-70 min | Medium |
| DenseNet121 | 84-87% | 55-75 min | Medium-High |
| Vision Transformer | 88-91% | 90-120 min | Very High |
| Ensemble (3 models) | 90-93% | 2-3 hours | Very High |

**Tips for Higher Accuracy:**

1. **Use Transfer Learning**: Start with ImageNet pre-trained weights
2. **Two-Stage Training**: Train classification head first, then fine-tune entire network
3. **Larger Image Size**: Use 224x224 or 299x299 instead of 90x120
4. **Data Augmentation**: Add more aggressive augmentation (color jittering, mixup, cutmix)
5. **Class Balancing**: Use class weights or oversampling for minority classes
6. **Test-Time Augmentation**: Average predictions over multiple augmented versions
7. **Learning Rate Schedule**: Use cosine annealing or one-cycle policy
8. **Regularization**: Add L2 regularization and dropout appropriately

**Adding New Preprocessing**:

1. Modify `DataLoader.preprocess_image()`
2. Update preprocessing in `SkinCancerPredictor`
3. Ensure consistency across train/inference

### Monitoring and Logging

**TensorBoard**:
```bash
tensorboard --logdir logs/
```

Access at `http://localhost:6006`

**View Training Logs**:
```bash
cat logs/<run_name>/training_log.csv
```

### Best Practices

1. **Always validate preprocessing**: Check image shapes and ranges
2. **Monitor training**: Use TensorBoard and validation metrics
3. **Version control models**: Save with descriptive names and dates
4. **Document experiments**: Keep notes on hyperparameters and results
5. **Test inference pipeline**: Verify predictions make sense
6. **Handle edge cases**: Missing data, corrupted images, etc.

---

For additional questions or issues, please refer to the main README or open an issue on the project repository.

