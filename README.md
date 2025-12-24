# Skin Cancer Detection System

A production-grade deep learning system for automated skin lesion classification using the HAM10000 dataset. This system provides training, evaluation, and real-time inference capabilities for detecting seven types of skin lesions.

## Overview

This project implements a computer vision system that can identify seven different types of skin lesions:

- **akiec** - Actinic keratoses and intraepithelial carcinoma
- **bcc** - Basal cell carcinoma
- **bkl** - Benign keratosis-like lesions
- **df** - Dermatofibroma
- **mel** - Melanoma
- **nv** - Melanocytic nevi
- **vasc** - Vascular lesions

The system achieves approximately 75% accuracy on the test set using a Sequential CNN architecture and provides multiple deployment options including real-time camera analysis and REST API integration.

## Features

- **Multiple Model Architectures**: Sequential CNN and Custom ResNet implementations
- **Comprehensive Training Pipeline**: Includes data augmentation, learning rate scheduling, and checkpointing
- **Detailed Evaluation Metrics**: Confusion matrices, classification reports, and per-class performance
- **Real-Time Camera Service**: Live skin lesion detection through webcam
- **REST API**: Flask-based API for integration with other applications
- **Command-Line Tools**: Easy-to-use scripts for training, evaluation, and inference

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in minutes with step-by-step instructions
- **[Technical Documentation](PROJECT_DOCUMENTATION.md)** - Comprehensive technical details including:
  - System architecture and design principles
  - Detailed module documentation
  - Model architecture deep-dive
  - Training and evaluation processes
  - API specifications
  - Performance optimization tips
  - Troubleshooting guide

## Project Structure

```
Skin-Cancer-Detection/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── models.py              # Model architectures
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation and metrics
│   ├── inference.py           # Single image prediction
│   └── camera_service.py      # Real-time camera and API service
├── data/
│   ├── HAM10000_metadata.csv  # Dataset metadata
│   └── images/                # Skin lesion images
├── models/                    # Saved models
├── logs/                      # Training logs
├── results/                   # Evaluation results
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
└── PROJECT_DOCUMENTATION.md   # Detailed technical documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Skin-Cancer-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the HAM10000 dataset:
   - Visit [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - Download and extract to `data/` directory
   - Ensure `HAM10000_metadata.csv` is in `data/`
   - Extract all images to `data/images/`

## Usage

### Training a Model

Train the Sequential CNN model:
```bash
python src/train.py --model sequential --epochs 30 --batch-size 64
```

Train the ResNet model:
```bash
python src/train.py --model resnet --epochs 16 --batch-size 64
```

Training outputs:
- Model checkpoints in `models/checkpoints/`
- Training logs in `logs/`
- Training history plots

### Evaluating a Model

Evaluate a trained model on the test set:
```bash
python src/evaluate.py --model models/checkpoints/sequential_best.h5
```

Evaluation outputs:
- Confusion matrices
- Classification report
- Per-class metrics
- Results saved in `results/`

### Running Inference

Predict on a single image:
```bash
python src/inference.py --model models/checkpoints/sequential_best.h5 --image path/to/image.jpg
```

Show top-3 predictions:
```bash
python src/inference.py --model models/checkpoints/sequential_best.h5 --image path/to/image.jpg --top-k 3
```

### Real-Time Camera Service

Run live detection using your webcam:
```bash
python src/camera_service.py --model models/checkpoints/sequential_best.h5 --mode camera
```

Camera controls:
- Press `q` to quit
- Press `s` to save screenshot
- Press `r` to toggle region of interest

### API Service

Start the REST API server:
```bash
python src/camera_service.py --model models/checkpoints/sequential_best.h5 --mode api --port 5000
```

API endpoints:

**Health Check**
```bash
GET http://localhost:5000/health
```

**Predict (file upload)**
```bash
POST http://localhost:5000/predict
Content-Type: multipart/form-data
Body: image file
```

**Predict (base64)**
```bash
POST http://localhost:5000/predict_base64
Content-Type: application/json
Body: {"image": "<base64-encoded-image>"}
```

Response format:
```json
{
  "class_code": "nv",
  "class_name": "Melanocytic nevi",
  "class_index": 5,
  "confidence": 0.856,
  "all_probabilities": {
    "Melanocytic nevi": 0.856,
    "Melanoma": 0.089,
    "Benign keratosis-like lesions": 0.032,
    ...
  }
}
```

## Model Architectures

### Sequential CNN

A straightforward convolutional neural network with:
- 3 blocks of Conv-BatchNorm-MaxPool layers
- Dropout for regularization
- Global average pooling instead of dense layers
- ~75% test accuracy

### Custom ResNet

A residual network inspired by ResNet18 with:
- 4 residual blocks with skip connections
- Cyclic learning rate policy
- L2 regularization
- ~71% test accuracy

The Sequential CNN performs better on this dataset as the images are relatively simple compared to more challenging datasets.

## Dataset

The HAM10000 dataset contains 10,015 dermatoscopic images collected from different populations. More than 50% of lesions are confirmed through histopathology, with the rest confirmed through follow-up examination, expert consensus, or confocal microscopy.

**Class Distribution:**
- Melanocytic nevi (nv): ~67%
- Melanoma (mel): ~11%
- Benign keratosis (bkl): ~11%
- Basal cell carcinoma (bcc): ~5%
- Actinic keratoses (akiec): ~3%
- Vascular lesions (vasc): ~1%
- Dermatofibroma (df): ~1%

The dataset is imbalanced, with melanocytic nevi being the dominant class.

## Performance

### Sequential CNN Model
- Test Accuracy: ~75%
- Training Time: ~30-40 minutes (30 epochs, GPU)
- Inference Speed: ~20ms per image (GPU)

### Custom ResNet Model
- Test Accuracy: ~71%
- Training Time: ~25-35 minutes (16 epochs, GPU)
- Inference Speed: ~25ms per image (GPU)

## Configuration

Key configurations can be modified in `src/config.py`:

- Image size and preprocessing parameters
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Model architecture parameters
- API and camera service settings

## Development

### Adding a New Model

1. Define the model architecture in `src/models.py`
2. Add model configuration to `src/config.py`
3. Update the `get_model()` function to support the new model
4. Train and evaluate using existing scripts

### Customizing Data Augmentation

Modify `AUGMENTATION_CONFIG` in `src/config.py`:
```python
AUGMENTATION_CONFIG = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': False,
}
```

## Limitations

- Model performance is limited by dataset size and quality
- Class imbalance affects predictions for rare classes
- Real-time camera service requires good lighting conditions
- The system is designed for research and educational purposes
- **Not intended for clinical diagnosis without expert review**

## Future Improvements

- Implement ensemble methods for better accuracy
- Add attention mechanisms to highlight important regions
- Support for multi-label classification
- Integration with DICOM medical imaging standards
- Mobile deployment using TensorFlow Lite
- Explainability features (Grad-CAM visualizations)

## References

- HAM10000 Dataset: [https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- Original Research Paper: [https://arxiv.org/abs/1803.10417](https://arxiv.org/abs/1803.10417)
- Dataset Description: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).

## License

This project is provided for educational and research purposes. Please ensure compliance with the HAM10000 dataset license when using this code.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Disclaimer

This system is intended for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## Contact

For questions or feedback, please open an issue on the project repository.
