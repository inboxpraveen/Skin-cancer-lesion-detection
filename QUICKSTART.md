# Quick Start Guide

Get started with the Skin Cancer Detection System in minutes.

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Skin-Cancer-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup directories**
```bash
python scripts/setup_directories.py
```

## Dataset Setup

### Option 1: Using Kaggle API (Recommended)

1. Get Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Scroll to 'API' section
   - Click 'Create New API Token'
   - Move `kaggle.json` to `~/.kaggle/`

2. Download dataset:
```bash
python scripts/download_dataset.py
```

### Option 2: Manual Download

1. Visit https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Download the dataset
3. Extract `HAM10000_metadata.csv` to `data/`
4. Extract all images to `data/images/`

## Training Your First Model

Train the Sequential CNN model (recommended for beginners):

```bash
cd src
python train.py --model sequential --epochs 30 --batch-size 64
```

This will:
- Load and preprocess the dataset
- Train for 30 epochs with data augmentation
- Save the best model to `models/checkpoints/sequential_best.h5`
- Generate training plots in `logs/`
- Take approximately 30-40 minutes on GPU, 2-3 hours on CPU

## Evaluating the Model

After training, evaluate on the test set:

```bash
python evaluate.py --model ../models/checkpoints/sequential_best.h5
```

This will generate:
- Confusion matrix
- Classification report
- Per-class metrics
- Results saved in `results/`

## Making Predictions

### Single Image Prediction

```bash
python inference.py --model ../models/checkpoints/sequential_best.h5 --image path/to/lesion.jpg
```

### Real-Time Camera Detection

```bash
python camera_service.py --model ../models/checkpoints/sequential_best.h5 --mode camera
```

Controls:
- `q` - Quit
- `s` - Save screenshot
- `r` - Toggle region of interest

### API Service

Start the REST API:

```bash
python camera_service.py --model ../models/checkpoints/sequential_best.h5 --mode api --port 5000
```

Test the API:

```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST -F "image=@lesion.jpg" http://localhost:5000/predict
```

## Common Issues

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python train.py --model sequential --batch-size 32
```

### Issue: Camera Not Found

**Solution**: Try different camera index
```bash
python camera_service.py --model model.h5 --mode camera --camera 1
```

### Issue: Slow Training

**Solution**: 
- Ensure TensorFlow is using GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Install CUDA if you have an NVIDIA GPU
- Use smaller image size in `config.py`

## Next Steps

1. **Try different models**: Train the ResNet model for comparison
```bash
python train.py --model resnet --epochs 16
```

2. **Customize training**: Edit `src/config.py` to change hyperparameters

3. **Deploy**: Use the API service to integrate with your application

4. **Improve accuracy**: 
   - Collect more training data
   - Implement ensemble methods
   - Fine-tune hyperparameters

## Getting Help

- Read the full [README.md](README.md) for detailed information
- Check [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for technical details
- Open an issue on GitHub for bugs or questions

## Project Structure Overview

```
Skin-Cancer-Detection/
├── src/                  # Source code
│   ├── config.py        # Configuration
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   ├── inference.py     # Prediction script
│   └── camera_service.py # Camera/API service
├── scripts/             # Helper scripts
├── data/                # Dataset (you provide)
├── models/              # Saved models
├── logs/                # Training logs
├── results/             # Evaluation results
└── requirements.txt     # Dependencies
```

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt
python scripts/setup_directories.py

# 2. Download data
python scripts/download_dataset.py

# 3. Train model
cd src
python train.py --model sequential --epochs 30

# 4. Evaluate
python evaluate.py --model ../models/checkpoints/sequential_best.h5

# 5. Try inference
python inference.py --model ../models/checkpoints/sequential_best.h5 --image test_image.jpg

# 6. Run camera service
python camera_service.py --model ../models/checkpoints/sequential_best.h5 --mode camera
```

That's it! You're ready to use the Skin Cancer Detection System.

