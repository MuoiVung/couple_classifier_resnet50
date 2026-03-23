# 💑 Couple vs 👤 Single Classifier

An end-to-end deep learning application to classify whether an image contains a "Couple" or a "Single" person. This project uses a pre-trained **ResNet50** from the `timm` (PyTorch Image Models) library.

## 🚀 Features
- **Automated Data Collection**: Scraping 200 images from Bing.
- **Data Augmentation**: Using PyTorch `Transforms` (Rotation, Flip, Color Jitter) to enhance dataset diversity.
- **Deep Learning Model**: ResNet50 architecture fine-tuned for binary classification.
- **Web UI**: User-friendly interface built with **Gradio** for real-time predictions.

## 🛠️ Installation

1. **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📈 How to Use

### 1. Collect Data
Download images for training (100 couples, 100 singles):
```bash
python download_data.py
```
*Images will be saved to the `dataset/` directory.*

### 2. Train the Model
Start the training process with 10 epochs (default):
```bash
python train.py --epochs 10 --batch_size 16
```
*The best model will be saved as `models/best_model.pth`.*

### 3. Launch the Application
Run the Gradio web interface:
```bash
python app.py
```

## 📁 Project Structure
- `download_data.py`: Script for image scraping.
- `dataset.py`: Data loading and augmentation.
- `model.py`: ResNet50 model definition.
- `train.py`: Training and validation loop.
- `app.py`: Gradio web interface.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Excludes environment and local data from Git.

## ⚖️ License
MIT License
