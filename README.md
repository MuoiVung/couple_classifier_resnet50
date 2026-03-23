# 👩‍❤️‍👨 Couple vs 👤 Single Classifier

An end-to-end deep learning application to classify whether an image contains a **Couple** (man and woman together) or a **Single person portrait**. This project uses a pre-trained **ResNet50** from the `timm` library, fine-tuned on a custom dataset.

## 🚀 Key Features
- **Binary Classification**: Expertly distinguishes between couples and individuals.
- **Transfer Learning**: Powered by **ResNet50** for high performance even on small datasets.
- **Enhanced Data Augmentation**: Including `RandomResizedCrop`, `RandomAffine`, and `ColorJitter` to improve generalization.
- **Smart Training**: Features **Early Stopping** and **Learning Rate Scheduling** for optimal convergence.
- **Interactive Web UI**: A sleek **Gradio** interface for drag-and-drop image prediction.

---

## 📥 Get Started Immediately (Demo Mode)

If you don't want to train the model from scratch, follow these steps:

1. **Clone the Repo & Install**:
   ```bash
   git clone https://github.com/MuoiVung/couple_classifier_resnet50.git
   cd couple_classifier_resnet50
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Weights**:
   - Go to the [Releases](https://github.com/MuoiVung/couple_classifier_resnet50/releases/tag/v1.0.0) page.
   - Download `models.zip` and extract it to the project root (ensure the path is `models/best_model.pth`).
   - (Optional) Download `dataset.zip` if you want to see the original training images.

3. **Run the App**:
   ```bash
   python3 app.py
   ```

---

## 📈 Training Your Own Model

If you'd like to retrain or use your own data:

1. **Collect Data**:
   ```bash
   python3 download_data.py
   ```
   *This scrapes 100 images per class from Bing and saves them to `dataset/`.*

2. **Start Training**:
   ```bash
   python3 train.py --batch_size 16 --epochs 50
   ```
   *The script will automatically stop if performance plateaus and save the best weights to `models/best_model.pth`.*

---

## 📁 Project Architecture
- `model.py`: ResNet50 architecture definition.
- `train.py`: Training logic with early stopping and persistent best-accuracy tracking.
- `dataset.py`: Advanced image preprocessing and augmentation pipeline.
- `app.py`: Gradio-based web interface for inference.
- `download_data.py`: Automated image scraper using `bing-image-downloader`.

## 📜 License
This project is licensed under the MIT License.
