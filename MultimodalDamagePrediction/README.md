# Multimodal AI System for Aircraft Composite Damage

This project provides an end-to-end multimodal pipeline to predict internal composite damage (C-scan generation) from surface dents (images, point clouds, metadata).

## 1. Setup Instructions
To get started, ensure you have Python 3.9+ installed.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Integrating Given Datasets**
   - The datasets should be structured and placed into `data/raw/` or `data/processed/`.
   - Update `data/dataset.py` specifically `_load_samples()` to read your specific dataset manifest (e.g. a JSON or CSV describing paths).
   - Ensure you are converting point clouds to valid numpy arrays of shape `[N, 3]` inside the `preprocess_pointcloud()` stage, or pre-save them to `.npy` formats.

## 2. Using the System

### Running the User Interface (App)
The dashboard uses Streamlit to allow users to upload files and instantly run the models.
```bash
python main.py ui
```
This will launch a Streamlit server on `http://localhost:8501`. 
Upload an image (`.png`/`.jpg`), upload a point cloud tensor (`.npy`), and fill in the required layout metadata to get a predicted C-Scan with a damage mask and severity score.

### Running the Training Pipeline
Ensure your dataset paths are valid. The training script will orchestrate dataset loading, GAN augmentation, early stopping, and metric tracking.
```bash
python main.py train --config configs/config.yaml --data_dir data/processed
```
The best checkpoint will be stored at `checkpoints/best_model.pth`.

## 3. Configuration
You can edit the core model architecture and hyperparameter settings directly within `configs/config.yaml`.
