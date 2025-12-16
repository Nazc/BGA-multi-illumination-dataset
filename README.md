# BGA Defect Classification: Multi-Illumination XAI Experimentation

## Project Overview
This repository presents a comprehensive pipeline for Ball Grid Array (BGA) defect detection using deep learning and explainable AI (XAI) methods. The main experimentation is documented in `notebooks/XAI_Chapter.ipynb`, which covers dataset construction, model training, evaluation, and interpretability analysis. The project is based on the methodology and results described in the paper "Multi-Illumination Feature Enhancement for Robust BGA Defect Detection and Explainable Attention Alignment".

## Dataset Description

### Source and Motivation
BGA packages are widely used in electronics, but their hidden solder joints pose significant inspection challenges. To address limitations of conventional RGB imaging, this project introduces a novel multi-illumination dataset (png4ch) alongside a standard RGB dataset (imgShiny). Both datasets are derived from original device images, manually cropped to focus on the device area.

### Acquisition Protocol
- **Devices:** Texas Instrument TPS7A0508PYKAR, manipulated to produce controlled defect conditions.
- **Imaging:** Fixed magnification, stable mounting, constant exposure/gain, variable illumination intensity using a 56-LED ring light and auxiliary LED (MIC-209).
- **Defect Taxonomy:**
  - Absence: Missing solder ball
  - Presence-defect: Deformed, misaligned, or partially removed solder ball
  - Presence-good: Intact solder ball
- **Labeling:** All samples verified and labeled by a semiconductor inspection specialist.

### Dataset Structure
All experiments use images from the `224px` subfolders of two datasets in the `../data` directory:

- `imgShiny/224px/` (RGB, 3 channels)
- `png4ch/224px/` (RGBA, 4 channels: stacked grayscale images under four illumination intensities)

**Note:** The original images (in `original/`) are not used directly in experiments, but were manually cropped to extract device images for the datasets.

#### Folder Organization
```
imgShiny/
    224px/
        orFold1/
            train/
            train_aug/
            val/
            test/
        orFold2/
        ...
        orFold5/

png4ch/
    224px/
        orFold1/
            train/
            train_aug/
            val/
            test/
        orFold2/
        ...
        orFold5/
```
- Each `orFoldX` folder is one of 5 cross-validation folds.
- Each fold contains:
  - `train/`: Training images
  - `train_aug/`: Augmented training images (via `augment_images`)
  - `val/`: Validation images
  - `test/`: Test images
- Each class (`absence`, `presence-defect`, `presence-good`) is a subfolder within these directories.

#### Image Format
- `imgShiny/224px/`: RGB, 224x224 pixels
- `png4ch/224px/`: RGBA, 224x224 pixels (each channel = grayscale under different illumination)

#### Data Augmentation and Splitting
- Augmentation pipeline applied to reach **1000 images per class** per dataset.
  - Random resized crop to 224x224 (scale 0.8–1.0)
  - Other transforms for diversity and class balance
- After augmentation: **3 classes × 1000 images = 3000 images per dataset**
- Five folds generated for cross-validation:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Splits are stratified by class for balance.

#### Normalization
- Mean/std normalization applied per channel after augmentation.
- All folds share the same preprocessing pipeline.

## Experimental Methodology

### Hardware and Frameworks
- **Workstation:** MSI Katana GF76 (Intel i7-11800H, RTX 3060 GPU, 16GB RAM, 1TB SSD)
- **Frameworks:**
  - Python 3.9.13
  - PyTorch 2.0.1
  - Torchvision 0.15.2
  - CUDA Toolkit 11.8
  - scikit-learn 1.3.0
  - Matplotlib, Seaborn, Pillow, tqdm
- GPU acceleration enabled for all training and inference.

### Model Architectures
- **ResNet50** (pretrained)
- **MobileNetV2** (pretrained)
- **SimpleCNN** (custom, trained from scratch)
- For `png4ch`, a learnable 1×1 convolution projects 4 channels to 3 for compatibility with pretrained backbones.

### Training Protocols
- Grid search over batch size, learning rate, and epochs
- Cross-entropy loss with class weighting
- Early stopping and best epoch tracking
- All models trained on 224x224 images
- Results and metrics logged incrementally to CSV files

### Evaluation Metrics
- Validation/Test F1, Accuracy, Precision
- Epochs to convergence (peak validation F1)
- Metrics computed per fold and averaged

### Explainable AI (XAI) Analysis
- **GradCAM, GradCAM++:** Pure PyTorch implementation for CAM methods
- **LIME, SHAP:** Local surrogate models for pixel-level explanations
- XAI outputs saved with informative filenames (class, fold, architecture)
- RGBA support for XAI visualizations
- Heatmaps overlaid on test images to assess attention alignment

### Experiment Tracking
- Results table includes best epoch, train/val metrics
- Curves table for plotting training/validation curves
- All outputs organized by architecture and fold

## Results Summary

### Quantitative Results
- Both datasets achieve high F1 scores, with RGB (imgShiny) yielding more consistent metrics.
- Multi-illumination (png4ch) improves attention alignment with defect regions, as confirmed by XAI methods.
- Training and inference times are fastest for MobileNetV2, followed by ResNet50 and SimpleCNN.

#### Example Metrics (see paper for full tables)
| Model        | Dataset   | ValF1 | TestAcc | TestF1 |
|--------------|-----------|-------|---------|--------|
| ResNet50     | imgShiny  | 1.000 | 1.000   | 1.000  |
| MobileNetV2  | png4ch    | 0.966 | 0.981   | 0.981  |
| SimpleCNN    | png4ch    | 0.940 | 0.830   | 0.835  |

### Qualitative Explainability
- GradCAM, GradCAM++, and LIME visualizations show that models trained on png4ch consistently highlight solder ball regions with defects, while imgShiny models may attend to reflections or background textures.
- Multi-illumination feature enhancement guides models toward semantically meaningful features, improving trustworthiness for industrial deployment.

## Reproducibility
- Follow the steps in `notebooks/XAI_Chapter.ipynb`.
- Ensure dataset folders are structured as described.
- All code is documented in English for clarity and collaboration.

## Notes
- Only images from the `224px` folders are used for training, validation, and testing.
- The original images are not used directly in experiments.
- Augmentation and XAI methods are robust to both RGB and RGBA formats.
- For further details, refer to the notebook, scripts, and the referenced paper in this repository.

---
For questions or collaboration, contact the authors or open an issue in this repository.