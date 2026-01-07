# Satellite Imagery-Based Property Valuation

Multimodal machine learning system combining structured property data with satellite imagery for real estate price prediction.

---

## Project Overview

This project implements a property valuation system that integrates traditional tabular features (square footage, bedrooms, location) with satellite imagery acquired programmatically via public APIs. The system achieves 0.85% improvement over tabular-only baselines by incorporating visual features extracted using deep learning.

**Key Results:**
- Enhanced XGBoost: R² = 0.8783, RMSE = $123,580
- Baseline XGBoost: R² = 0.8709, RMSE = $127,298
- Improvement: $3,718 average error reduction per property

**Technologies:** Python, PyTorch, XGBoost, ResNet50, Grad-CAM, ESRI ArcGIS API

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Running the Project

**Option 1: Main Notebook (Recommended)**

```bash
# Open the main notebook
jupyter notebook PropertyValuation_SatelliteImagery_23113099.ipynb

# Run all cells sequentially (Kernel → Restart & Run All)
```

This single notebook contains the complete end-to-end pipeline:
1. Programmatic satellite image acquisition
2. Exploratory data analysis
3. Feature extraction (CNN + PCA)
4. Model training (Baseline, Neural Fusion, Enhanced XGBoost)
5. Grad-CAM explainability
6. Performance comparison
7. Final predictions generation

**Output:** `23113099_final.csv` with predictions for test set

**Option 2: Modular Workflow (Alternative &  NOT RECOMMENDED)**

```bash
# Step 1: Preprocessing
jupyter notebook preprocessing.ipynb

# Step 2: Model Training
jupyter notebook model_training.ipynb
```

Note: This alternative workflow is simpler but excludes comprehensive EDA and Grad-CAM analysis.

---

## Repository Structure

```
├── PropertyValuation_SatelliteImagery_23113099.ipynb  # Main notebook (use this)
├── preprocessing.ipynb                                # Alternative: data prep only
├── model_training.ipynb                               # Alternative: training only
├── data_fetcher.py                                    # Satellite image downloader
├── requirements.txt                                   # Python dependencies
├── 23113099_final.csv                                 # Final predictions
├── 23113099_report.pdf                                # Project report
├── README.md                                          # This file
├── data/
│   └── raw/
│       ├── train.csv                                  # Training data (21,613 samples)
│       └── test.csv                                   # Test data (4,323 samples)
└── outputs/                                           # Generated files
    ├── *.png                                          # Visualizations (10 files)
    ├── *.csv                                          # Results and comparisons
    ├── *.pkl                                          # Saved models and scalers
    └── *.pth                                          # PyTorch model weights
```

**Note:** `data/images/` folder is created automatically by the notebook during image acquisition (not included in repo due to size).

---

## Features

### Data Acquisition
- Programmatic satellite image download using ESRI ArcGIS World Imagery API
- Free public API (no authentication required)
- 16,209 training images + 5,404 test images acquired
- Zoom level 17 (~2.39 meters/pixel resolution)

### Feature Engineering
- Tabular features: 19 property attributes (size, location, quality)
- Image features: ResNet50 CNN extraction (2048-D embeddings)
- Dimensionality reduction: PCA (2048-D → 20-D, retains 54% variance)
- Combined feature space: 39 features total

### Models Implemented
1. **Baseline XGBoost** - Tabular features only
2. **Neural Network Fusion** - Late fusion of tabular MLP + image CNN
3. **Enhanced XGBoost** - Tabular + PCA-reduced image features (BEST)

### Explainability
- Grad-CAM visualization showing CNN attention patterns
- Feature importance analysis
- Visual interpretation of model decisions

---

## Key Results

| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| Baseline XGBoost (Tabular) | $127,298 | 0.8709 | $71,149 |
| Neural Fusion (Tab + Img) | $183,695 | 0.7311 | $99,022 |
| **Enhanced XGBoost** (Tab + Img PCA) | **$123,580** | **0.8783** | **$69,535** |

**Findings:**
- Enhanced XGBoost outperforms baseline by 0.85% R²
- Feature augmentation superior to neural fusion (16% R² difference)
- Visual features most valuable for waterfront properties (-$12,500 error)
- Top image feature (cnn_pc1) ranks 6th overall in importance

---

## Dataset

**Source:** King County House Sales (Kaggle)

**Training Data:**
- 21,613 property sales
- 19 features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, year_built, lat, long, etc.
- Target: Sale price (USD)
- Price range: $78,000 - $7,700,000

**Test Data:**
- 4,323 properties
- Same features (excluding price)

---

## Requirements

Main dependencies (see `requirements.txt` for complete list):

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

**GPU Support:** CUDA-enabled PyTorch recommended for faster CNN feature extraction (6x speedup).

---

## Methodology

### Pipeline Overview

1. **Data Loading:** Load 21,613 properties with 19 tabular features
2. **Image Acquisition:** Download satellite images via ESRI API using lat/long
3. **Feature Extraction:** Extract 2048-D features using pre-trained ResNet50
4. **Dimensionality Reduction:** Apply PCA to reduce to 20 dimensions
5. **Feature Fusion:** Concatenate tabular (19) + image PCA (20) = 39 features
6. **Model Training:** Train XGBoost regressor on combined features
7. **Evaluation:** Compare against tabular-only baseline
8. **Explainability:** Generate Grad-CAM visualizations
9. **Prediction:** Generate final predictions on test set

### Architecture (Enhanced XGBoost - Best Model)

```
Tabular (19) ──→ StandardScaler ────────────────┐
                                                │
                                                ├──→ Concatenate (39) ──→ XGBoost ──→ Price
                                                │
Image (224×224) ──→ ResNet50 ──→ 2048-D ──→ PCA (20-D) 
```

---

## Outputs

The notebook generates:

### Files
- `23113099_final.csv` - Final predictions (5,404 test samples)
- `complete_model_comparison.csv` - Performance comparison table

### Visualizations (10 files)
1. `visual_analysis_by_price.png` - Price distribution and quartile analysis
2. `geospatial_analysis.png` - Geographic distribution and patterns
3. `correlation_heatmap.png` - Feature correlation matrix
4. `temporal_analysis.png` - Time-series trends
5. `feature_distributions.png` - Feature histograms
6. `price_vs_features.png` - Scatter plots and relationships
7. `outlier_analysis.png` - Outlier detection
8. `enhanced_geographic.png` - Hexbin density mapping
9. `comprehensive_comparison_all_models.png` - Model comparison charts
10. `gradcam_explainability.png` - CNN attention heatmaps

### Models
- `best_fusion_model.pth` - Trained neural fusion model
- `cnn_scaler.pkl` - CNN feature scaler
- `y_scaler.pkl` - Target price scaler
- `cnn_pca.pkl` - PCA transformer

---

## Grad-CAM Explainability

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of satellite images influence predictions:

**Key Findings:**
- Highest activation: Waterfront boundaries (+$973K premium)
- Medium activation: Vegetation density (+12% average)
- Moderate activation: Urban context and lot characteristics

These patterns validate that the model focuses on domain-relevant features.

---

## Acknowledgments

- ESRI for providing free access to World Imagery API
- Kaggle for hosting the King County house sales dataset
- PyTorch and scikit-learn communities

---

**Last Updated:** January 2026
