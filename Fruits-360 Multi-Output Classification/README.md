# 🌱 Agricultural Multi-Output Classification with MobileNetV2

A comprehensive deep learning solution for agricultural product classification using transfer learning and multi-output neural networks.

##  Project Objective

The goal of this project is to build a **multi-output** image classification model for agricultural products using deep learning. The system simultaneously predicts four different attributes of agricultural products from a single image:

| Attribute | Description |
|-----------|-------------|
|  **Object Name** | Specific fruit/vegetable class |
|  **Type** | Fruit, Vegetable, or Nut |
|  **Defects/Diseases** | Presence of defects (Yes/No) |
|  **Maturity Stage** | Unripe, Ripe, Overripe |

## 📊 Dataset Description

The dataset is a **customized version** of the [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits/data) dataset from Kaggle.

### Dataset Customization Process:
- Renamed folder structure for better organization
- Grouped related images together
- Labeled each folder with comprehensive metadata
- Created multi-output classification structure

### Dataset Structure:
```
dataset/
├── train/
│   ├── apple_red_fruit_no_ripe/
│   ├── banana_fruit_no_overripe/
│   └── ...
├── test/
│   └── ...
└── metadata.csv
```

##  Model Architecture & Approach

### Baseline CNN Model
- Custom CNN architecture from scratch
- 4 convolutional layers with MaxPooling
- GlobalAveragePooling2D
- Dense layers with dropout (0.4)
- Multiple output heads for each task

### MobileNetV2 (Transfer Learning)
- Pre-trained MobileNetV2 backbone
- ImageNet weights initialization
- Progressive unfreezing strategy
- Data augmentation pipeline
- L2 regularization and BatchNormalization

### Multi-Output Architecture:
```
Input (100x100x3) → Data Augmentation → MobileNetV2 Base → GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.5)
                                                                                                         ↓
├── defects_diseases: Dense(1, sigmoid)
├── type: Dense(3, softmax)
├── maturity_stage: Dense(3, softmax)
└── object_name: Dense(17, softmax)
```

## ⚙️ Training Strategy

### Three-Phase Training Approach:

1. **Phase 1: Class Balancing**
   - Applied oversampling to handle class imbalance
   - 50/50 balance for defects_diseases

2. **Phase 2: Frozen Training**
   - Initial training with frozen MobileNetV2 layers (15 epochs)

3. **Phase 3: Progressive Unfreezing**
   - Gradually unfroze 20%, 25%, and 30% of base model layers across 13 phases

### Key Training Techniques:
-  **Data Augmentation:** Random flip, rotation, zoom, translation
-  **Dynamic Loss Weights:** Adaptive weighting based on F1 scores
-  **Early Stopping:** Patience=5, monitor validation loss
-  **Learning Rate Reduction:** ReduceLROnPlateau with factor=0.5
-  **Regularization:** L2 regularization (0.005), Dropout (0.5)
-  **Batch Normalization:** Stabilized training process

## 📈 Model Performance & Results

### Final F1 Scores (MobileNetV2 Model):
| Task | F1 Score |
|------|----------|
| Defects/Diseases | **0.85+** |
| Type | **0.98+** |
| Maturity Stage | **0.91+** |
| Object Name | **0.91+** |

### Model Comparison:
| Model | Training Approach | Key Strengths | Observations |
|-------|------------------|---------------|--------------|
| Baseline CNN | From scratch | High accuracy on original split | ⚠️ Potential overfitting due to data leakage |
| MobileNetV2 | Transfer learning + Progressive unfreezing | Better generalization, robust performance | ✅ Consistent performance across all tasks |

## 💻 Implementation Details

### Dependencies & Requirements:
```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
```

### Key Configuration:
| Image Processing | Training Setup |
|------------------|----------------|
| **Image Size:** 100×100×3 | **Optimizer:** Adam (lr=1e-4) |
| **Batch Size:** 32 | **Loss Functions:** Binary/Categorical Cross-entropy |
| **Normalization:** [0, 1] | **Validation Split:** 80/20 stratified |
| **Format:** RGB | **Total Phases:** 14 progressive training phases |



## ⭐ Key Features & Innovations

| Feature | Description |
|---------|-------------|
|  **Multi-Output Learning** | Simultaneous prediction of 4 different agricultural attributes from a single model |
|  **Dynamic Loss Weighting** | Adaptive loss weights based on real-time F1 score performance |
|  **Progressive Unfreezing** | Systematic approach to fine-tuning pre-trained layers |
|  **Class Imbalance Handling** | Sophisticated oversampling strategies for minority classes |
|  **Data Augmentation Pipeline** | Comprehensive augmentation including flip, rotation, zoom, and brightness |
|  **Robust Validation** | Stratified splits to prevent data leakage and ensure fair evaluation |

##  Future Improvements & Extensions

### Technical Enhancements:
- → Implement attention mechanisms for better feature focus
- → Experiment with other architectures (EfficientNet, ResNet)
- → Add uncertainty quantification for predictions
- → Implement model ensemble strategies

### Dataset & Application:
- → Expand dataset with more agricultural products
- → Add temporal tracking for ripeness progression
- → Implement real-time mobile application
- → Integration with IoT sensors for comprehensive monitoring

##  Project Conclusion

This project successfully demonstrates the power of transfer learning and multi-output classification for agricultural applications. The MobileNetV2-based model achieved robust performance across all classification tasks while maintaining computational efficiency suitable for practical deployment.

The progressive training strategy and dynamic loss weighting proved effective in handling the inherent challenges of multi-output learning with imbalanced agricultural datasets.


##  Contact

- GitHub: [3lis0](https://github.com/3lis0)
- LinkedIn: [ali-salama](https://www.linkedin.com/in/ali-salama/)
- kaggle: [alisalama0](https://www.kaggle.com/alisalama0)