# Tongue Analysis Module

## Overview
The tongue analysis module is designed to automatically classify tongue images into different pathological and physiological categories using deep learning techniques. In traditional Siddha and TCM diagnosis, tongue appearance provides important information about internal body conditions and dosha imbalance.

This module uses Convolutional Neural Networks (CNNs) to analyze tongue color, coating, texture, cracks, and spots for automated disease-oriented assessment.

---

## Dataset

The tongue dataset consists of tongue images collected from:
- TCM Tongue Dataset
- Roboflow Normal Tongue Dataset

### Total Dataset Size
- Total Images: 7,119
- Classes: 9

### Tongue Classes
- Normal
- Pale
- Red
- Purple
- White Coat
- Yellow Coat
- Black Coat
- Spots
- Cracked

---

## Preprocessing

The following preprocessing techniques were applied:
- Image resizing to 224 × 224
- Image normalization
- Tensor conversion
- Dataset splitting:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

---

## Deep Learning Models Used

The following pretrained CNN architectures were implemented and compared:

- MobileNetV2
- ResNet50
- DenseNet121
- EfficientNetB0

Transfer learning was used to improve classification accuracy and reduce training time.

---

## Training Configuration

- Framework: PyTorch
- Epochs: 10
- Operating System: Windows
- Processor: Intel i5
- RAM: 8GB

---

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

---

## Results

| Model | Accuracy |
|------|------|
| MobileNetV2 | 78.50% |
| DenseNet121 | 78.07% |
| EfficientNetB0 | 77.86% |
| ResNet50 | 77.22% |

MobileNetV2 achieved the best performance for tongue classification with improved generalization capability.

---

## Sample Output

The system predicts:
- Tongue class
- Associated disease condition
- Dominant dosha
- Related symptoms

Example:
- Predicted Class: White Coat
- Dominant Dosha: Kabam
- Symptoms: Bloating, heaviness, mucus formation

---

## Applications

- AI-assisted traditional diagnosis
- Digital Siddha healthcare systems
- Remote health monitoring
- Automated tongue-based disease screening

---

## Future Improvements

- Increase dataset size
- Improve model accuracy
- Real-time mobile application support
- Clinical validation with larger populations
