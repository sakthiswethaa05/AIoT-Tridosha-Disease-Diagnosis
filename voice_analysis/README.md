# Voice Analysis Module

## Overview
The voice analysis module is designed to analyze voice recordings for automated physiological and pathological assessment using deep learning techniques. In traditional Siddha and Ayurvedic diagnosis, voice characteristics such as tone, pitch, clarity, and intensity are considered important indicators of internal health conditions and dosha imbalance.

This module converts voice recordings into spectrogram representations and uses Convolutional Neural Networks (CNNs) for classification and dosha-oriented analysis.

---

## Dataset

The voice analysis system uses the Saarbruecken Voice Database (SVD), which contains recordings from healthy individuals and patients with voice disorders.

### Dataset Features
- Sustained vowel sounds (/a/, /i/, /u/)
- Healthy and pathological voice samples
- Multiple pitch variations
- Clinical voice disorder recordings

### Classes
- Healthy Voice
- Pathological Voice

### Total Samples Used
- Approximately 2,484 voice samples

---

## Audio Preprocessing

The following preprocessing techniques were applied:

- Audio normalization
- Noise reduction
- Conversion to spectrograms
- Mel-frequency analysis
- Resizing spectrograms to 224 × 224

The generated spectrogram images are used as input for CNN-based classification models.

---

## Feature Extraction

Important voice features extracted include:
- Frequency patterns
- Pitch
- Energy distribution
- Spectral information
- Mel Frequency Cepstral Coefficients (MFCC)

These features help identify pathological variations and dosha-related voice characteristics.

---

## Deep Learning Models Used

The following pretrained CNN architectures were implemented and compared:

- ResNet34
- ResNet50
- MobileNetV2

Transfer learning was used to improve classification accuracy and reduce training time.

---

## Training Configuration

- Framework: PyTorch
- Epochs: 10
- Input Size: 224 × 224
- Train / Validation / Test Split:
  - 70% Training
  - 15% Validation
  - 15% Testing

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
| MobileNetV2 | 99.84% |
| ResNet50 | 90.00% |
| ResNet34 | 85.48% |

MobileNetV2 achieved the highest classification accuracy with efficient performance and lightweight architecture.

---

## Output

The system predicts:
- Voice condition
- Dominant dosha
- Related physiological symptoms

Example:
- Dominant Dosha: Pitham
- Symptoms:
  - Sharp and intense voice
  - High pitch tone
  - Strong vocal energy
  - Slightly aggressive sound

---

## Technologies Used

- Python
- PyTorch
- Librosa
- NumPy
- Matplotlib
- CNN Architectures

---

## Applications

- Voice pathology detection
- AI-assisted traditional diagnosis
- Non-invasive healthcare systems
- Remote health monitoring
- Automated dosha analysis

---

## Future Improvements

- Real-time voice analysis
- Multilingual voice support
- Advanced feature fusion
- Mobile application integration
- Clinical validation with larger datasets
