# AIoT-Tridosha-Disease-Diagnosis
# AIoT Based Multimodal Framework for Tridosha-Oriented Disease Diagnosis

## Overview
This project proposes an AIoT based multimodal healthcare system that integrates tongue analysis, pulse analysis, and voice analysis for Tridosha-oriented disease diagnosis.

The system uses Artificial Intelligence and IoT technologies to digitize traditional diagnostic methods and provide objective, non-invasive, and low-cost health assessment.

## Objectives
- To develop an AIoT based system integrating traditional medical diagnosis with sensors and AI models.
- To analyze tongue, pulse, and voice features using deep learning and signal processing.
- To identify dominant dosha such as Vatham, Pitham, and Kabam.
- To store and monitor results using an IoT cloud platform.

## Modules
### 1. Tongue Analysis
Tongue images are classified using deep learning models such as MobileNetV2, ResNet50, DenseNet121, and EfficientNetB0.

### 2. Pulse Analysis
Pulse signals are collected using three PPG sensors connected to Raspberry Pi Pico. The signals are processed to identify dominant dosha based on amplitude features.

### 3. Voice Analysis
Voice recordings are converted into spectrograms and classified using CNN models such as ResNet34, ResNet50, and MobileNetV2.

### 4. IoT Cloud Integration
The processed patient data is uploaded to ThingSpeak for cloud-based storage and monitoring.

## Technologies Used
- Python
- PyTorch
- TensorFlow
- OpenCV
- Librosa
- NumPy
- Pandas
- Matplotlib
- Raspberry Pi Pico
- PPG Sensors
- ThingSpeak

## Results
- Tongue Analysis: MobileNetV2 achieved the best performance among tested models.
- Voice Analysis: MobileNetV2 achieved high classification accuracy.
- Pulse Analysis: Dominant dosha is identified using PPG waveform amplitude comparison.

## Hardware Components
- Raspberry Pi Pico
- 3 PPG Pulse Sensors
- Jumper Wires
- USB Cable
- Computer / Laptop

## Future Work
- Improve dataset size and clinical validation.
- Develop mobile application support.
- Add advanced multimodal fusion techniques.
- Improve real-time cloud dashboard.
- Include explainable AI for diagnostic support.

