# Pulse Analysis Module

## Overview
The pulse analysis module is developed to digitally analyze physiological pulse signals for Tridosha-oriented health assessment. In traditional Siddha and Ayurvedic diagnosis, pulse examination (Nadi analysis) is used to identify the balance of Vata, Pitta, and Kapha doshas.

This system uses Photoplethysmography (PPG) sensors and Raspberry Pi Pico to capture pulse waveforms and perform signal processing for automated dosha classification.

---

## Hardware Components

- Raspberry Pi Pico
- 3 PPG Pulse Sensors
- Jumper Wires
- USB Cable
- Breadboard

---

## Working Principle

Three PPG sensors are placed at different pulse positions corresponding to:
- Vata
- Pitta
- Kapha

The sensors detect blood volume changes and generate analog pulse signals. These signals are converted into digital values using the ADC of Raspberry Pi Pico.

The processed signals are analyzed to determine the dominant dosha based on waveform characteristics and amplitude variations.

---

## Signal Acquisition

- Sampling Rate: 100 Hz
- Sampling Interval: 10 ms
- Total Samples: 500
- Communication: Serial Communication (115200 baud)

GPIO Connections:
- GPIO26 → Vata Sensor
- GPIO27 → Pitta Sensor
- GPIO28 → Kapha Sensor

---

## Signal Processing

The following preprocessing techniques are applied:

- DC Offset Removal
- Moving Average Smoothing
- Noise Reduction
- Peak-to-Peak Amplitude Extraction

### Amplitude Formula

A = xmax - xmin

The highest amplitude among the three channels is considered the dominant dosha.

---

## Output Parameters

The system provides:
- Dominant Dosha
- Percentage Contribution
- Pulse Waveform Graph
- Related Physiological Symptoms

Example:
- Dominant Dosha: Kapha
- Symptoms:
  - Obesity
  - Cold
  - Allergy
  - Sinus Congestion

---

## Features

- Non-invasive pulse monitoring
- Real-time waveform visualization
- Automated dosha identification
- Low-cost embedded healthcare system
- Cloud-ready data integration

---

## Technologies Used

- Python
- MicroPython
- Raspberry Pi Pico
- Serial Communication
- NumPy
- Matplotlib

---

## Applications

- Siddha and Ayurvedic digital diagnosis
- IoT healthcare monitoring
- Physiological signal analysis
- Remote patient monitoring

---

## Future Improvements

- Advanced feature extraction
- Heart rate variability analysis
- Machine learning-based pulse classification
- Mobile app integration
- Clinical validation with larger datasets
