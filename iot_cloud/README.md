# IoT Cloud Integration Module

## Overview
The IoT cloud integration module is used to store and monitor the results obtained from tongue, pulse, and voice analysis. In this project, IoT technology helps in connecting the AI-based diagnostic system with a cloud platform for digital record keeping and remote health monitoring.

The processed patient data is uploaded to ThingSpeak, where it can be viewed, stored, and analyzed through a cloud dashboard.

---

## Purpose

The main purpose of IoT cloud integration is to:

- Store patient diagnostic outputs digitally
- Monitor tongue, pulse, and voice results remotely
- Maintain long-term patient health records
- Support real-time healthcare monitoring
- Reduce manual record maintenance

---

## Cloud Platform Used

### ThingSpeak

ThingSpeak is used as the IoT cloud platform for storing and visualizing patient data.

It allows:
- Cloud-based data storage
- Real-time data upload
- Graph visualization
- CSV data export
- Remote monitoring

---

## Data Uploaded to Cloud

The following data can be uploaded to the cloud:

- Patient Name
- Age
- Gender
- Tongue Analysis Output
- Pulse Analysis Output
- Voice Analysis Output
- Dominant Dosha
- Symptoms
- Date and Time

---

## Workflow

1. Tongue image is analyzed using deep learning models.
2. Pulse signals are collected using PPG sensors and processed.
3. Voice recordings are converted into spectrograms and classified.
4. The final diagnostic outputs are combined.
5. The combined results are uploaded to ThingSpeak cloud.
6. The stored data can be monitored remotely through the cloud dashboard.

---

## System Architecture

```txt
Tongue Analysis
      |
Pulse Analysis
      |
Voice Analysis
      |
AI-Based Result Processing
      |
IoT Communication
      |
ThingSpeak Cloud Dashboard
      |
Patient Data Storage and Monitoring
