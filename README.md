┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ███████╗██╗  ██╗██╗███╗   ██╗                                │
│   ██╔════╝██║ ██╔╝██║████╗  ██║                                │
│   ███████╗█████╔╝ ██║██╔██╗ ██║                                │
│   ╚════██║██╔═██╗ ██║██║╚██╗██║                                │
│   ███████║██║  ██╗██║██║ ╚████║                                │
│   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                                │
│                                                                 │
│   ██████╗ ██╗███████╗███████╗ █████╗ ███████╗███████╗          │
│   ██╔══██╗██║██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝          │
│   ██║  ██║██║███████╗█████╗  ███████║███████╗█████╗            │
│   ██║  ██║██║╚════██║██╔══╝  ██╔══██║╚════██║██╔══╝            │
│   ██████╔╝██║███████║███████╗██║  ██║███████║███████╗          │
│   ╚═════╝ ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝          │
│                                                                 │
│           AI-POWERED MEDICAL CLASSIFICATION SYSTEM              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.13+-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge)

</div>

---

# 🩺 **Skin Disease AI Classifier**
### ⭐ NTI Graduation Project – Ensemble Deep Learning + Explainable AI

> **An advanced medical AI system** that diagnoses **4 types of skin diseases** using **Weighted Ensemble Learning**, **Transfer Learning**, and **Grad-CAM Explainability**.  
> Deployed as an **interactive Streamlit Web App** for real-world clinical demonstration.

---

## 🌐 **Live Demo**

<div align="center">

### 🚀 **Try the App Now:**  
[![Streamlit App](https://img.shields.io/badge/🔴_LIVE_DEMO-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://skin-disease-classification-3eamqxyc2hqdyqcpfuttba.streamlit.app/)

**Features:**
- ✅ **Instant AI Diagnosis** (4 disease classes)
- 📊 **Probability Distribution** with visual charts
- 🔥 **Grad-CAM Heatmap** showing model attention
- 🩺 **Medical-grade explainability** for clinical trust
- ⚡ **Real-time inference** (< 5 seconds)

</div>

---

## Models Link

- **DENSENET121_model** = https://drive.google.com/file/d/1ZbNmoMJpT9yJ3tEFxEVBAfIAhrOnq8Dx/view?usp=sharing
- **DENSENET169_FT_model** = https://drive.google.com/file/d/1NjR_7DlFqM75segwvexbCFqq0qNlN0NR/view?usp=sharing
- **EFFNETB3_model** = https://drive.google.com/file/d/1KU9JpiXdfW34A2L8y8Sxdz-wJbJdoQ79/view?usp=drive_link

---

## 📋 **Table of Contents**

- [🔍 Overview](#-overview)
- [🌐 Live Demo](#-live-demo)
- [🧠 Supported Classes](#-supported-classes)
- [🚀 Key Features](#-key-features)
  - [Ensemble Learning](#️-1-weighted-ensemble-learning)
  - [Medical Image Preprocessing](#️-2-medical-grade-image-preprocessing)
  - [Explainable AI (Grad-CAM)](#️-3-explainability--ensemble-grad-cam)
  - [Streamlit Web App](#️-4-streamlit-web-app)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🎯 Usage](#-usage)
- [📦 Pre-trained Models](#-pre-trained-models)
- [📚 Dataset](#-dataset)
- [🔬 Technical Details](#-technical-details)
- [⚠️ Medical Disclaimer](#️-medical-disclaimer)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👥 Team](#-team)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🔍 **Overview**

This project implements a **production-ready medical AI system** for automated skin disease classification using state-of-the-art deep learning techniques:

### **Core Technologies:**
- 🧠 **Weighted Ensemble Learning**: Fusion of 3 fine-tuned CNN architectures
- 🔄 **Transfer Learning**: Pre-trained ImageNet weights, fine-tuned on dermatology data
- 🔥 **Ensemble Grad-CAM**: Visual explainability through weighted heatmap fusion
- 🩺 **Medical Preprocessing Pipeline**: Hair removal, color normalization, lesion cropping
- 🌐 **Production Deployment**: Interactive Streamlit web interface
- ⚖️ **Balanced Dataset**: Carefully curated and cleaned 4,800 images

### **Why This Matters:**
Medical AI systems must be both **accurate** and **explainable**. Doctors cannot trust "black box" models. Our **Ensemble Grad-CAM** provides transparent visual evidence, showing exactly where the model detected pathological features – crucial for clinical adoption and patient safety.

---

## 🧠 **Supported Classes**

The system classifies **4 common dermatological conditions**:

<table>
<thead>
  <tr>
    <th>Class</th>
    <th>Description</th>
    <th>Clinical Severity</th>
    <th>Visual Characteristics</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>🔴 Eczema</strong></td>
    <td>Chronic inflammatory skin condition</td>
    <td>Moderate</td>
    <td>Dry, itchy, red patches with rough, cracked texture</td>
  </tr>
  <tr>
    <td><strong>🟠 Psoriasis</strong></td>
    <td>Autoimmune skin disease</td>
    <td>Moderate-High</td>
    <td>Red inflamed patches covered with thick silver scales</td>
  </tr>
  <tr>
    <td><strong>🟢 Benign Tumors</strong></td>
    <td>Non-cancerous skin growths</td>
    <td>Low</td>
    <td>Smooth, well-defined lesions (moles, seborrheic keratoses)</td>
  </tr>
  <tr>
    <td><strong>⚫ Melanoma</strong></td>
    <td>Aggressive malignant skin cancer</td>
    <td><strong>Critical</strong></td>
    <td>Irregular borders, multiple colors, asymmetric shape</td>
  </tr>
</tbody>
</table>

> **⚠️ Critical Note:** Early detection of melanoma increases survival rates to **>90%**. This system aids in screening but **never replaces professional diagnosis**.

---

## 🚀 **Key Features**

### ✔️ **1) Weighted Ensemble Learning**

Instead of relying on a single model, we implement a **weighted ensemble** of 3 carefully selected CNN architectures:

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Architecture</th>
    <th>Test Accuracy</th>
    <th>Ensemble Weight</th>
    <th>Rationale</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>DenseNet169</strong> (Fine-tuned)</td>
    <td>169 layers, Dense connections</td>
    <td><strong>75.35%</strong> 🏆</td>
    <td><strong>55%</strong></td>
    <td>Best individual performer, proven in medical imaging</td>
  </tr>
  <tr>
    <td><strong>DenseNet121</strong></td>
    <td>121 layers, Dense connections</td>
    <td><strong>72.82%</strong></td>
    <td><strong>30%</strong></td>
    <td>Stable baseline, excellent feature extraction</td>
  </tr>
  <tr>
    <td><strong>EfficientNetB3</strong></td>
    <td>Compound scaling, Efficient</td>
    <td>38.71%</td>
    <td><strong>15%</strong></td>
    <td>Diverse architecture perspective, adds robustness</td>
  </tr>
</tbody>
</table>

**Ensemble Formula:**
```python
Final_Prediction = 0.55 × P(DenseNet169) + 0.30 × P(DenseNet121) + 0.15 × P(EfficientNetB3)
