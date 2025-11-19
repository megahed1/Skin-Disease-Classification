███████╗██╗  ██╗██╗███╗   ██╗     ██████╗ ██╗███████╗██╗   ██╗███████╗
██╔════╝██║ ██╔╝██║████╗  ██║    ██╔════╝ ██║██╔════╝██║   ██║██╔════╝
███████╗█████╔╝ ██║██╔██╗ ██║    ██║  ███╗██║█████╗  ██║   ██║███████╗
╚════██║██╔═██╗ ██║██║╚██╗██║    ██║   ██║██║██╔══╝  ██║   ██║╚════██║
███████║██║  ██╗██║██║ ╚████║    ╚██████╔╝██║██║     ╚██████╔╝███████║
╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝     ╚═════╝ ╚═╝╚═╝      ╚═════╝ ╚══════╝
![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![DeepLearning](https://img.shields.io/badge/AI-DeepLearning-yellow)
![KaggleDataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

# 🩺 Skin Disease Classification (Ensemble + Grad-CAM)
### ⭐ NTI Graduation Project – Medical Deep Learning System

This project builds a reliable **medical AI system** that detects **4 types of skin diseases** using **Ensemble Learning**, **Transfer Learning**, and **Explainable AI (Grad-CAM)**.  
The final system is deployed as an interactive **Streamlit Web App**.

---

## 🔍 Overview

The project combines several state-of-the-art deep learning techniques:

- **Ensemble Learning** (weighted fusion of 3 CNN models)
- **Transfer Learning**: DenseNet121, DenseNet169 FT, EfficientNetB3
- **CBAM Attention** (Hybrid model)
- **Grad-CAM Explainability**
- **Hair Removal + Color Normalization + Lesion Cropping**
- **Streamlit Deployment**
- **Balanced custom dataset**

📌 **Dataset:**  
[Kaggle – Skin Disease Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

---

## 🧠 Supported Classes

| Class           | Description                        |
|----------------|------------------------------------|
| **Eczema**     | Dry, itchy, cracked skin           |
| **Psoriasis**  | Red patches with silver scales     |
| **Benign Tumors** | Non-cancerous skin lesions       |
| **Melanoma**   | Serious form of skin cancer        |

---

## 🚀 Features

### ✔️ 1) Ensemble Learning (Final Classifier)

Three models are combined with weighted voting for improved accuracy:

| Model              | Weight |
|-------------------|--------|
| **DenseNet169 (Fine-tuned)** | 0.55   |
| **DenseNet121**              | 0.30   |
| **EfficientNetB3**           | 0.15   |

This ensemble boosts **stability, robustness, and accuracy**.

---

### ✔️ 2) Advanced Image Preprocessing

Applied before training to remove dataset noise:

- ✂️ **Hair Removal** (DullRazor)
- 🎨 **Shades-of-Gray Color Constancy**
- 📦 **Lesion Smart Cropping**
- 🔍 **Blurriness Filter**
- 💡 **Exposure Correction**
- ⚖️ **Full Class Balancing & Cleaned Dataset**

---

### ✔️ 3) Explainability — Ensemble Grad-CAM

Instead of Grad-CAM for one model, we generate:

- Grad-CAM for DenseNet169  
- Grad-CAM for DenseNet121  
- Grad-CAM for EfficientNetB3  

Then combine them using the same ensemble weights → **Unified Heatmap** 🔥  
This gives a **clear medical explanation** showing exactly where the lesion was detected.

---

### ✔️ 4) Streamlit Web App

The deployed interface allows users to:

- Upload a skin lesion image
- View predicted disease
- See probability for each class
- Display **Ensemble Grad-CAM** highlighting the affected region

Perfect for **doctors**, **students**, and **medical AI demos**.

---

## 📁 Project Structure
skin-disease-ensemble/
│── app.py
│── requirements.txt
│── README.md
│── models/
  ├── DENSENET121_model.h5
  ├── DENSENET169_FT_model.h5
  └── EFFNETB3_model.h5
