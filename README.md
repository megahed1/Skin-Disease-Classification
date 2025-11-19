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

**🩺 Skin Disease Classification (Ensemble + Grad-CAM)**
⭐ NTI Graduation Project – Medical Deep Learning System
🔍 Overview

This project builds a medical-grade AI system for detecting 4 skin diseases using:
Ensemble Learning
Transfer Learning
DenseNet & EfficientNet
CBAM Attention
Grad-CAM Explainability
Streamlit Web Deployment

The system is trained on Kaggle: ISMAILPROMUS Skin Disease Dataset
and delivers both diagnosis and visual explanation.

🧠 Supported Classes
Class	Description
Eczema	Dry, itchy, cracked skin
Psoriasis	Red patches + white scales
Benign Tumors	Non-cancerous lesions
Melanoma	Serious skin cancer
🚀 Features
✔️ 1) Ensemble Learning

Weighted fusion of 3 models:

Model	Weight
DenseNet169 FT	0.55
DenseNet121	0.30
EfficientNetB3	0.15

These weights boost stability + accuracy.

✔️ 2) Advanced Preprocessing
Hair removal (DullRazor)
Color normalization
Smart lesion cropping
Blurriness filter
Exposure correction
Full dataset balancing

✔️ 3) Explainability (Ensemble Grad-CAM)
Grad-CAM is applied to each model → then merged into a single heatmap.
Highlights EXACTLY which part of the skin the model focused on.

✔️ 4) Streamlit Web App
The web interface allows you to:
Upload an image
View predicted disease
View class probabilities
See heatmap showing the lesion
