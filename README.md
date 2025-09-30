# 🏙️ ViadV2 Object Detection (YOLOv11 + Streamlit)

This repository contains my **YOLOv11 object detection model** trained on the **[ViadV2 dataset](https://universe.roboflow.com/viad-optics-senior-design/viadv2)** (Roboflow).  
The deployment is done using **Streamlit** and hosted on **Hugging Face Spaces**.

👉 Live Demo: [HuggingFace Space](https://huggingface.co/spaces/Kamigon/object-detection-yolov11)

---

## ✨ Features

- Detects **urban scene objects** (ViadV2 classes: road signs, pedestrians, poles, vehicles, etc.)  
- Supports:
  - 📷 Image Inference
  - 🎥 Video Inference
  - 📹 Webcam Snapshot
- Adjustable:
  - Confidence threshold
  - Inference size (640–1280 px)
  - Class filters
- Option to **save annotated images/videos** with bounding boxes

---

## 🛠️ Tech Stack

- **Model**: YOLOv11 (Ultralytics)  
- **Dataset**: [ViadV2 (Roboflow Universe)](https://universe.roboflow.com/viad-optics-senior-design/viadv2)  
- **Deployment**: Streamlit + Hugging Face Spaces  
- **Runs on CPU/GPU** (depending on available Space hardware)

---

## 📒 Notebook

You can view the **training & fine-tuning notebook** on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1THeHTY-9MKb9LPA2Ve9CvE5FyK8r2Clh?usp=sharing
)

---

## 🚀 Demo

- Hugging Face Space: [Kamigon/object-detection-yolov11](https://huggingface.co/spaces/Kamigon/object-detection-yolov11)  
---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/KamigonNoMercy/viadv2-yolov11-streamlit.git
cd viadv2-yolov11-streamlit
pip install -r requirements.txt
```

## 📜 License

- Code and trained YOLOv11 model weights in this repository are released under the **MIT License**.  
- Dataset: [ViadV2 dataset](https://universe.roboflow.com/viad-optics-senior-design/viadv2) by **Viad Optics Senior Design** on Roboflow Universe, licensed under **CC BY 4.0**.  
- ⚠️ This repository does not include the dataset itself. Please download it directly from Roboflow.

