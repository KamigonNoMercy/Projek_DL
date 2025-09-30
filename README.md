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

The training and fine-tuning process is documented in a Jupyter notebook (coming soon).  
For now, you can explore the dataset and training pipeline directly on Roboflow.

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
