# app.py
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv11 Object Detection", layout="centered")

st.title("🔍 YOLOv11 Object Detection App")
st.markdown("Upload an image and let the model detect objects!")

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")  # gunakan file .pt yang sudah kamu upload

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects..."):
        results = model.predict(image)

    for r in results:
        st.image(r.plot(), caption="Detection Result", use_column_width=True)
