# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf

"""
source .venv/bin/activate && streamlit run main.py
"""
# App Tittle
st.title("Lunges Diseases Classficiation")

# Header
st.header("Please upload a chest X-Ray image")

# Files support
st.file_uploader('',type=["JPEG","JPG","PNG","Webp","SVG","GIF"])

# loading the model

MODEL_PATH = "models/best_xray_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# 