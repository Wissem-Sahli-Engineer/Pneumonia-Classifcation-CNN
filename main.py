# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image
from utils import classify

"""
source .venv/bin/activate && streamlit run main.py
"""
# App Tittle
st.title("Lunges Diseases Classficiation")

# Header
st.header("Please upload a chest X-Ray image")

# Files uploading and support?
file = st.file_uploader('',type=["jpeg","jpg","png","webp","svg","gif"])

# Caching the model so Streamlit doesn't reload it on every button click
MODEL_PATH = "models/best_xray_model.keras"

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Display Image
if file is not None :
    img = Image.open(file).convert('RGB')
    st.image(img, use_column_width = True)

    with st.spinner("Analyzing X-Ray..."):
        results = classify(model, img, DISEASE_CLASSES)

    # Display prediction results
    st.subheader("Diagnostic Probabilities:")
    for disease, prob in results:
        st.write(f"**{disease}**: {prob * 100:.1f}%")
        st.progress(float(prob))
