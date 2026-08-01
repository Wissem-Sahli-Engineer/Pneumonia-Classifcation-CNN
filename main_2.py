from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image
from utils import classify, predict_if_cxr
import base64
import io


# source .venv/bin/activate && streamlit run main_2.py

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Lung Disease Classification",
    page_icon="🫁",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------
PANEL_HEIGHT = 380  # px — both panels share this; tuned so title + uploader
                     # + panels + disclaimer all fit one viewport, no scroll

st.markdown(
    f"""
    <style>
        html, body {{
            overflow: hidden !important;
        }}
        [data-testid="stAppViewContainer"] {{
            height: 100vh;
            overflow: hidden;
        }}
        [data-testid="stMain"] {{
            overflow: hidden;
        }}

        /* Remove Streamlit's default chrome that eats vertical space */
        header[data-testid="stHeader"] {{
            height: 0;
            background: transparent;
        }}
        [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            display: none;
        }}

        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0.5rem !important;
            max-width: 1200px;
        }}

        h1 {{
            font-weight: 700;
            margin: 0.3rem 0 0 0;
            font-size: 1.6rem;
        }}
        .app-subtitle {{
            color: #9ca3af;
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }}

        /* Compact the file uploader */
        [data-testid="stFileUploader"] {{
            padding-bottom: 0.4rem;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            padding: 0.4rem !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 14px;
        }}

        .result-card {{
            background-color: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 0.5rem 0.9rem;
            margin-bottom: 0.3rem;
        }}
        .disease-name {{
            font-weight: 600;
            font-size: 0.9rem;
            color: #f3f4f6;
        }}
        .disease-prob {{
            font-weight: 600;
            font-size: 0.9rem;
            float: right;
            color: #93c5fd;
        }}
        .upload-hint {{
            color: #9ca3af;
            font-size: 0.85rem;
        }}
        [data-testid="stCaptionContainer"] {{
            margin-top: 0.3rem;
        }}
        .stProgress > div > div > div > div {{
            border-radius: 6px;
        }}
        /* Image panel: fixed height, overflow HIDDEN (not auto) so it can
           never show a scrollbar, no matter how large the source image is. */
        .img-panel {{
            height: {PANEL_HEIGHT}px;
            overflow: hidden;
            border: 1px solid rgba(250, 250, 250, 0.2);
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 0.5rem;
            box-sizing: border-box;
        }}
        .img-panel img {{
            max-height: calc({PANEL_HEIGHT}px - 2.2rem);
            max-width: 100%;
            object-fit: contain;
            border-radius: 8px;
        }}
        .img-panel .img-caption {{
            color: #9ca3af;
            font-size: 0.8rem;
            margin-top: 0.4rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _img_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🫁 Lung Disease Classification")
st.markdown(
    '<div class="app-subtitle">Upload a chest X-Ray to screen for common thoracic findings.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------
MODEL_PATH_CLASSIFIER = "models/best_xray_model.keras"
MODEL_PATH_DETECTOR = "models/chest_xray_detector.keras"


@st.cache_resource
def load_my_model():
    return (
        tf.keras.models.load_model(MODEL_PATH_CLASSIFIER),
        load_model(MODEL_PATH_DETECTOR),
    )


classifier, detector = load_my_model()

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------
file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpeg", "jpg", "png", "webp", "gif"],
    label_visibility="collapsed",
)

if file is None:
    st.markdown(
        '<p class="upload-hint">Supported formats: JPEG, JPG, PNG, WEBP, GIF. '
        "The image should be a grayscale chest X-ray for best results.</p>",
        unsafe_allow_html=True,
    )
else:
    img = Image.open(file).convert("RGB")

    # Resize a copy for display only — keep aspect ratio, cap size so it
    # never overflows the panel regardless of the uploaded resolution.
    display_img = img.copy()
    display_img.thumbnail((PANEL_HEIGHT, PANEL_HEIGHT))

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        # Fixed-height panel, overflow:hidden — image is embedded as a single
        # base64 <img> inside one markdown call so it's truly nested (not a
        # sibling), and can never trigger a scrollbar.
        img_b64 = _img_to_base64(display_img)
        st.markdown(
            f"""
            <div class="img-panel">
                <img src="data:image/png;base64,{img_b64}" />
                <div class="img-caption">{file.name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        if not predict_if_cxr(img, detector):
            with st.container(height=PANEL_HEIGHT, border=True):
                st.error(
                    "⚠️ **Invalid image.** Please upload a valid grayscale "
                    "chest X-ray image."
                )
        else:
            with st.spinner("Analyzing X-ray..."):
                results = classify(classifier, img, DISEASE_CLASSES)

            # Fixed-height, internally scrollable results panel.
            with st.container(height=PANEL_HEIGHT, border=True):
                st.markdown("**Diagnostic Probabilities**")
                for disease, prob in results:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <span class="disease-name">{disease}</span>
                            <span class="disease-prob">{prob * 100:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(float(prob))

            st.caption(
                "This tool provides an automated screening estimate only "
                "and is not a substitute for professional medical diagnosis."
            )