import opendatasets as od
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import streamlit as st
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import random
import streamlit.components.v1 as components

st.set_page_config(page_title="Image Classifier", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #efa9ce !important;
    }
    .css-1d391kg {
        background-color: #efa9ce !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class_names = ['calcification BENIGN', 'calcification MALIGNANT', 'mass BENIGN', 'mass MALIGNANT']
image_size = (224, 224)

def process_image(input_file, image_size):
    # input_file can be either a filepath (str) or an uploaded file (BytesIO)

    if isinstance(input_file, str):
        # If it's a path, read with cv2
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    else:
        # If uploaded file, read with PIL, convert to grayscale
        img_pil = Image.open(input_file).convert("L")  # "L" mode = grayscale
        img = np.array(img_pil)

    # Resize with cv2 (expects numpy array)
    img = cv2.resize(img, image_size)

    # Normalize and reshape
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dim
    img = np.expand_dims(img, axis=0)   # Add batch dim

    return img

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model("my_cnn_model.h5")

model = load_model()

@st.cache_data
def load_metrics_table():
    return pd.read_pickle("metrics_table.pkl")

def render_custom_table(df):
    styles = """
    <style>
    body {
        margin: 0;
        background-color: #efa9ce;  /* purple background for the whole iframe */
        color: white;
        font-family: Arial, sans-serif;
    }
    table.custom-table {
        border-collapse: collapse;
        width: 100%;
        color: white;
        table-layout: fixed;
        background-color: transparent;  /* transparent so purple shows through */
    }
    table.custom-table th, table.custom-table td {
        border: 1px solid white;
        padding: 8px;
        text-align: center;
        word-wrap: break-word;
    }
    table.custom-table th {
        background-color: #efa9ce;
        color: black;  /* for contrast on pink header */
    }
    </style>
    """
    html_table = df.to_html(classes="custom-table", index=True)
    # Wrap table in a div container to inherit purple bg and control padding
    return styles + f'<div style="padding: 10px;">{html_table}</div>'

metrics_table = load_metrics_table()

# Predict and display
def predict_and_display(image_array, raw_path):
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_class = class_names[predicted_index]

    
    st.subheader("Prediction Results")
    st.write(f"**Predicted Category:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if isinstance(raw_path, str):
        display_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    else:
        display_img = np.array(Image.open(raw_path).convert("L"))

    display_img = cv2.resize(display_img, (224, 224))  # RESIZE step

    # Plot image with no borders or white space
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    fig.patch.set_facecolor('#efa9ce')  # Set figure background
    ax.set_facecolor('#efa9ce')         # Set axes background
    ax.imshow(display_img, cmap='viridis')
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # REMOVE padding
    st.pyplot(fig)

if "image_source" not in st.session_state:
    st.session_state.image_source = None
    
st.title("MammoAI: An AI for breast cancer detection")

components.html(
    render_custom_table(metrics_table),
    height=240,     # fixed height so table is fully visible (adjust as needed)
    scrolling=True,
)


# Upload image

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.session_state.image_source = "upload"
    st.session_state.uploaded_file = uploaded_file

# Random test image button (from fixed folder)
sample_images_folder = "fixed_samples"
sample_image_paths = [
    os.path.join(sample_images_folder, fname)
    for fname in os.listdir(sample_images_folder)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
]

if st.button("Use a Random Test Image"):
    if not sample_image_paths:
        st.error("No images found in 'fixed_samples/' folder.")
    else:
        st.session_state.image_source = "random"
        st.session_state.random_path = random.choice(sample_image_paths)

# === DISPLAY based on the last action ===
if st.session_state.image_source == "upload":
    file = st.session_state.uploaded_file
    processed = process_image(file, image_size)
    if processed is not None:
        predict_and_display(processed, file)
    else:
        st.error("Failed to process uploaded image.")

elif st.session_state.image_source == "random":
    path = st.session_state.random_path
    processed = process_image(path, image_size)
    if processed is not None:
        predict_and_display(processed, path)
    else:
        st.error(f"Failed to load image from path: {path}")

