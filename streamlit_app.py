import os
import random
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import streamlit.components.v1 as components

# -------------------- Streamlit UI Settings --------------------
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

# -------------------- Constants --------------------
class_names = ['calcification BENIGN', 'calcification MALIGNANT', 'mass BENIGN', 'mass MALIGNANT']
image_size = (224, 224)
sample_images_folder = "fixed_samples"

# -------------------- Image Preprocessing --------------------
def process_image(input_file, image_size):
    if isinstance(input_file, str):
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    else:
        img_pil = Image.open(input_file).convert("L")
        img = np.array(img_pil)

    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

# -------------------- Grad-CAM Utilities --------------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if isinstance(img_array, np.ndarray) and img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 1)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


def apply_gradcam(image, heatmap, alpha=0.4):
    # Rescale heatmap to [0, 255]
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)  # Grayscale to RGB
    superimposed_img = heatmap_color * alpha + image
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img

# -------------------- Model and Metrics Loading --------------------
@st.cache_resource
def load_model():
    return keras.models.load_model("my_cnn_model_main.h5")

@st.cache_data
def load_metrics_table():
    return pd.read_pickle("metrics_table.pkl")

def render_custom_table(df):
    styles = """
    <style>
    body {
        margin: 0;
        background-color: #efa9ce;
        color: white;
        font-family: Arial, sans-serif;
    }
    table.custom-table {
        border-collapse: collapse;
        width: 100%;
        color: white;
        table-layout: fixed;
        background-color: transparent;
    }
    table.custom-table th, table.custom-table td {
        border: 1px solid white;
        padding: 8px;
        text-align: center;
        word-wrap: break-word;
    }
    table.custom-table th {
        background-color: #efa9ce;
        color: black;
    }
    </style>
    """
    html_table = df.to_html(classes="custom-table", index=True)
    return styles + f'<div style="padding: 10px;">{html_table}</div>'

model = load_model()
metrics_table = load_metrics_table()

# -------------------- Prediction and Display --------------------
def predict_and_display(input_img):
    image_array = np.expand_dims(input_img, axis=0)
    
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_class = class_names[predicted_index]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Category:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Load original image for display
    input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32)

    # Grad-CAM
    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name)

    overlay = apply_gradcam((input_img * 255).astype("uint8"), heatmap)

    # Toggle between views
    view_choice = st.radio("View", ["Original", "Grad-CAM Overlay"], horizontal=True)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    fig.patch.set_facecolor('#efa9ce')
    ax.set_facecolor('#efa9ce')
    
    if view_choice == "Original":
        ax.imshow(input_img.squeeze(), cmap='viridis')
    else:
        ax.imshow(overlay, cmap='jet')

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig)

# -------------------- App Interface --------------------
if "image_source" not in st.session_state:
    st.session_state.image_source = None

st.title("MammoAI: An AI for Breast Cancer Detection")

st.markdown(
"""
**Welcome to MammoAI!**  
This app will allow you to upload mammogram images and identify wether there is a mass or a calcification present, and whether said feature is benign or malign. To use the app, you can simply upload the image by clicking the "Upload button" or by dragging the photo to the box above. Alternatively, you can click the "Choose Random" button to select one out of 5 possible images from the CBIS-DDSM dataset to check how the app works. 

The main outputs of this app will be the predicted category for the chosen photo, the confidence score of this prediction, and a Gradient-weighted Class Activation Mapping (GradCam) showing the regions of the image that most influenced the model's prediction. We have also included the table below which shows the precision, recall, and F1 scores the app achieved on each of the 4 possible classification categories. 
"""
)

components.html(
    render_custom_table(metrics_table),
    height=240,
    scrolling=True,
)

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.session_state.image_source = "upload"
    st.session_state.uploaded_file = uploaded_file

# Random image button
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
        st.session_state.uploaded_file = None

        prev_path = st.session_state.get("random_path", None)

        # Try up to N times to get a different image
        attempts = 0
        max_attempts = 10
        new_path = prev_path

        while new_path == prev_path and attempts < max_attempts:
            new_path = random.choice(sample_image_paths)
            attempts += 1

        st.session_state.random_path = new_path

# Show results
if st.session_state.image_source == "upload":
    file = st.session_state.uploaded_file
    processed = process_image(file, image_size)
    if processed is not None:
        predict_and_display(processed)
    else:
        st.error("Failed to process uploaded image.")

elif st.session_state.image_source == "random":
    path = st.session_state.random_path
    processed = process_image(path, image_size)
    if processed is not None:
        predict_and_display(processed)
    else:
        st.error(f"Failed to load image from path: {path}")

