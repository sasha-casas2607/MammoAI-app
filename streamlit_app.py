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
import shutil

#Set the colour of the website to pink
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

#Define the possible predicted categories and the input image size
class_names = ['calcification BENIGN', 'calcification MALIGNANT', 'mass BENIGN', 'mass MALIGNANT']
image_size = (224, 224, 3)

#Folder containing the example images from the CBIS-DDSM dataset
sample_images_folder = "fixed_samples"

#Image processing functions
def process_image(image_path, target_size):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        img_pil = Image.open(image_path)
        image = np.array(img_pil)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    image = 255 - image
    
    image = image.astype(np.float32) / 255.0
    return image
    
#GradCam functions
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    superimposed_img = heatmap_color * alpha + image
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img

# Functions to load the table with the app's metrics and the model from their respective pickle files
@st.cache_resource
def load_model():
    return keras.models.load_model("my_cnn_model_main.h5")

@st.cache_data
def load_metrics_table():
    return pd.read_pickle("metrics_table.pkl")

model = load_model()
metrics_table = load_metrics_table()

#Define the styling for the metrics table (white font and borders, wrap long words, keep cell background color same as background and make headers font black)
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
    #Turn the metrics table imported from the pickle file into html
    html_table = df.to_html(classes="custom-table", index=True)
    return styles + f'<div style="padding: 10px;">{html_table}</div>'

# Take the suer-selected image as input and use the model to obtain its predicted category and the confidence score of the prediction
def predict_and_display(input_img, image_source="upload", roi_path=None):
    image_array = np.expand_dims(input_img, axis=0)

    prediction = model.predict(image_array)
    #The predicted class will eb the one with the greatest output
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    
    #The output size will correspond to the model's confidence of the score (softmax being used so outputs add up to 1)
    confidence = np.max(prediction)

    #Display the predicted class and the prediction confidence to the user
    st.subheader("Prediction Results")
    st.write("Predicted Category: "+ predicted_class)
    st.write("Confidence: "+str(round(confidence, 2)))

    #Display text explaining what the images displayed (original, GradCam, and ROI) represent and how to toggle between images
    st.subheader("Visualization")
    st.markdown(
        """
        Using the buttons below you can toggle between two images: the original image you uploaded (visualized with the viridis colour map) and an image
        generated by GradCam sowing the main regions of the image that contributed to the model's prediction. 
        
        To better understand these highlighted regions, we have included the regions of interest for the sample images. To access them, simply click on the
        generate random image and the ROI option will be unlocked. By comparing these ROIs with the GradCam highlighted regions one can get a good idea of the
        model's performance. 
        """
    )

    # The ROI image will be available only if the image was one of the examples introduced from the CBIS-DDSM dataset. 
    if image_source == "random" and roi_path and os.path.exists(roi_path):
        view_options = ["Original", "Grad-CAM Overlay", "ROI"]
    else:
        view_options = ["Original", "Grad-CAM Overlay"]

    #Create a radio button that allows the user to toggle between the image options
    view_choice = st.radio("View", view_options, horizontal=True)

    #Create 3 subplots to display the images and ensure they have the same background colour as the app (also use square image given input is 224 by 224)
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor('#efa9ce')
    ax.set_facecolor('#efa9ce')

    #Display the original, gradCam or ROI image based on user choice
    if view_choice == "Original":
        ax.imshow(input_img[..., 0].squeeze(), cmap='viridis')
    elif view_choice == "Grad-CAM Overlay":
        # Use the GradCam functions to obtain the heatmap for the selected image
        last_conv_layer_name = get_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name)
        overlay = apply_gradcam((input_img * 255).astype("uint8"), heatmap)
        ax.imshow(overlay[..., 0], cmap='jet')
    elif view_choice == "ROI":
        #Load the ROI image using its path
        roi_img = process_image(roi_path, image_size)
        ax.imshow(roi_img[..., 0].squeeze(), cmap='viridis')

    ax.axis("off")
    #Ensure image occupies all allocated space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig)

# The image_source variable stores wether a random image or a user-uploaded image is to be analysed
if "image_source" not in st.session_state:
    st.session_state.image_source = None

#Title display
st.title("MammoAI: An AI for Breast Cancer Detection")

#App explanation display
st.markdown(
"""
**Welcome to MammoAI!**  
This app will allow you to upload mammogram images and identify wether there is a mass or a calcification present, and whether said feature is benign or malign. To use the app, you can simply upload the image by clicking the "Upload button" or by dragging the photo to the box above. Alternatively, you can click the "Choose Random" button to select one out of 5 possible images from the CBIS-DDSM dataset to check how the app works. 

The main outputs of this app will be the predicted category for the chosen photo, the confidence score of this prediction, and a Gradient-weighted Class Activation Mapping (GradCam) showing the regions of the image that most influenced the model's prediction. We have also included the table below which shows the precision, recall, and F1 scores the app achieved on each of the 4 possible classification categories. 
"""
)

#Metrics table display
components.html(
    render_custom_table(metrics_table),
    height=240,
    scrolling=True,
)

#Metrics explanation display
st.markdown(
"""
**Metrics Explanation**

- Precision: the number of times a category was predicted correctly divided by the number of times that category was predicted.
- Recall: the number of times a category was predicted correctly divided by the number of times that category was actually the correct one. 
- F1-score: the harmonic mean of precision and recall, calculated as 2 * (precision * recall) / (precision + recall).
"""
)

#Create a file uplaoder so user can upload their own mammograms for analysis
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.session_state.image_source = "upload"
    st.session_state.uploaded_file = uploaded_file

#Define the paths of all images stored from the CBIS-DDSM dataset as examples
sample_image_paths = [
    os.path.join(sample_images_folder, fname)
    for fname in os.listdir(sample_images_folder)
    if fname.lower().endswith("_image.png")
]

#If the random image button is clicked
if st.button("Use a Random Test Image"):
    st.session_state.image_source = "random"
    st.session_state.uploaded_file = None

    #Ensure that if the button is clicked a second time a new image is shown
    prev_path = st.session_state.get("random_path", None)
    new_path = prev_path
    while new_path == prev_path:
        #Select a random image form those stored in sample_image_paths
        new_path = random.choice(sample_image_paths)

    st.session_state.random_path = new_path

if st.session_state.image_source == "random":
    #The image has the path randomly selected by the code above
    path = st.session_state.random_path
    #Process the image
    processed = process_image(path, image_size)
    #Get path for the ROI image
    base_name = os.path.basename(path).split("_")[0]  
    roi_path = os.path.join(sample_images_folder, f"{base_name}_ROI.png")
    #Send the image to the model for predictions
    predict_and_display(processed, image_source="random", roi_path=roi_path)

# If the user uploads a photo
elif st.session_state.image_source == "upload":
    #The path is dependent on where the user has the image uploaded stored
    file = st.session_state.uploaded_file
    #Process the image
    processed = process_image(file, image_size)
    #Send the image to the model for predictions
    predict_and_display(processed, image_source="upload")
