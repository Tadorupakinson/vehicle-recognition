import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np


st.markdown(
    """
    <style>
    body {
        background-color: #022f5d; /* Set the background color to black */
        color: white; /* Change the font color to white */
    }
    .stApp {
        background-color: #022f5d; /* Background of the main app to black */
        color: white; /* Text color to white */
        
        padding: 20px;
        border-radius: 0px; /* No rounded corners */
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important; /* Make all headers, labels, and paragraphs white */
    }
    .stFileUploader {
        height: 200px; /* Adjust height as needed */
        width: 100%;   /* Adjust width as needed */
        padding: 20px; /* Adjust padding as needed */
        font-size: 20px; /* Adjust font size for better visibility */
        color: white; /* Make file uploader text white */
    }
    .image-section {
        background-color: #1f185a ; /* Blue background for the image section */
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
        text-align: center;
    }
    .about-section {
        background-color: #132F6D ; /* Blue background for about section */
        padding: 15px;
        color: white;
        border-radius: 5px;
        position: fixed; /* Fix the position at the bottom */
        bottom: 0;
        left: 0;
        width: 100%; /* Full width */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Set up header for the Streamlit app
st.header('Vehicle Classification CNN Model')

# List of vehicle names
vehicle_names = [
    'airplane', 'ambulance', 'bicycle', 'boat', 'bus', 'car', 'fire_truck',
    'helicopter', 'hovercraft', 'jet_ski', 'kayak', 'motorcycle', 'rickshaw',
    'scooter', 'segway', 'skateboard', 'tractor', 'truck', 'unicycle', 'van'
]

# Load the trained model
model = tf.keras.models.load_model('vehicle_model.h5')

# Function to classify images
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = (
    f"This is {vehicle_names[np.argmax(result)]}.\n"
    f"The chance of being right is {np.max(result) * 100:.5f}%.")
    return outcome

# File uploader for image input
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

# Ensure upload directory exists
upload_directory = 'upload'
os.makedirs(upload_directory, exist_ok=True)

if uploaded_file is not None:
    
    # Save the uploaded file to the 'upload' directory
    file_path = os.path.join(upload_directory, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(uploaded_file, width=700)
    st.markdown("</div>", unsafe_allow_html=True)
    # Classify the uploaded image and display the result
    st.markdown(classify_images(file_path))

st.markdown(
    """
    <div class="about-section">
        <h3>About</h3>
        <p>This is a vehicle classification model that identifies different types of vehicles.</p>
        <p>Make by Phakin Jitsakunchaidet. Student Number is 6510110347</p>
    </div>
    """,
    unsafe_allow_html=True
)