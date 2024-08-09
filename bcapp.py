# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:40:25 2024

@author: ADMIN
"""

import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
import streamlit as st
import sys

# Temporarily increase recursion limit
sys.setrecursionlimit(10000)

# Load the trained model safely
try:
    loaded_model = pickle.load(open('C:/Users/ADMIN/Downloads/trained_model2.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Creating a function for prediction
def breast_cancer_prediction(image_path):
    try:
        # Load the image using PIL
        img = Image.open(image_path)
        
        # Convert the image to RGB format
        img_rgb = img.convert('RGB')
        
        # Display the original image
        st.image(img_rgb, caption='Uploaded Image', use_column_width=True)
        
        # Resize the image to match the model's input shape
        resized_img = img_rgb.resize((256, 256))
        
        # Convert the image to a numpy array and normalize it
        img_array = np.array(resized_img) / 255.0
        
        # Expand dimensions to match the model's expected input
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        yhat = loaded_model.predict(img_array)

        # Define class labels
        class_labels = {0: 'benign', 1: 'malignant'}

        # Interpret the prediction
        if yhat > 0.5:
            return f'Predicted class is {class_labels[1]}'
        else:
            return f'Predicted class is {class_labels[0]}'
    
    except Exception as e:
        return f"Error during prediction: {e}"

# Main function for Streamlit app
def main():
    # Set the title of the web app
    st.title('Breast Cancer Prediction Web App')
    
    # Add header image
    st.image('https://www.yourhealth.net.au/wp-content/uploads/2017/09/pink-ribbon-for-breast-cancer-awareness.jpg', use_column_width=True)
    
    # Sidebar image
    st.sidebar.image('https://illinoiscancercare.com/wp-content/uploads/2020/10/ILCC_BreastCancerAwareness.jpg', use_column_width=True)
    
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])

    # Collect additional information
    name = st.text_input('Enter patient name:')
    age = st.text_input('Enter Age')
    doctor = st.text_input('Enter Doctor name')
    resolution = st.text_input('Enter image resolution (e.g., x40, X100, X400):')

    # Initialize the diagnosis result
    diagnosis = ''

    # Perform the prediction if the button is clicked
    if st.button('Breast Cancer Prediction Result'):
        if uploaded_file is not None:
            # Save the uploaded image to a temporary location
            image_path = f"temp_image.{uploaded_file.type.split('/')[-1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Make a prediction and display the result
            diagnosis = breast_cancer_prediction(image_path)
            st.success(diagnosis)
        else:
            st.error("Please upload an image file to proceed.")

if __name__ == '__main__':
    main()