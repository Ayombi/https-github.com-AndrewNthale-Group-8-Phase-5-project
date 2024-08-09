# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:48:14 2024

@author: ADMIN
"""

import numpy as np
import pickle
import cv2  # Import OpenCV to handle image processing
import tensorflow as tf
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/ADMIN/OneDrive/Desktop/Data-Science/Breast_cancer/trained_model2.sav', 'rb'))

# Creating a function for prediction
def breast_cancer_prediction(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Check if image loading is successful
    if img is None:
        return "Error: Image not loaded properly."
    
    # Convert BGR (OpenCV format) to RGB (TensorFlow format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the model's input shape
    resized_img = tf.image.resize(img_rgb, (256, 256))
    
    # Normalize and expand dimensions to match the model's expected input
    img_array = np.expand_dims(resized_img / 255.0, axis=0)
    
    # Make the prediction
    yhat = loaded_model.predict(img_array)
    
    # Define class labels
    class_labels = {0: 'benign', 1: 'malignant'}
    
    # Interpret the prediction
    if yhat > 0.5:
        return f'Predicted class is {class_labels[1]}'
    else:
        return f'Predicted class is {class_labels[0]}'

# Main function for Streamlit app
def main():
    # Set the title of the web app
    st.title('Breast Cancer Prediction Web App')
    
    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
    
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