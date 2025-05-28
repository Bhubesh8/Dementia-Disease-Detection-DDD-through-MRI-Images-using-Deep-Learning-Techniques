# import streamlit as st
# import tensorflow as tf
# from tensorflow import image
# from PIL import Image, UnidentifiedImageError
# import numpy as np

# # Load the trained model
# try:
#     model = tf.keras.models.load_model('Dementia_Model_binary.h5')  # Updated model file
# except (OSError, IOError) as e:
#     st.error("Error loading model. Please ensure the model file is in the correct location.")
#     st.stop()

# # Define a function to predict pneumonia
# def predict_pneumonia(img, model):
#     try:
#         img = img.convert("RGB")  # Ensure the image has 3 channels (RGB)
#         img = img.resize((244, 244))  # Resize the image to match the inpustret shape of the model
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array /= 255.0  # Normalize the image
#         prediction = model.predict(img_array)
#         return prediction[0][0]
#     except Exception as e:
#         st.error(f"Error during prediction: {str(e)}")
#         return None

# # Streamlit web app layout
# st.title("Pneumonia Detection from Chest X-Ray")
# st.write("Upload a chest X-ray image to detect pneumonia.")

# # File uploader
# uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "png", "jpeg"])
# if uploaded_file is not None:
#     try:
#         img = Image.open(uploaded_file)
#         st.image(img, caption='Uploaded Chest X-ray Image', use_column_width=True)

#         if st.button("Predict"):
#             prediction = predict_pneumonia(img, model)
            
#             if prediction is not None:
#                 prediction_percentage = prediction * 100  
                
#                 if prediction > 0.5:
#                     st.success(f"### Result: Pneumonia Positive with {prediction_percentage:.2f}% confidence")
#                 else:
#                     st.success(f"### Result: Pneumonia Negative with {100 - prediction_percentage:.2f}% confidence")
#             else:
#                 st.error("Prediction failed. Please try again.")
#     except UnidentifiedImageError:
#         st.error("Invalid image format. Please upload a valid chest X-ray image.")
        
# import tensorflow  as tf
# print("TensorFlow version:", tf.__version__)

# import PIL
# from PIL import Image
# print(PIL.__version__)

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image  # Ensure this import
import pandas as pd
import os
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("Dementia_Model_binary.h5")

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Streamlit UI
st.title("Dementia Disease Detection")
st.write("Upload an MRI scan image to classify the dementia stage.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    # Ensure the image is in RGB format (3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("\nProcessing...")
    
    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    pred_label = class_labels[pred_class]
    confidence = np.max(prediction) * 100
    
    # Display result
    st.write(f"**Predicted Class:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Show probability scores
    prob_df = pd.DataFrame(prediction, columns=class_labels).T
    prob_df.columns = ['Probability']
    st.bar_chart(prob_df)

