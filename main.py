import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# Function to preprocess a single image
def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array


# Function to load the model and make predictions
def predict_image(model, image, target_size, class_indices):
    img_array = preprocess_image(image, target_size)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    # Invert the class_indices dictionary to get label from index
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label


# Streamlit app
st.title("Image Classification with Trained Model")

model_path = "trained_model.h5"
target_size = (128, 128)  # Replace with the target size used in training

# Example class_indices; should match the indices used during training
class_indices = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18,
                 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27,
                 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}

# Load the model
model = load_model(model_path)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict the class of the image
    predicted_class = predict_image(model, image, target_size, class_indices)
    st.write(f"The predicted class for the image is: {predicted_class}")
