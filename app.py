import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Vehicle Classification",
    page_icon="🚗",
    layout="centered"
)


# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('vehicle_classification_model.h5')
        return model
    except:
        st.error(
            "Model file not found. Please ensure 'vehicle_classification_model.h5' exists in the project directory.")
        return None


# Prediction function
def predict_vehicle(image, model):
    # Resize image to match model input
    img_resized = tf.image.resize(image, (224, 224))
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, 0)

    # Make prediction
    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Map class to vehicle type
    class_names = {0: 'Bus', 1: 'Car', 2: 'Bike', 3: 'Truck'}
    vehicle_type = class_names[predicted_class]

    return vehicle_type, confidence, prediction[0]


# Main app
def main():
    st.title("🚗 Vehicle Classification App")
    st.markdown("Upload an image to classify vehicles (Bus, Car, Bike, or Truck)")

    # Load model
    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to numpy array
        image_array = np.array(image)

        # Make prediction
        with st.spinner("Classifying..."):
            vehicle_type, confidence, all_predictions = predict_vehicle(image_array, model)

        # Display results
        st.success(f"**Predicted Vehicle: {vehicle_type}**")
        st.info(f"Confidence: {confidence:.2f}%")

    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload an image using the file uploader
    2. The model will classify the vehicle type
    3. View the prediction and confidence score
    4. See probability distribution across all classes
    """)

    st.sidebar.header("Supported Vehicles")
    st.sidebar.markdown("""
    - 🚌 Bus
    - 🚗 Car  
    - 🏍️ Bike
    - 🚛 Truck
    """)


if __name__ == "__main__":
    main()
