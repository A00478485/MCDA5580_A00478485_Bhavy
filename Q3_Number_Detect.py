import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.applications.mobilenet_v2 import preprocess_input


# Load the pre-trained model
def load_pretrained_model():
    model_path = "digit_classifier_model.keras"  # Update with your model path
    return load_model(model_path)


model = load_pretrained_model()


# Function to preprocess and classify the uploaded image
def predict_digit(image):
    # Resize image to square size
    resized_image = image.resize((28, 28))

    # Convert image to grayscale
    grayscale_image = resized_image.convert('L')

    # Convert image to numpy array
    img_array = np.array(grayscale_image)

    # Reshape array to match model input shape
    img_array = img_array.reshape((1, 28, 28, 1))

    # Normalize the image data
    img_array = img_array.astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    return prediction


# Main function
def main():
    st.title("Digit Classifier")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction when button is clicked
        if st.button("Classify"):
            with st.spinner('Classifying...'):
                # Predict digit
                digit = predict_digit(image)

                # Display prediction
                st.success(f"Predicted Digit: {digit}")


if __name__ == "__main__":
    main()
