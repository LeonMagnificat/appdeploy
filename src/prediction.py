import tensorflow as tf
import numpy as np
import cv2
import sys

# Class names for plant diseases
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the trained model
MODEL_PATH = "/Users/samenergy/Documents/Projects/PlantDiseaseDetection/models/plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    """
    Preprocesses an input image: 
    - Reads the image
    - Resizes it to match model input size (128x128)
    - Normalizes pixel values
    - Expands dimensions for model inference
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (128, 128))  # Resize to model's expected input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

def predict_disease(image_path):
    """
    Predicts the plant disease for a given image.
    Returns the disease name and confidence score.
    """
    try:
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        disease = class_name[predicted_index]
        return disease, confidence
    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prediction.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    disease, confidence = predict_disease(image_path)
    print(f"🔍 Predicted Disease: {disease} (Confidence: {confidence:.2f})")
