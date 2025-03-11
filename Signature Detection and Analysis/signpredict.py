import os
import cv2
import numpy as np
import pickle
import network
import preprocessor
from PIL import Image

# Paths for input and output
SIGNATURE_PATH = "signatures/test_signature.png"
PROCESSED_PATH = "processed_signatures/processed_test_signature.png"
FEATURES_PATH = "features/processed_test_signature.npy"
MODEL_PATH = "model/trained_model.pkl"

def enhance_and_crop():
    """Enhance and crop the signature before classification."""
    os.makedirs("processed_signatures", exist_ok=True)

    try:
        img = Image.open(SIGNATURE_PATH)
        # Enhance signature
        bw = ImageEnhance.Color(img).enhance(0.0)
        bright = ImageEnhance.Brightness(bw).enhance(2.2)
        contrast = ImageEnhance.Contrast(bright).enhance(2.0)
        img = contrast.convert("RGBA")

        # Remove white background
        datas = img.getdata()
        newData = [(255, 255, 255, 0) if item[0] > 200 and item[1] > 200 and item[2] > 200 else item for item in datas]
        img.putdata(newData)

        # Save enhanced image
        img.save(PROCESSED_PATH, "PNG")
        print(f"Enhanced signature saved to: {PROCESSED_PATH}")

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return False

    return True

def extract_features():
    """Extract features from the processed signature."""
    os.makedirs("features", exist_ok=True)

    try:
        img = cv2.imread(PROCESSED_PATH, 0)  # Load as grayscale
        if img is None:
            print("Error: Cannot load processed image!")
            return False

        features = preprocessor.prepare(img)  # Extract features
        np.save(FEATURES_PATH, features)  # Save as numpy file
        print(f"Features extracted and saved to: {FEATURES_PATH}")

    except Exception as e:
        print(f"Error extracting features: {e}")
        return False

    return True

def classify_signature():
    """Load the trained model and classify the extracted features."""
    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)  # Load trained model
            print("Model loaded successfully.")

        features = np.load(FEATURES_PATH)  # Load extracted features
        features = np.reshape(features, (901, 1))  # Ensure correct input shape

        output = model.feedforward(features)
        prediction = np.argmax(output)  # Get highest probability class (0 = fake, 1 = genuine)

        print(f"Prediction: {'Genuine (1)' if prediction == 1 else 'Fake (0)'}")
        return prediction

    except Exception as e:
        print(f"Error during classification: {e}")
        return None

if __name__ == "_main_":
    print("\nStarting signature prediction...")

    if enhance_and_crop() and extract_features():
        classify_signature()