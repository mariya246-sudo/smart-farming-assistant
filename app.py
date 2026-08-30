
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Permanent model location
MODEL_PATH = "/content/drive/MyDrive/Smart_Farming_Assistant/crop_disease_model.keras"

# Dataset path
DATASET = "/content/fast_dataset"

# Load model
model = load_model(MODEL_PATH)

# Get class names
class_names = sorted([
    folder for folder in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, folder))
])


def get_risk_advice(disease):

    disease = disease.lower()

    if "healthy" in disease:
        return (
            "LOW 🟢",
            "Plant appears healthy. Continue regular monitoring and proper watering."
        )

    if "blight" in disease:
        return (
            "HIGH 🔴",
            "Remove infected leaves, avoid excess moisture and take suitable crop-protection measures."
        )

    if "spot" in disease:
        return (
            "MEDIUM 🟡",
            "Remove affected leaves, improve air circulation and avoid overhead watering."
        )

    if "mildew" in disease:
        return (
            "MEDIUM 🟡",
            "Improve ventilation, keep leaves dry and monitor the crop regularly."
        )

    if "rust" in disease:
        return (
            "MEDIUM 🟡",
            "Remove infected leaves and maintain good air circulation."
        )

    if "virus" in disease:
        return (
            "HIGH 🔴",
            "Remove severely infected plants and control insect vectors."
        )

    return (
        "MEDIUM 🟡",
        "Monitor the crop closely and consult an agricultural expert if symptoms increase."
    )


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            image = Image.open(file).convert("RGB")
            image = image.resize((160, 160))

            img = np.array(image)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)

            index = np.argmax(prediction[0])

            predicted_class = class_names[index]
            confidence = prediction[0][index] * 100

            parts = predicted_class.split("___")

            crop = parts[0].replace("_", " ")

            if len(parts) > 1:
                disease = parts[1].replace("_", " ")
            else:
                disease = predicted_class.replace("_", " ")

            risk, advice = get_risk_advice(disease)

            result = {
                "crop": crop.title(),
                "disease": disease.title(),
                "confidence": round(float(confidence), 2),
                "risk": risk,
                "advice": advice
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
