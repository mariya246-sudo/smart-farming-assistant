from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

MODEL_PATH = "crop_disease_model.keras"

model = load_model(MODEL_PATH)

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def get_risk_advice(disease):

    disease = disease.lower()

    if "healthy" in disease:
        return "LOW 🟢", "Plant appears healthy. Continue regular monitoring."

    if "blight" in disease or "virus" in disease:
        return "HIGH 🔴", "Remove infected leaves and take suitable crop-protection measures."

    if "spot" in disease or "mold" in disease or "mildew" in disease:
        return "MEDIUM 🟡", "Remove affected leaves, improve air circulation and avoid overhead watering."

    return "MEDIUM 🟡", "Monitor the crop closely and consult an agricultural expert."


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

            disease = parts[1].replace("_", " ") if len(parts) > 1 else predicted_class

            risk, advice = get_risk_advice(disease)

            result = {
                "crop": crop.title(),
                "disease": disease.replace("(", "").replace(")", "").title(),
                "confidence": round(float(confidence), 2),
                "risk": risk,
                "advice": advice
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
