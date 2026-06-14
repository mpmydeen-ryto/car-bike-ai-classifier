from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("car_bike_model.keras")

# Create uploads folder if not exists
os.makedirs("uploads", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = 0

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                prediction="No file selected",
                confidence=0
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                prediction="No file selected",
                confidence=0
            )

        # Save uploaded image
        filepath = os.path.join(
            "uploads",
            file.filename
        )

        file.save(filepath)

        # Load image
        img = load_img(
            filepath,
            target_size=(180, 180)
        )

        # Convert image to array
        img_array = img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Normalize
        img_array = img_array / 255.0

        # Predict
        prediction_result = model.predict(img_array)

        print("Raw Prediction:", prediction_result)

        class_names = ["bike", "car"]

        prediction = class_names[
            np.argmax(prediction_result)
        ]

        confidence = round(
            float(np.max(prediction_result)) * 100,
            2
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)