from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io

app = FastAPI()

MODEL_PATH = "mobilenetv2_final_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = 256

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

@app.get("/")
def home():
    return {"message": "Rice Leaf Disease API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        # IMPORTANT:
        # Do NOT use preprocess_input here if your saved model already contains it.
        img_array = np.array(image).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        class_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)

        disease_name = class_names[class_index].replace("_", " ").title()

        return {
            "success": True,
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "class_index": class_index,
            "raw_prediction": prediction[0].tolist()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }