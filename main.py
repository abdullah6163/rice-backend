from fastapi import FastAPI, UploadFile, File, Body
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os
import requests

app = FastAPI()

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "mobilenetv2_final_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = 256

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# =========================
# API KEYS
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "Rice Leaf Disease API is running"}

# =========================
# PREDICTION API
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_array = np.array(image).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        class_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)

        disease_name = class_names[class_index].replace("_", " ").title()

        return {
            "success": True,
            "disease": disease_name,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =========================
# CHAT API (GROQ)
# =========================
@app.post("/chat")
async def chat(user_message: dict = Body(...)):
    try:
        message = user_message.get("message", "")

        if not message:
            return {
                "success": False,
                "error": "Message is empty"
            }

        if not GROQ_API_KEY:
            return {
                "success": False,
                "error": "GROQ_API_KEY is missing in Render"
            }

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = f"""
You are an expert agriculture assistant for rice farmers.

Give simple, clear, and practical advice about:
- Rice diseases
- Symptoms
- Prevention
- Treatment

Use easy language.

User question:
{message}
"""

        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are an expert in rice plant diseases."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        result = response.json()

        if response.status_code != 200:
            return {
                "success": False,
                "error": result
            }

        reply = result["choices"][0]["message"]["content"]

        return {
            "success": True,
            "reply": reply
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }