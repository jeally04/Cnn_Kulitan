import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import shutil
import tensorflow as tf

# ==== CONFIG ====
MODEL_PATH = "cnn_kulitanv2.tflite"  # Use your converted TFLite model
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
CLASS_LABELS = [
    'a', 'b', 'bang', 'be', 'bi', 'bi-i', 'bo', 'bu', 'bu-u', 'da', 'dang', 'de', 'di', 'di-i', 'do', 'du', 'du-u',
    'e', 'ga', 'gang', 'ge', 'gi', 'gi-i', 'go', 'gu', 'gu-u', 'i', 'ka', 'kank', 'ke', 'ki', 'ki-i', 'ko', 'ku', 'ku-u',
    'la', 'lang', 'le', 'li', 'li-i', 'lo', 'lu', 'lu-u', 'ma', 'mang', 'me', 'mi', 'mi-i', 'mo', 'mu', 'mu-u',
    'na', 'nang', 'ne', 'nga', 'ngang', 'nge', 'ngi', 'ngi-i', 'ngo', 'ngu', 'ngu-u', 'ni', 'ni-i', 'no', 'nu', 'nu-u',
    'o', 'pa', 'pang', 'pe', 'pi', 'pi-i', 'po', 'pu', 'pu-u', 'sa', 'sang', 'se', 'si', 'si-i', 'so', 'su', 'su-u',
    'ta', 'tang', 'te', 'ti', 'ti-i', 'to', 'tu', 'tu-u', 'u','unknown'
]

# ==== LOAD TFLITE MODEL ====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded successfully.")

# ==== INIT FASTAPI ====
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess image
        img = Image.open(temp_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        # Map "Unidentified" → "Unknown"
        predicted_label = CLASS_LABELS[predicted_class_index]
        if predicted_label == "Unknown":
            predicted_label = "Unknown"

        # Remove temp file
        os.remove(temp_path)

        # Confidence check
        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "detected": False,
                "message": "Unknown",
                "confidence": confidence
            })

        return JSONResponse(content={
            "detected": True,
            "class": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ==== ENTRY POINT ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
