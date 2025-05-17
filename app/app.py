import os
import tempfile
import cv2
import numpy as np
import torch
import pickle
import json
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import hashlib
from torch.serialization import add_safe_globals

# modelo
import sys
sys.path.append('../models')
from lstm import SignLSTM
from torch.serialization import add_safe_globals
# registrar clase como segura para deserialización
add_safe_globals([SignLSTM])

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "../models/best_model_lstm.pt"   # modelo seleccionado
ENCODER_PATH = "label_encoder.pkl"            # LabelEncoder
SCALER_PATH = "scaler.pkl"                    # Pickle containing {'mean':…, 'std':…}
SEQUENCE_LENGTH = 60
HAND_LANDMARKS = 21
TOTAL_KEYPOINTS = 2 * HAND_LANDMARKS  # sólo manos

# ─── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # in prod, lock this down
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD ARTIFACTS ─────────────────────────────────────────────────────────────
# Label encoder
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Scaler (dict con 'mean' y 'std')
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
mean = scaler['mean']
std  = scaler['std']

# Modelo PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device).eval()

# Debug: comprobar hash del modelo
model_hash = hashlib.md5(open(MODEL_PATH, 'rb').read()).hexdigest()
print(f"❗️ Model hash: {model_hash}")

# MediaPipe Holistic (igual que en entrenamiento)
import mediapipe as mp
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def extract_keypoint_sequence(frames, save_path=None):
    """
    frames: lista de imágenes BGR
    Devuelve np.ndarray de forma (1, SEQUENCE_LENGTH, TOTAL_KEYPOINTS * 3)
    normalizado igual que en entrenamiento.
    """
    keypoints_list = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Inicializar a ceros
        keypoints = [(0.0, 0.0, 0.0)] * TOTAL_KEYPOINTS
        # Mano izquierda
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                keypoints[i] = (lm.x, lm.y, lm.z)
        # Mano derecha
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                keypoints[HAND_LANDMARKS + i] = (lm.x, lm.y, lm.z)
        keypoints_list.append(keypoints)

    # Padding o truncado
    if len(keypoints_list) < SEQUENCE_LENGTH:
        padding = [[(0.0, 0.0, 0.0)] * TOTAL_KEYPOINTS] * (SEQUENCE_LENGTH - len(keypoints_list))
        keypoints_list.extend(padding)
    else:
        keypoints_list = keypoints_list[:SEQUENCE_LENGTH]

    arr = np.array(keypoints_list)                       # (60, 42, 3)
    seq = arr.reshape(SEQUENCE_LENGTH, -1)               # (60, 126)
    seq = (seq - mean) / std                             # Normaliza una sola vez

    if save_path:
        np.save(save_path, arr)
        print(f"✅ Keypoint sequence saved to {save_path}")
    
    return np.expand_dims(seq, axis=0)                   # (1, 60, 126)

class Prediction(BaseModel):
    sign: str
    confidence: float

# ─── ENDPOINT DE PREDICCIÓN ────────────────────────────────────────────────────
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Guardar vídeo temporal
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    # Extraer frames
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    os.unlink(tmp.name)

    # Procesar keypoints
    save_keypoints_to = "debug_sequence3.npy"
    seq = extract_keypoint_sequence(frames, save_path=save_keypoints_to)

    # Tensor y dispositivo
    tensor = torch.from_numpy(seq.astype(np.float32)).to(device)  # (1,60,126)

    # Inferencia
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx    = int(np.argmax(probs))
        sign   = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

    print("Raw logits:", logits)
    print("Probabilities:")
    for i, p in enumerate(probs):
        label = label_encoder.inverse_transform([i])[0]
        print(f"  {label}: {p:.4f}")
    print("Prediction:", sign, "| Confidence:", confidence)

    return {"sign": sign, "confidence": confidence}

# ─── RUN PARA DESARROLLO ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
