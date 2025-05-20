import os
import pickle
import tempfile
import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# importamos utils comunes
import sys
sys.path.append('../')
from shared_elements.lstm import SignLSTM
from shared_elements.shared_utils import detect_active_window
from shared_elements.shared_utils import fix_length
from shared_elements.shared_utils import get_holistic_model
from shared_elements.shared_utils import extract_keypoints_from_video
from shared_elements.shared_utils import compute_hand_fraction

# ─── CONFIG ────────────────────────────────────────────────────────────────────

SHARED_FOLDER = os.path.join(os.path.dirname(__file__), '../shared_elements')
MODEL_PATH    = os.path.join(SHARED_FOLDER, 'best_model.pt')
LABEL_PATH    = os.path.join(SHARED_FOLDER, 'label_encoder.pkl')
MEAN_PATH     = os.path.join(SHARED_FOLDER, 'X_mean.npy')
STD_PATH      = os.path.join(SHARED_FOLDER, 'X_std.npy')

# longitus fija y dims
TARGET_LENGTH  = 73
INPUT_DIM      = 126
NUM_CLASSES    = 15

# ─── CARGA ARTEFACTOS ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Modelo
model = SignLSTM(input_dim=INPUT_DIM,
                 hidden_dim=128,
                 num_layers=1,
                 num_classes=NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# 2) Label encoder
with open(LABEL_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# 3) Normalización
mean = np.load(MEAN_PATH)  # shape (126,)
std  = np.load(STD_PATH)   # shape (126,)

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    # seq: (73,126)
    return (seq - mean) / std

# 4) MediaPipe Holistic (una sola instancia para toda la app)
holistic = get_holistic_model(static_image_mode=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

# ─── FASTAPI SETUP ───────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    sign: str
    confidence: float

# ─── ENDPOINT /predict ──────────────────────────────────────────────────────────

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # 1) Guardar vídeo temporal
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    video_path = tmp.name

    # 2) Extraer keypoints del vídeo
    keypts = extract_keypoints_from_video(video_path, holistic_model=holistic)
    os.unlink(video_path)  # borramos temp

    # 3) Detectar ventana activa
    t0, t1, duration = detect_active_window(keypts, threshold_ratio=0.1)
    seq_active = keypts[t0:t1+1]  # (duration, 42, 3)

    # 4) Fijar longitud
    seq_fixed = fix_length(seq_active, target_length=TARGET_LENGTH)  # (73,42,3)

    # 5) Aplanar y normalizar
    seq_flat = seq_fixed.reshape(TARGET_LENGTH, -1)  # (73,126)
    seq_norm = normalize_sequence(seq_flat)  # (73,126)

    # 6) Calcular duración normalizada y fracción de manos
    length_feat = np.array([duration / TARGET_LENGTH], dtype=np.float32)   # (1,)
    hand_feat   = compute_hand_fraction(seq_fixed)                         # (1,)

    # 7) Tensores y batch dim
    seq_tensor    = torch.from_numpy(seq_norm[np.newaxis,...]).to(device)     # (1,73,126)
    length_tensor = torch.from_numpy(length_feat[np.newaxis,...]).to(device) # (1,1)
    hand_tensor   = torch.from_numpy(hand_feat[np.newaxis,...]).to(device)   # (1,1)

    # 8) Inferencia
    SIGNS = ['hello', 'bye', 'world', 'thank_you', 'yes', 'no', 'please', 'sorry', 'good', 'bad', 'me', 'you', 'love', 'help', 'stop']

    with torch.no_grad():
        logits = model(seq_tensor, length_tensor, hand_tensor)  # (1,15)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx    = int(probs.argmax())
        sign   = SIGNS[idx]                       # ← mapeo directo a nombre
        confidence = float(probs[idx])

    return {"sign": sign, "confidence": confidence}


# ─── RUN FOR DEV ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
