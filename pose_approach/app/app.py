import os
import tempfile
import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.pth"              # Your saved PyTorch model weights
ENCODER_PATH = "label_encoder.pkl"    # Your saved LabelEncoder
SCALER_PATH = "scaler.pkl"            # Pickle containing {'mean':…, 'std':…}
SEQUENCE_LENGTH = 30

# ─── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # in prod, lock this down
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD YOUR ARTIFACTS ──────────────────────────────────────────────────────
# 1) Label encoder
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# 2) Scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)  # dict with 'mean' and 'std'

# 3) PyTorch model
class SignLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLSTM(input_size=75*3, hidden_size=128, num_layers=1, num_classes=len(label_encoder.classes_)) # cambiar de acuerdo con la estructura final 
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# 4) MediaPipe Holistic for keypoints
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def extract_keypoint_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # RGB + detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_holistic.process(image)

        # collect 75 landmarks (pose 33 + 21 + 21 hands)
        kp = np.zeros((75, 3), dtype=np.float32)
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                kp[i] = (lm.x, lm.y, lm.z)
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                kp[33 + i] = (lm.x, lm.y, lm.z)
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                kp[33 + 21 + i] = (lm.x, lm.y, lm.z)

        seq.append(kp.reshape(-1))  # flatten to (225,)

    cap.release()
    # pad/truncate
    if len(seq) < SEQUENCE_LENGTH:
        pad = [np.zeros(75*3,)]*(SEQUENCE_LENGTH - len(seq))
        seq.extend(pad)
    else:
        seq = seq[:SEQUENCE_LENGTH]
    return np.array(seq, dtype=np.float32)  # shape (SEQUENCE_LENGTH,225)

class Prediction(BaseModel):
    sign: str
    confidence: float

# ─── PREDICTION ENDPOINT ──────────────────────────────────────────────────────
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # save to temp video file
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    # extract & preprocess
    seq = extract_keypoint_sequence(tmp.name)
    os.unlink(tmp.name)  # remove temp

    # normalize
    seq = (seq - scaler['mean']) / scaler['std']
    tensor = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)  # (1, SEQ, 225)

    # inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        sign = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
    
    print("Raw logits:", logits)
    print("Probabilities:", probs)
    print("Prediction:", sign, "| Confidence:", confidence)


    return {"sign": sign, "confidence": confidence}

# ─── RUN (for dev) ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

