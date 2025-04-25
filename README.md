# 🤟 ASL Gesture Recognition (American Sign Language)

This project enables users to record sign language gestures using a webcam, process them using MediaPipe, and classify the sign in real-time using a trained LSTM model. Built with **FastAPI**, **PyTorch**, and **JavaScript**, it combines deep learning, computer vision, and web technologies in a full-stack ASL recognition system.

## Project Structure
asl-gesture-recognition/ 
├── app/ # FastAPI backend 
│ ├── app.py # Prediction endpoint using PyTorch 
│ ├── index.html
│ ├── index.css
│ ├── index.js
│ ├── model.pth # Trained LSTM model 
│ ├── label_encoder.pkl # Encoded class labels 
│ └── scaler.pkl # Mean/std for normalization 
│
├── code/ 
│ ├── hand_landmarker.task
│ └── train_model.ipynb
│
└── data/ 
  ├── keypoints
  └── videos


---

# Train Your Own Model 
Use train_model.py to:
 - Load .npy keypoints
 - Normalize data
 - Train an LSTM model
 - Save model.pth, label_encoder.pkl, and scaler.pkl

Input Format:
Each .npy file should contain a sequence with shape (T, 75, 3) — 75 keypoints from pose and hands, each with (x, y, z).

## Use 

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn torch mediapipe scikit-learn python-multipart opencv-python

# RUN THE APP 
cd app
# backend
python3 -m uvicorn app:app --reload
# frontend
python3 -m http.server 3000
