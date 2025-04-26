# 🤟 ASL Gesture Recognition (American Sign Language)

This project enables users to record sign language gestures using a webcam, process them using MediaPipe, and classify the sign in real-time using a trained LSTM model. Built with **FastAPI**, **PyTorch**, and **JavaScript**, it combines deep learning, computer vision, and web technologies in a full-stack ASL recognition system.

## Project Structure
```
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
│ ├── record_videos.html
│ └── train_model.ipynb
│
└── data/ 
  ├── keypoints
  └── videos

```
---

## Record data 
You can do this ```using record_videos.html``` A simple web app to record ASL gesture videos:
- Select a sign from the dropdown.
- Press spacebar (or click) to start/stop recording.
- Videos are automatically saved as {sign}_{id}.mp4 in your Downloads folder.

**How to Reset Video Indexes**
The app uses localStorage to track video numbering.
To reset counters:
1. Open the page.
2. Open Console (F12 → Console tab).
3. Run: ```localStorage.clear();```
Next recordings will start again from sign_1.mp4.

## Train Your Own Model 
Use train_model.py to:
 - Load .npy keypoints
 - Normalize data
 - Train an LSTM model
 - Save model.pth, label_encoder.pkl, and scaler.pkl

Input Format:
Each .npy file should contain a sequence with shape (T, 75, 3) — 75 keypoints from pose and hands, each with (x, y, z).

## Use App 

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
