import os
import json
import numpy as np
import pickle
import mediapipe as mp
import cv2

# modelo mp Holistic 
def get_holistic_model(static_image_mode=True,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5):
    """
    Devuelve una instancia de mp.solutions.holistic.Holistic configurada.
    """
    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

HAND_LANDMARKS = 21
TOTAL_KEYPOINTS = 2 * HAND_LANDMARKS

def extract_keypoints_from_frame(results):
    """
    Dada la salida `results` de Holistic.process(), devuelve un array (K,3)
    de keypoints para manos izquierda y derecha concatenados.
    """
    keypoints = np.zeros((TOTAL_KEYPOINTS, 3), dtype=np.float32)
    # mano izquierda
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            keypoints[i] = (lm.x, lm.y, lm.z)
    # mano derecha
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            keypoints[HAND_LANDMARKS + i] = (lm.x, lm.y, lm.z)
    return keypoints


def extract_keypoints_from_video(video_path, holistic_model=None):
    """
    Procesa un vídeo (cualquiera) y devuelve np.ndarray de shape (T, TOTAL_KEYPOINTS, 3).
    Si se proporciona `holistic_model`, lo reutiliza, sino crea uno.
    """
    cap = cv2.VideoCapture(video_path)
    if holistic_model is None:
        holistic_model = get_holistic_model()
    keypoints_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convierte y procesa
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(frame_rgb)
        keypoints_list.append(extract_keypoints_from_frame(results))
    cap.release()
    return np.stack(keypoints_list, axis=0) if keypoints_list else np.empty((0, TOTAL_KEYPOINTS, 3), dtype=np.float32)



# Detección de ventana activa
def detect_active_window(seq: np.ndarray, threshold_ratio: float = 0.1):
    """
    Dada una secuencia (T, K, 3), detecta inicio y fin del tramo con movimiento.
    threshold_ratio: umbral relativo (p.ej. 0.1 para 10% de la velocidad máxima).
    Devuelve (t_start, t_end, duration).
    """
    # calculamos las velocidades frame a frame (sumatoria de distancias L2 entre keypoints)
    T = seq.shape[0]
    if T < 2:
        return 0, T-1, T
    # diferencias entre frames consecutivos
    diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)
    speeds = diffs.sum(axis=1)
    if speeds.size == 0:
        return 0, T-1, T
    # umbral absoluto basado en la velocidad máxima de la secuencia
    thr = threshold_ratio * speeds.max()
    # índices donde la velocidad supera el umbral
    active = np.where(speeds > thr)[0]
    if active.size == 0:
        # si no hay movimiento fuerte, toma toda la secuencia
        return 0, T-1, T
    t_start = int(active[0])
    t_end   = int(active[-1] + 1)  # +1 por el shift en speeds
    duration = t_end - t_start + 1
    return t_start, t_end, duration


# Ajuste de longitud fija
def fix_length(seq: np.ndarray, target_length: int = 73):
    """
    Dado seq_active (duration frames), devuelve seq_fixed de shape (target_length, K, 3):
    - Si duration >= target_length: center crop.
    - Si duration < target_length: pad con ceros por ambos lados.
    """
    duration = seq.shape[0]
    if duration >= target_length:
        # center crop
        excess = duration - target_length
        start_off = excess // 2
        return seq[start_off:start_off + target_length]
    else:
        # pad equally antes y después
        pad_total = target_length - duration
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        K = seq.shape[1]
        pad_frame = np.zeros((1, K, 3), dtype=seq.dtype)
        left = np.repeat(pad_frame, pad_left, axis=0)
        right = np.repeat(pad_frame, pad_right, axis=0)
        return np.concatenate([left, seq, right], axis=0)


def compute_hand_fraction(seq_keypoints: np.ndarray) -> np.ndarray:
    """
    seq_keypoints: array (T=73, K=42, 3) solo manos.
    Devuelve un array shape (1,) con la media de:
      (presencia_izq + presencia_der) / 2 en cada frame,
    normalizado a [0,1].
    """
    # Detecta presencia de keypoints no nulos en cada frame
    left_present  = np.any(seq_keypoints[:, :21, :2] != 0, axis=(1,2))
    right_present = np.any(seq_keypoints[:, 21:, :2] != 0, axis=(1,2))
    frac_per_frame = (left_present.astype(float) + right_present.astype(float)) / 2.0
    return np.array([frac_per_frame.mean()], dtype=np.float32)
