import math
import pickle
import joblib
import string
import numpy as np
import threading

from flask import Flask, request, jsonify
import tensorflow as tf

# =====================
# Config
# =====================
MODEL_PATH = "best_model_epoch_40.keras"
CHAR2IDX_PATH = "char2idx.pkl"
SCALER_PATH = "scaler.pkl"
MAX_LEN = 200
ALL_CHARS = string.printable
THRESHOLD = 0.5  # tuned threshold for recall vs precision

# =====================
# Load Model & Artifacts
# =====================
def focal_loss(alpha=0.6, gamma=2.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        return -tf.reduce_mean(focal_weight * tf.math.log(p_t))
    return focal_loss_fixed

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH,
                                   custom_objects={"focal_loss_fixed": focal_loss(0.9, 1.5)})

print("Loading character index...")
with open(CHAR2IDX_PATH, "rb") as f:
    char2idx = pickle.load(f)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

_model_lock = threading.Lock()

# =====================
# Text Processing Functions
# =====================
def encode_text(s):
    ids = [char2idx.get(c, 0) for c in s[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

# =====================
# Numeric Feature Functions
# =====================
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())

def possible_combinations(s: str) -> float:
    charset_size = 0
    if any(c.islower() for c in s): charset_size += 26
    if any(c.isupper() for c in s): charset_size += 26
    if any(c.isdigit() for c in s): charset_size += 10
    if any(c in string.punctuation for c in s): charset_size += len(string.punctuation)
    return math.log2(charset_size ** len(s)) if charset_size > 0 else 0.0

def extract_features(text: str):
    length = len(text)
    return [
        shannon_entropy(text),
        possible_combinations(text),
        length,
        sum(c.isdigit() for c in text) / max(1, length),
        sum(c.isupper() for c in text) / max(1, length),
        sum(c.islower() for c in text) / max(1, length),
        sum(c in string.punctuation for c in text) / max(1, length),
        int(any(k in text.lower() for k in ["key", "password", "secret", "token"])),
        int(any(k in text.lower() for k in ["db", "auth", "config"]))
    ]

# =====================
# Flask App
# =====================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify({"results": []})
    
    # Encode text inputs
    X_seq = np.array([encode_text(t) for t in texts], dtype=np.int32)
    
    # Numeric features
    num_feats = np.array([extract_features(t) for t in texts], dtype=float)
    
    # Scale only first 2 features
    scaled_features = np.zeros_like(num_feats)
    scaled_features[:, 0:2] = scaler.transform(num_feats[:, 0:2])
    scaled_features[:, 2:] = num_feats[:, 2:]
    num_feats = scaled_features
    
    # Run prediction
    with _model_lock:
        probs = model.predict({"text_input": X_seq, "num_input": num_feats}, verbose=0).flatten()
    
    # Build response
    results = []
    for text, prob in zip(texts, probs):
        pred = int(prob > THRESHOLD)
        results.append({
            "text": text,
            "prediction": pred,
            "probability": float(prob)
        })
    
    return jsonify({"results": results})

# =====================
# Run server
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
