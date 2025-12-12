import os
import requests
import onnxruntime as ort
import numpy as np
from scipy.special import softmax
from transformers import XLMRobertaTokenizer

# === Konfigurasi ===
MODEL_DIR = "emotion_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "xlmr_emotion_model_final.onnx")
HF_MODEL_URL = "https://huggingface.co/Felik4949/xlmr-emotion-onnx/resolve/main/xlmr_emotion_model_final.onnx"

# === Download model jika belum ada ===
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading ONNX model from Hugging Face...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    r = requests.get(HF_MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded to:", MODEL_PATH)

# === Load model ===
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
print("🧠 ONNX model loaded successfully!")

# === Tokenizer ===
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

label_names = ["anger", "fear", "happy", "harassment", "love", "neutral", "racist", "sadness", "violence"]
best_params = [0.39, 0.67, 0.99]

def softmax_with_temp(logits, temp=2.0):
    logits = logits / temp
    return softmax(logits)

def fuzzy_intensity(score, params):
    low, mid, high = params
    if score < low:
        return "rendah"
    elif score < mid:
        return "sedang"
    else:
        return "tinggi"

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {session.get_inputs()[0].name: inputs["input_ids"].astype(np.int64)}
    outputs = session.run(None, ort_inputs)
    probs = softmax_with_temp(outputs[0][0])
    idx = np.argmax(probs)
    label = label_names[idx]
    return label, float(probs[idx])
