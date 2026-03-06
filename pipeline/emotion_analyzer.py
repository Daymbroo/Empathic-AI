import os
import requests
import onnxruntime as ort
import numpy as np
from scipy.special import softmax
from transformers import XLMRobertaTokenizerFast

# =========================
# PATH & CONFIG
# =========================
MODEL_DIR = "emotion_onnx"
MODEL_ONNX = os.path.join(MODEL_DIR, "xlmr_emotion_model_final.onnx")
MODEL_DATA = os.path.join(MODEL_DIR, "xlmr_emotion_model_final.onnx.data")

HF_BASE = "https://huggingface.co/Felik4949/xlmr-emotion-onnx/resolve/main"
URL_ONNX = f"{HF_BASE}/xlmr_emotion_model_final.onnx"
URL_DATA = f"{HF_BASE}/xlmr_emotion_model_final.onnx.data"

LABEL_NAMES = [
    "anger",
    "fear",
    "happy",
    "harassment",
    "love",
    "neutral",
    "racist",
    "sadness",
    "violence",
]

# =========================
# DOWNLOAD HELPER
# =========================
def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"📦 File already exists: {dest}")
        return

    print(f"🔽 Downloading {os.path.basename(dest)} ...")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✅ Saved to {dest}")


# =========================
# DOWNLOAD MODEL FILES
# =========================
download_file(URL_ONNX, MODEL_ONNX)
download_file(URL_DATA, MODEL_DATA)

# =========================
# LOAD TOKENIZER
# =========================
print("🔤 Loading tokenizer...")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
print("✅ Tokenizer loaded")

# =========================
# LOAD ONNX MODEL
# =========================
print(f"📦 Loading ONNX model: {MODEL_ONNX}")

if not os.path.exists(MODEL_ONNX):
    raise RuntimeError(f"ONNX model not found: {MODEL_ONNX}")

session = ort.InferenceSession(
    MODEL_ONNX,
    providers=["CPUExecutionProvider"],
)

print("🧠 ONNX model loaded successfully")


# =========================
# UTILITY FUNCTIONS
# =========================
def softmax_with_temp(logits, temp=2.0):
    logits = logits / temp
    return softmax(logits)


# =========================
# MAIN PREDICTION FUNCTION
# =========================
def predict_emotion(text: str):
    if session is None:
        raise RuntimeError("ONNX session not loaded")

    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    # Tokenize input
    tokens = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )

    ort_inputs = {
        session.get_inputs()[0].name: tokens["input_ids"].astype(np.int64),
        session.get_inputs()[1].name: tokens["attention_mask"].astype(np.int64),
    }

    # Run inference
    outputs = session.run(None, ort_inputs)

    probs = softmax_with_temp(outputs[0][0])

    idx = int(np.argmax(probs))
    label = LABEL_NAMES[idx]
    confidence = float(probs[idx])

    return {
        "emotion": label,
        "confidence": confidence,
        "text": text,
    }
