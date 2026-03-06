import os
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import XLMRobertaTokenizerFast
from huggingface_hub import hf_hub_download

# ===============================
# CONFIG
# ===============================

HF_REPO = "Felik4949/xlmr-emotion-onnx"

MODEL_ONNX = "xlmr_emotion_model_final.onnx"
MODEL_DATA = "xlmr_emotion_model_final.onnx.data"

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

# ===============================
# GLOBAL OBJECTS
# ===============================

session = None
tokenizer = None


# ===============================
# LOAD MODEL (called at startup)
# ===============================

def load_model():
    global session, tokenizer

    print("🚀 Loading emotion model from HuggingFace...")

    # download ONNX structure
    onnx_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_ONNX
    )

    # download external tensor weights
    data_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_DATA
    )

    print("📦 ONNX file:", onnx_path)
    print("📦 DATA file:", data_path)

    print("🔤 Loading tokenizer...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    print("✅ Tokenizer loaded")

    print("🧠 Loading ONNX model...")

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    print("✅ ONNX model loaded successfully")


# ===============================
# UTILITY
# ===============================

def softmax_with_temp(logits, temp=2.0):
    logits = logits / temp
    return softmax(logits)


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_emotion(text: str):

    if session is None:
        raise RuntimeError("Model not initialized")

    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized")

    if not text:
        raise ValueError("Input text cannot be empty")

    tokens = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128
    )

    ort_inputs = {
        session.get_inputs()[0].name: tokens["input_ids"].astype(np.int64),
        session.get_inputs()[1].name: tokens["attention_mask"].astype(np.int64)
    }

    outputs = session.run(None, ort_inputs)

    probs = softmax_with_temp(outputs[0][0])

    idx = int(np.argmax(probs))

    return {
        "emotion": LABEL_NAMES[idx],
        "confidence": float(probs[idx]),
        "text": text
    }
