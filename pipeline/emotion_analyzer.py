import os, requests, onnxruntime as ort, numpy as np
from scipy.special import softmax
from transformers import XLMRobertaTokenizer

MODEL_DIR = "emotion_onnx"
MODEL_ONNX = os.path.join(MODEL_DIR, "xlmr_emotion_model_final.onnx")
MODEL_DATA = os.path.join(MODEL_DIR, "xlmr_emotion_model_final.onnx.data")

# URL dari Hugging Face
HF_BASE = "https://huggingface.co/Felik4949/xlmr-emotion-onnx/resolve/main"
URL_ONNX = f"{HF_BASE}/xlmr_emotion_model_final.onnx"
URL_DATA = f"{HF_BASE}/xlmr_emotion_model_final.onnx.data"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"🔽 Downloading {os.path.basename(dest)} ...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved to {dest}")

# download dua file
download_file(URL_ONNX, MODEL_ONNX)
download_file(URL_DATA, MODEL_DATA)

# load onnx
session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
print("🧠 ONNX model loaded successfully!")

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
label_names = ["anger","fear","happy","harassment","love","neutral","racist","sadness","violence"]
best_params = [0.39,0.67,0.99]

def softmax_with_temp(logits,temp=2.0):
    logits = logits/temp
    return softmax(logits)

def fuzzy_intensity(score,params):
    low,mid,high=params
    if score<low: return "rendah"
    elif score<mid: return "sedang"
    else: return "tinggi"

def predict_emotion(text):
    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {session.get_inputs()[0].name: tokens["input_ids"].astype(np.int64)}
    outputs = session.run(None, ort_inputs)
    probs = softmax_with_temp(outputs[0][0])
    idx = np.argmax(probs)
    label = label_names[idx]
    return label, float(probs[idx])
