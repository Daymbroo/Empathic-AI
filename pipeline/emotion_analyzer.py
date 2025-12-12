# =====================================================
# 🧠 EMPATHIC EMOTION ANALYZER (ONNX VERSION)
# =====================================================

import random, numpy as np, onnxruntime as ort
from scipy.special import softmax
from transformers import XLMRobertaTokenizer

# =====================================================
# 1️⃣ LOAD ONNX MODEL
# =====================================================

onnx_model_path = r"E:\felik\empathic_fastapi\emotion_onnx\xlmr_emotion_model_final.onnx"
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Jalankan ONNX session
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]
)

label_names = [
    "anger", "fear", "happy", "harassment",
    "love", "neutral", "racist", "sadness", "violence"
]
best_params = [0.39, 0.67, 0.99]

activity_map = {
    "joy": ["Nikmati momen ini sepenuhnya 🌞", "Tuliskan hal yang kamu syukuri hari ini 🧡"],
    "sadness": ["Tenangkan diri dan journaling sejenak 💭", "Hubungi teman dekat yang kamu percaya 🗨"],
    "anger": ["Tarik napas perlahan, beri ruang untuk tenang 💢", "Coba lepaskan emosi lewat tulisan ✍️"],
    "fear": ["Tuliskan ketakutanmu dan langkah kecil untuk mengatasinya 🪘"],
    "love": ["Kirim pesan positif ke orang yang kamu sayangi ❤️", "Lakukan self-care sederhana hari ini 🌼"],
    "neutral": ["Refleksi singkat: hal baik apa yang terjadi hari ini? 😌"]
}

empathetic_templates = {
    "sadness": [
        "Aku tahu ini gak mudah, tapi kamu udah hebat bisa bertahan 💭.",
        "Kadang gak apa-apa buat nangis, itu tanda kamu masih punya hati ⛈️.",
        "Jangan lupa, badai juga punya ujung. Kamu gak sendirian 🌙."
    ],
    "joy": [
        "Senang banget dengarnya! 🌞 Semoga hari-harimu penuh cahaya seperti ini.",
        "Kamu pantas buat bahagia, terus nikmati setiap momennya 😄."
    ],
    "anger": [
        "Aku ngerti banget perasaanmu 💢 Kadang kita cuma perlu waktu buat reda.",
        "Gak apa-apa marah, asal gak nyakiti dirimu sendiri 🕊️."
    ],
    "fear": [
        "Rasa takut itu wajar, tapi kamu cukup kuat buat melewatinya 🌱.",
        "Pelan-pelan aja ya, kamu gak harus ngelawan semuanya sekaligus 🕊️."
    ],
    "neutral": [
        "Terima kasih sudah berbagi 🌿 Kadang ketenangan juga bentuk kekuatan.",
        "Hari tenang juga penting, tubuhmu sedang berterima kasih 😌."
    ]
}

# =====================================================
# 2️⃣ HELPER FUNCTIONS
# =====================================================

def softmax_with_temp(logits, temp=2.0):
    logits = logits / temp
    return softmax(logits)

def fuzzy_intensity(score, params):
    low, mid, high = params
    if score < low: return "rendah"
    elif score < mid: return "sedang"
    else: return "tinggi"

# =====================================================
# 3️⃣ INFERENCE FUNCTION (ONNX)
# =====================================================

def predict_emotion_onnx(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0][0]
    probs = softmax_with_temp(logits)
    idx = int(np.argmax(probs))
    label = label_names[idx]
    score = float(probs[idx])
    intensity = fuzzy_intensity(score, best_params)
    return label, score, intensity

# =====================================================
# 4️⃣ MAIN ANALYZER PIPELINE
# =====================================================

def generate_empathic_response(label, intensity):
    responses = empathetic_templates.get(label, ["Aku mendengar perasaanmu dengan empati 🧡"])
    base = random.choice(responses)
    tone = {
        "rendah": "",
        "sedang": " Aku bisa rasakan, perasaan itu pasti bercampur ya.",
        "tinggi": " Kamu luar biasa kuat untuk menghadapi semuanya ✨."
    }[intensity]
    return base + tone

def analyze_journal_entry(text, user_id="default_user"):
    label, score, intensity = predict_emotion_onnx(text)
    recommendation = random.choice(activity_map.get(label, ["Ambil waktu untuk dirimu hari ini 🌿"]))
    response = generate_empathic_response(label, intensity)

    return {
        "user_id": user_id,
        "emotion": label,
        "intensity": intensity,
        "score": round(score, 3),
        "recommendation": recommendation,
        "ai_response": response
    }
