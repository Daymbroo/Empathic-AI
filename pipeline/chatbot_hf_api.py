import os
import requests
import random
from dotenv import load_dotenv

# =====================================================
# 🔧 Setup Token & Endpoint
# =====================================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN tidak ditemukan. Pastikan sudah diset di file .env")

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "google/gemma-2-2b-it:nebius"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# =====================================================
# 🧠 Fungsi Pembuat Prompt & Generator Respon
# =====================================================
def build_prompt(history, user_text):
    safe_history = history[-5:] if history else []

    context = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in safe_history
    )

    return f"""
Kamu adalah AI yang empatik dan suportif.
Balas dalam bahasa Indonesia dengan nada hangat.
Panjang maksimal 8 kalimat.

{context}
user: {user_text}
assistant:
"""



def generate_chat_response(user_id, user_text, history=[]):
    prompt = build_prompt(history, user_text)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an empathetic assistant that helps users regulate emotions and reflect positively."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.85,
        "top_p": 0.9
    }

    try:
        print("📡 [DEBUG] Sending request to Hugging Face Router...")
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=45)
        print("📌 [DEBUG] Status:", response.status_code)
        print("📌 [DEBUG] Raw response:", response.text)

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                reply = choices[0]["message"]["content"].strip()
                return {"reply": reply}

        # Fallback handling for gated/invalid model
        if response.status_code in [401, 403]:
            return {"reply": "Token Hugging Face kamu tidak valid atau belum mengaktifkan akses inference 😔"}

        # Kalau server gemma lagi down
        if response.status_code == 503:
            return {"reply": "Server model sedang sibuk. Coba lagi sebentar lagi ya 🌙"}

        return {"reply": "Maaf, aku lagi gak bisa merespons sekarang 😔"}

    except Exception as e:
        print("💥 [ERROR] Gagal menghubungi API:", str(e))
        return {"reply": random.choice([
            "Aku paham kamu butuh teman bicara, tapi koneksi ke server lagi bermasalah 😔",
            "Server AI-nya lagi error. Coba lagi beberapa saat lagi ya 💭"
        ])}

