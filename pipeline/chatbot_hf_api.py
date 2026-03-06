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
    raise ValueError("HF_TOKEN tidak ditemukan. Pastikan sudah diset di Railway Variables atau .env")

API_URL = "https://router.huggingface.co/v1/chat/completions"

# Model utama dan fallback
PRIMARY_MODEL = "google/gemma-2-2b-it"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


# =====================================================
# 🧠 Prompt Builder
# =====================================================

def build_prompt(history, user_text):

    safe_history = history[-5:] if history else []

    context = "\n".join(
        f"{msg.get('role','user')}: {msg.get('content','')}"
        for msg in safe_history
    )

    return f"""
Kamu adalah AI yang empatik dan suportif.
Balas dalam bahasa Indonesia dengan nada hangat.
Panjang maksimal 6 kalimat.

{context}

user: {user_text}
assistant:
"""


# =====================================================
# 🤖 Fungsi Call HF Router
# =====================================================

def call_hf_model(model_name, prompt):

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are an empathetic assistant helping users understand emotions and reflect positively."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    return response


# =====================================================
# 💬 Chat Generator
# =====================================================

def generate_chat_response(user_id, user_text, history=None):

    history = history or []

    prompt = build_prompt(history, user_text)

    try:

        print("📡 Trying PRIMARY MODEL:", PRIMARY_MODEL)

        response = call_hf_model(PRIMARY_MODEL, prompt)

        print("📌 Status:", response.status_code)
        print("📌 Raw:", response.text)

        if response.status_code == 200:

            data = response.json()

            reply = data["choices"][0]["message"]["content"].strip()

            return {
                "user_id": user_id,
                "model": PRIMARY_MODEL,
                "reply": reply
            }

        # =====================================================
        # Fallback model jika gemma gagal
        # =====================================================

        print("⚠️ Gemma gagal, mencoba fallback model...")

        response = call_hf_model(FALLBACK_MODEL, prompt)

        if response.status_code == 200:

            data = response.json()

            reply = data["choices"][0]["message"]["content"].strip()

            return {
                "user_id": user_id,
                "model": FALLBACK_MODEL,
                "reply": reply
            }

        # =====================================================
        # Error Handling
        # =====================================================

        if response.status_code == 401:
            return {"reply": "Token HuggingFace tidak valid 🔑"}

        if response.status_code == 403:
            return {"reply": "Model tidak memiliki akses inference 😔"}

        if response.status_code == 429:
            return {"reply": "API HuggingFace sedang sibuk. Coba lagi sebentar lagi ⏳"}

        if response.status_code == 503:
            return {"reply": "Server model sedang overload. Coba lagi beberapa saat lagi 🌙"}

        return {"reply": "Maaf, aku sedang tidak bisa merespon saat ini 😔"}

    except Exception as e:

        print("💥 Error:", str(e))

        return {
            "reply": random.choice([
                "Aku ingin membantu, tapi koneksi ke server AI sedang bermasalah 😔",
                "Sepertinya server AI sedang error. Coba lagi sebentar ya 💭",
                "Maaf ya, aku lagi tidak bisa merespon sekarang."
            ])
        }
