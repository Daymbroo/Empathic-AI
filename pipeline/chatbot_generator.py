import os
import requests

# pastikan kamu udah set environment variable HF_TOKEN di sistem kamu
# contoh (PowerShell):
# setx HF_TOKEN "hf_your_real_token_here"

API_URL = "https://api-inference.huggingface.co/v1/chat/completions"

payload = {
    "model": "google/gemma-2-2b-it",
    "messages": [
        {"role": "system", "content": "You are an empathetic assistant that responds kindly in Indonesian."},
        {"role": "user", "content": "Aku capek banget hari ini, tolong kasih semangat."}
    ],
    "max_tokens": 200
}
MODEL_NAME = "google/gemma-2-2b-it:nebius"

def generate_chat_response(user_id, user_text, history=None):
    if history is None:
        history = []

    headers = {
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
        "Content-Type": "application/json"
    }

    messages = []
    for h in history[-5:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_text})

    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.8,
        "top_p": 0.9,
    }


    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()
        return {"reply": reply}
    except Exception as e:
        print("⚠️ Gagal panggil API:", e)
        return {"reply": "Maaf, aku lagi gak bisa merespons sekarang 😔"}
