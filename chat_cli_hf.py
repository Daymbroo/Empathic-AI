import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "google/gemma-2-2b-it:nebius"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Simpan percakapan supaya bisa kirim konteks
history = []

def simulate_typing(text):
    print("\n🤖 AI:", end=" ", flush=True)
    for c in text:
        print(c, end="", flush=True)
        time.sleep(0.02)
    print("\n")

def chat_with_ai(user_input):
    global history
    history.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Kamu adalah teman AI yang empatik dan suportif. Balas dengan hangat dan solusi realistis dalam bahasa Indonesia."},
            *history  # kirim seluruh percakapan
        ],
        "max_tokens": 200,
        "temperature": 0.85,
        "top_p": 0.9
    }

    try:
        print("\n📡 [DEBUG] Mengirim ke Hugging Face Router...")
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=45)
        print("📌 [DEBUG] Status:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            ai_reply = data["choices"][0]["message"]["content"].strip()
            history.append({"role": "assistant", "content": ai_reply})
            simulate_typing(ai_reply)
        elif response.status_code == 401:
            print("❌ Token invalid atau environment variable HF_TOKEN belum diset.")
        elif response.status_code == 403:
            print("🚫 Model ini masih gated, pastikan kamu sudah klik 'Agree to terms'.")
        else:
            print("⚠️ Server error:", response.text)
    except Exception as e:
        print("💥 Error:", str(e))

def chat_loop():
    print("💬 Empathic Chat CLI (Router API Gemma 2B)")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        user_input = input("🧍 Kamu: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Sampai jumpa! Semoga harimu tenang 🌙")
            break
        if len(user_input) == 0:
            continue
        chat_with_ai(user_input)

if __name__ == "__main__":
    chat_loop()
