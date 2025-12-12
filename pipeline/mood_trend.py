#Model Mood Analyzer
from collections import defaultdict, deque

user_mood_history = defaultdict(lambda: deque(maxlen=10))

mood_messages = {
    "sadness": "Kamu sering merasa sedih akhir-akhir ini 💭. Gak apa-apa, tapi jangan lupa istirahat ya 🌙.",
    "anger": "Sepertinya kamu lagi sering stres. Coba berikan waktu buat diri sendiri 🌿.",
    "joy": "Wah, kamu sering terlihat ceria akhir-akhir ini 🌞. Pertahankan energi positifmu!",
    "fear": "Kamu mungkin sering cemas belakangan ini 😔. Coba tuliskan hal-hal kecil yang bisa kamu kontrol.",
    "neutral": "Perasaanmu stabil, itu tanda keseimbangan batin 🌤️."
}

def log_mood(user_id, emotion):
    user_mood_history[user_id].append(emotion)

def analyze_mood_trend(user_id):
    if not user_mood_history[user_id]:
        return "Belum ada cukup data untuk analisis mood kamu 🪞"
    recent = list(user_mood_history[user_id])
    dominant = max(set(recent), key=recent.count)
    return mood_messages.get(dominant, "Mood kamu unik, jaga keseimbangan dirimu 🌿.")