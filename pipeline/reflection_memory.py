# Pipeline Reflections
from collections import defaultdict

user_profile = defaultdict(lambda: {
    "accept": 0, "decline": 0, "skip": 0, "reflect": 0,
    "reflections": []
})

def user_feedback(user_id, emotion, action, feedback_text=None):
    p = user_profile[user_id]
    p[action] += 1
    if feedback_text:
        p["reflections"].append({"emotion": emotion, "text": feedback_text})

    if action == "accept":
        responses = ["Aku senang kamu bisa menerima perasaanmu 🤍", "Hebat banget kamu 🌸"]
    elif action == "decline":
        responses = ["Oke, gak apa-apa ⛈️ Kita bisa coba cara lain nanti ☀️"]
    elif action == "skip":
        responses = ["Oke, kita lanjut kapan-kapan kalau kamu siap 🕊️"]
    elif action == "reflect":
        if feedback_text:
            if any(w in feedback_text.lower() for w in ["lega", "tenang", "syukur"]):
                responses = ["Senang dengarnya 🥰 Kadang menulis bisa jadi obat hati terbaik."]
            elif any(w in feedback_text.lower() for w in ["sedih", "capek", "marah"]):
                responses = ["Aku paham banget 😣 Terima kasih sudah berani berbagi."]
            else:
                responses = ["Terima kasih atas refleksimu 🌿 Aku ikut belajar dari ceritamu."]
        else:
            responses = ["Kamu mau tulis sedikit refleksi hari ini? Aku siap dengerin 🗨"]
    else:
        responses = ["Hmm, aku belum paham maksudmu."]
    return responses[0]