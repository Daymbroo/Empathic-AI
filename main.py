from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import emotion_analyzer, reflection_memory, mood_trend, chatbot_generator, chatbot_hf_api
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Empathic AI Companion", version="1.0")

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # kalau mau lebih aman: ["http://localhost:50800"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JournalEntry(BaseModel):
    text: str
    history: list | None = None

@app.post("/chat")
def chat(entry: JournalEntry):
    reply = chatbot_hf_api.generate_chat_response(entry.user_id, entry.text, entry.history)
    return reply

@app.post("/analyze")
def analyze_emotion(entry: JournalEntry):
    # Hanya ambil `text` untuk dianalisis
    result = emotion_analyzer.analyze_journal_entry(entry.text)  # Hanya mengirimkan `text` tanpa `user_id`
    mood_trend.log_mood(entry.user_id, result["emotion"])  # Hanya jika perlu menyimpan tren mood
    return result



@app.post("/feedback")
def send_feedback(entry: JournalEntry, action: str, feedback_text: str = None):
    reply = reflection_memory.user_feedback(entry.user_id, "unknown", action, feedback_text)
    return {"feedback_response": reply}

@app.get("/mood/{user_id}")
def get_mood(user_id: str):
    trend = mood_trend.analyze_mood_trend(user_id)
    return {"user_id": user_id, "trend_summary": trend}

@app.get("/")
def home():
    return {"message": "Empathic AI Gemma is running ✅"}





