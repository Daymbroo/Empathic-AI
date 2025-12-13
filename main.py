from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from pipeline.emotion_analyzer import predict_emotion
from pipeline import mood_trend, reflection_memory, chatbot_hf_api

app = FastAPI(title="Empathic AI Companion", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SCHEMAS
# =========================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    text: str
    history: Optional[List[Message]] = []

class AnalyzeRequest(BaseModel):
    user_id: str
    text: str

class FeedbackRequest(BaseModel):
    user_id: str
    action: str  # accept / reject
    feedback_text: Optional[str] = None

# =========================
# ENDPOINTS
# =========================

@app.post("/chat")
def chat(entry: ChatRequest):
    history = entry.history or []

    # pastikan history aman
    safe_history = [
        {"role": m.role, "content": m.content}
        for m in history
        if m.role and m.content
    ]

    reply = chatbot_hf_api.generate_chat_response(
        entry.user_id,
        entry.text,
        safe_history
    )

    return {"reply": reply}


@app.post("/analyze")
def analyze(entry: AnalyzeRequest):
    emotion, prob = predict_emotion(entry.text)
    mood_trend.log_mood(entry.user_id, emotion)

    return {
        "emotion": emotion,
        "probability": prob
    }


@app.post("/feedback")
def feedback(entry: FeedbackRequest):
    response = reflection_memory.user_feedback(
        entry.user_id,
        "ai_reply",
        entry.action,
        entry.feedback_text
    )
    return {"result": response}


@app.get("/mood/{user_id}")
def mood(user_id: str):
    trend = mood_trend.analyze_mood_trend(user_id)
    return {"trend": trend}


@app.get("/")
def home():
    return {"message": "Empathic AI Gemma is running ✅"}
