from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.emotion_analyzer import predict_emotion
from pipeline import mood_trend, reflection_memory

app = FastAPI(title="Empathic AI Companion", version="1.0")

# =========================
# CORS (Flutter wajib)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODELS
# =========================

class AnalyzeRequest(BaseModel):
    text: str
    user_id: str


class FeedbackRequest(BaseModel):
    user_id: str
    action: str              # "accept" | "reject"
    feedback_text: str | None = None


# =========================
# ENDPOINTS
# =========================

@app.post("/analyze")
def analyze_emotion(entry: AnalyzeRequest):
    emotion, probability = predict_emotion(entry.text)

    mood_trend.log_mood(entry.user_id, emotion)

    return {
        "emotion": emotion,
        "probability": probability
    }


@app.get("/mood/{user_id}")
def get_mood_trend(user_id: str):
    trend = mood_trend.analyze_mood_trend(user_id)
    return {
        "user_id": user_id,
        "trend_summary": trend
    }


@app.post("/feedback")
def send_feedback(entry: FeedbackRequest):
    response = reflection_memory.user_feedback(
        user_id=entry.user_id,
        ai_reply_id="latest",
        action=entry.action,
        feedback_text=entry.feedback_text
    )
    return {
        "status": "ok",
        "response": response
    }


@app.get("/")
def home():
    return {"message": "Empathic AI API is running ✅"}
