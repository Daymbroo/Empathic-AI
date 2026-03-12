from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from pipeline.emotion_analyzer import load_model

#test
try:
    from pipeline.emotion_analyzer import predict_emotion
except Exception:
    def predict_emotion(text):
        return {"emotion": "mock", "confidence": 0.0}
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
    emotion: Optional[str] = "neutral"
    feedback_text: Optional[str] = None

# =========================
# ENDPOINTS
# =========================

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/chat")
def chat(entry: ChatRequest):
    history = entry.history or []

    # pastikan history aman
    safe_history = [
        {"role": m.role, "content": m.content}
        for m in history
        if m.role and m.content
    ]

    result = chatbot_hf_api.generate_chat_response(
        entry.user_id,
        entry.text,
        safe_history
    )

    # FIX: chatbot_hf_api returns {"reply": "text", ...} dict
    # We need to unwrap it so the final response is {"reply": "string"}
    if isinstance(result, dict):
        reply_text = result.get("reply", "Maaf, aku tidak bisa merespon saat ini 😔")
    else:
        reply_text = str(result)

    return {"reply": reply_text}


# FIX: Changed from `analyze(text: str)` to `analyze(entry: AnalyzeRequest)`
# so it accepts JSON body from Flutter app instead of query parameter
@app.post("/analyze")
def analyze(entry: AnalyzeRequest):
    try:
        result = predict_emotion(entry.text)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "text": entry.text
        }


@app.post("/feedback")
def feedback(entry: FeedbackRequest):
    response = reflection_memory.user_feedback(
        entry.user_id,
        entry.emotion,  # FIX: use emotion from request instead of hardcoded
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
