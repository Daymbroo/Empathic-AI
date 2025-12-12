import json
from pipeline import emotion_analyzer, reflection_memory, mood_trend

uid = "user01"

print("=== 🔬 TESTING PIPELINE EMPATHIC AI ===\n")

# =========================================================
# 1️⃣ EMOTION ANALYZER TEST
# =========================================================
sample_journal = """
Hari ini aku ngerasa capek banget. 
kesal kali aku lek sama si itu kayak sok ngatur kali weh
pokokny emosi kli aku.
"""

print("🧾 Input jurnal:")
print(sample_journal)

result = emotion_analyzer.analyze_journal_entry(sample_journal, uid)
print("\n🎭 Emotion Analyzer Output:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# =========================================================
# 2️⃣ MOOD TREND LOGGER & ANALYZER
# =========================================================
mood_trend.log_mood(uid, result["emotion"])
trend_summary = mood_trend.analyze_mood_trend(uid)

print("\n📊 Mood Trend Summary:")
print(trend_summary)

# =========================================================
# 3️⃣ USER REFLECTION MEMORY
# =========================================================
feedback_text = "aku merasa agak lega setelah menulis, meski masih sedih sedikit."
feedback_action = "reflect"

reply = reflection_memory.user_feedback(
    user_id=uid,
    emotion=result["emotion"],
    action=feedback_action,
    feedback_text=feedback_text
)

print("\n🪞 Reflection Feedback Response:")
print(reply)

# =========================================================
# 4️⃣ OPTIONAL: SAVE RESULT
# =========================================================
log_data = {
    "user_id": uid,
    "emotion_result": result,
    "trend_summary": trend_summary,
    "reflection_reply": reply
}

with open("test_result.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=2, ensure_ascii=False)

print("\n✅ Semua pipeline berhasil diuji.")
print("📂 Hasil tersimpan di: test_result.json")
