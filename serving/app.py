from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS middleware (yahi hona chahiye upar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("../models/student_perf_calibrated.joblib")

# Input schema
class StudentData(BaseModel):
    prior_gpa: float
    attendance_pct: float
    quiz_avg: float
    assign_avg: float
    midterm: float
    study_hours_wk: float
    on_time_submit_pct: float
    lms_logins_wk: float
    forum_posts: float
    commute_min: float
    gender: str
    school_type: str
    parent_edu: str

@app.post("/predict")
def get_prediction(data: StudentData):
    try:
        features = pd.DataFrame([data.model_dump()])
        risk_probability = float(model.predict_proba(features)[0, 1])
        is_at_risk = bool(risk_probability >= 0.5)

        return {
            "status": "Success",
            "risk_prob": risk_probability,
            "at_risk": is_at_risk
        }
    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}