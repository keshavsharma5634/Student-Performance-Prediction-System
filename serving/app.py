from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# 🔥 SMART STUDENT MODEL PATH FINDER ENGINE
try:
    path = None
    possible_paths = [
        "models/student_perf_calibrated.joblib", 
        "../models/student_perf_calibrated.joblib", 
        "student_perf_calibrated.joblib",
        "/opt/render/project/src/models/student_perf_calibrated.joblib"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if path:
        model = joblib.load(path)
        print(f"✅ SUCCESS: Student Performance XGBoost Model Loaded perfectly from '{path}'!")
    else:
        print("⚠️ Direct paths failed. Running deep scan engine on root layout...")
        found = False
        for root, dirs, files in os.walk("."):
            if "student_perf_calibrated.joblib" in files:
                target_path = os.path.join(root, "student_perf_calibrated.joblib")
                model = joblib.load(target_path)
                print(f"✅ SUCCESS: Deep scan found and loaded model from: '{target_path}'")
                found = True
                break
        if not found:
            print("❌ CRITICAL ERROR: student_perf_calibrated.joblib file could not be discovered on system node.")
            
except Exception as e:
    print(f"❌ ERROR During Loading: {e}")

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
    if model is None:
        return {"status": "Error", "error": "Model Pipeline not initialized on server layer."}
        
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
