"""
SkillMatch - Serving Module
Lifecycle stage: packaged artifact -> served endpoint -> monitored production (M1, M6)

A minimal FastAPI app that loads the packaged artifact (preprocessing +
model, saved together to prevent training-serving skew) and exposes a
/predict endpoint for single-employee attrition-risk scoring.

Run with:
    uvicorn serve:app --reload --port 8000

Then POST to /predict with a JSON body matching EmployeeRecord.
"""

import sys
import time
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from train import add_cluster_feature  # reuse the same clustering step used at train time

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "skillmatch_model.joblib"
LOG_PATH = Path(__file__).resolve().parent.parent / "reports" / "prediction_log.jsonl"

app = FastAPI(title="SkillMatch - Attrition Risk Scoring API")

STATIC_DIR = Path(__file__).resolve().parent / "static"

_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_kmeans = _artifact["kmeans"]


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend UI at the site root."""
    return FileResponse(STATIC_DIR / "index.html")


# Any other static assets (css/js/images) added later can go in app/static
# and will be reachable under /static/...
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class EmployeeRecord(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


def _log_prediction(payload: dict, result: dict, latency_ms: float):
    """Very small monitoring hook (M6): append each prediction + latency
    to a log file. A real deployment would ship this to a metrics/drift
    monitoring system instead of a flat file.
    """
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {"timestamp": time.time(), "latency_ms": latency_ms,
              "input": payload, "output": result}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "champion_model": _artifact["champion_name"]}


@app.post("/predict")
def predict(record: EmployeeRecord):
    start = time.time()
    df = pd.DataFrame([record.model_dump()])

    # engineered ratio features (M2) - must mirror feature_engineering.py exactly
    df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)
    df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"].replace(0, 1)
    df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

    # unsupervised segment feature (M4), using the frozen kmeans from training
    df, _ = add_cluster_feature(df, kmeans=_kmeans)

    proba = float(_model.predict_proba(df)[:, 1][0])
    label = "Leave" if proba >= 0.5 else "Stay"
    result = {"attrition_probability": round(proba, 4), "predicted_label": label}

    latency_ms = (time.time() - start) * 1000
    _log_prediction(record.model_dump(), result, latency_ms)

    return result



# Dashboard data endpoints (same localhost server as the UI)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "attrition_clean.csv"
METRICS_PATH = Path(__file__).resolve().parent.parent / "reports" / "metrics.json"

def _load_dataset():
    return pd.read_csv(DATA_PATH)

def _counts(series):
    return [{"label": str(k), "value": int(v)} for k, v in series.value_counts(dropna=False).items()]

@app.get("/api/overview")
def api_overview():
    df = _load_dataset()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    try:
        events = _read_prediction_events()
    except Exception:
        events = []
    # The cleaned IBM HR dataset stores Attrition as 0 = No (Stay), 1 = Yes (Leave).
    # Also accept the original string labels so the dashboard works with either dataset form.
    attrition_values = df["Attrition"].astype(str).str.strip().str.lower()
    leave_mask = attrition_values.isin(["1", "yes", "true", "leave"])
    return {
        "employees": int(len(df)),
        "attrition_rate": float(leave_mask.mean()),
        "features": int(len([c for c in df.columns if c not in ("Attrition", "EmployeeNumber")])),
        "predictions": len(events),
        "metrics": metrics,
        "champion": metrics.get("champion", _artifact.get("champion_name", "unknown")),
        "train_size": metrics.get("train_size"),
        "val_size": metrics.get("val_size"),
        "test_size": metrics.get("test_size"),
        "base_rate_attrition": metrics.get("base_rate_attrition"),
    }

@app.get("/api/eda")
def api_eda():
    df = _load_dataset()
    # Cleaned data encodes Attrition as 0 = No/Stay and 1 = Yes/Leave.
    # Support both numeric labels and the original Yes/No strings.
    attr = df["Attrition"].astype(str).str.strip().str.lower()
    df = df.copy()
    df["_attrition"] = attr.map({
        "1": "Leave", "yes": "Leave", "true": "Leave", "leave": "Leave",
        "0": "Stay", "no": "Stay", "false": "Stay", "stay": "Stay",
    }).fillna("Unknown")
    by_dept = []
    for name, group in df.groupby("Department", dropna=False):
        by_dept.append({
            "label": str(name),
            "stay": int((group["_attrition"] == "Stay").sum()),
            "leave": int((group["_attrition"] == "Leave").sum())
        })
    age_bins = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65], labels=["18–25", "26–35", "36–45", "46–55", "56–65"], include_lowest=True)
    income_bins = pd.cut(df["MonthlyIncome"], bins=[-1, 3000, 6000, 10000, 15000, float("inf")], labels=["≤3k", "3k–6k", "6k–10k", "10k–15k", "15k+"])
    tenure_bins = pd.cut(df["YearsAtCompany"], bins=[-1, 1, 5, 10, 20, float("inf")], labels=["0–1", "2–5", "6–10", "11–20", "20+"])
    overtime = df["OverTime"].astype(str).map({"Yes": "Overtime: Yes", "No": "Overtime: No"}).fillna(df["OverTime"].astype(str))
    satisfaction = df["JobSatisfaction"].astype(str).map({"1":"Low (1)","2":"Fair (2)","3":"Good (3)","4":"High (4)"}).fillna(df["JobSatisfaction"].astype(str))
    return {
        "attrition": [{"label":"Stay","value":int((df["_attrition"]=="Stay").sum())},{"label":"Leave","value":int((df["_attrition"]=="Leave").sum())}],
        "department": _counts(df["Department"]),
        "attrition_by_department": by_dept,
        "job_role": _counts(df["JobRole"]),
        "age": _counts(age_bins),
        "income": _counts(income_bins),
        "tenure": _counts(tenure_bins),
        "overtime": _counts(overtime),
        "satisfaction": _counts(satisfaction),
    }

def _read_prediction_events():
    events = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            try:
                item = json.loads(line)
                output = item.get("output", {})
                events.append({
                    "timestamp": item.get("timestamp"),
                    "latency_ms": round(float(item.get("latency_ms", 0)), 2),
                    "label": output.get("predicted_label", "Unknown"),
                    "probability": float(output.get("attrition_probability", 0)),
                })
            except (ValueError, TypeError):
                continue
    return events

@app.get("/api/predictions")
def api_predictions():
    items = _read_prediction_events()
    return {"items": items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
