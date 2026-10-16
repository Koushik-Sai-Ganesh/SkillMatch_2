"""
SkillMatch - Monitoring Module
Lifecycle stage: monitored production -> retraining loop (M1, M6)

Reads the prediction log written by app/serve.py and computes a very
simple data-drift signal: the mean of each numeric input field seen in
production vs. the mean seen in the training data. A large deviation on
any field is flagged as a drift candidate, which in a real deployment
would trigger a drift-triggered retraining job (M6).

This is intentionally lightweight (a teaching-scale stand-in for a tool
like Evidently / a Feast-backed monitoring pipeline), but demonstrates
the concept end-to-end: predictive performance and data-drift monitoring
closing the loop back to retraining.
"""

import json
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "attrition_clean.csv"
LOG_PATH = Path(__file__).resolve().parent.parent / "reports" / "prediction_log.jsonl"
DRIFT_REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "drift_report.json"

NUMERIC_CHECK_FIELDS = [
    "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
    "DistanceFromHome", "JobSatisfaction",
]

DRIFT_THRESHOLD_PCT = 0.20  # flag if production mean deviates >20% from training mean


def load_production_inputs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            rows.append(rec["input"])
    return pd.DataFrame(rows)


def run():
    train_df = pd.read_csv(DATA_PATH)
    prod_df = load_production_inputs()

    if prod_df.empty:
        print("No production predictions logged yet - nothing to monitor.")
        report = {"status": "no_data", "checked_at": pd.Timestamp.now().isoformat()}
    else:
        drift_flags = {}
        for col in NUMERIC_CHECK_FIELDS:
            if col not in prod_df.columns:
                continue
            train_mean = train_df[col].mean()
            prod_mean = prod_df[col].mean()
            pct_change = abs(prod_mean - train_mean) / (abs(train_mean) + 1e-9)
            drift_flags[col] = {
                "train_mean": round(float(train_mean), 3),
                "production_mean": round(float(prod_mean), 3),
                "pct_change": round(float(pct_change), 3),
                "drift_flagged": bool(pct_change > DRIFT_THRESHOLD_PCT),
            }
        any_drift = any(v["drift_flagged"] for v in drift_flags.values())
        report = {
            "status": "drift_detected" if any_drift else "stable",
            "n_predictions_logged": len(prod_df),
            "fields": drift_flags,
            "recommendation": (
                "Trigger retraining pipeline" if any_drift
                else "No retraining needed at this time"
            ),
            "checked_at": pd.Timestamp.now().isoformat(),
        }

    DRIFT_REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
