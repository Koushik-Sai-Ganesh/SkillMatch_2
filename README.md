# SkillMatch — Employee Attrition Risk Prediction System

An end-to-end ML system (raw data → cleaned dataset → engineered features →
trained model → packaged artifact → served endpoint → monitored production)
built on the IBM HR Analytics Employee Attrition & Performance dataset.

## Project structure
```
skillmatch/
├── data/
│   ├── attrition_raw.csv        # original dataset
│   └── attrition_clean.csv      # cleaned dataset (generated)
├── src/
│   ├── data_ingestion.py        # raw -> cleaned dataset
│   ├── feature_engineering.py   # encoding, scaling, derived features
│   ├── train.py                 # training, CV, evaluation, packaging
│   └── monitor.py               # production drift monitoring stub
├── app/
│   ├── serve.py                 # FastAPI serving endpoint (/predict) + serves the UI
│   └── static/
│       └── index.html           # frontend — attrition risk scoring form
├── models/
│   └── skillmatch_model.joblib  # packaged preprocessor + champion model
├── reports/
│   ├── metrics.json             # evaluation results
│   ├── drift_report.json        # monitoring output
│   └── prediction_log.jsonl     # logged production predictions
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. Train (runs ingestion + feature engineering internally, saves model + metrics)
python src/train.py

# 2. Serve
cd app && uvicorn serve:app --reload --port 8000

# 3. Open the app in a browser
# http://localhost:8000        <- the form UI
# http://localhost:8000/docs   <- raw API docs, if you want to call /predict directly

# 4. Monitor (after some predictions have been logged)
python src/monitor.py
```

## What each module does
- **data_ingestion.py** — loads the raw CSV, drops constant/zero-variance
  columns, checks for missing values, normalises the target.
- **feature_engineering.py** — one-hot encodes categoricals, scales
  numerics, adds four derived ratio features (tenure ratio, promotion
  gap ratio, income per job level, manager stability).
- **train.py** — trains a ridge logistic regression baseline and an
  XGBoost gradient-boosted tree model, both inside a 5-fold stratified
  cross-validation loop; adds a K-means "risk segment" feature; picks a
  champion model on held-out test ROC-AUC; calibrates probabilities;
  packages the preprocessor + model together as one artifact.
- **serve.py** — FastAPI app that loads the packaged artifact and
  exposes `/predict`, applying the identical feature pipeline used at
  training time to avoid training-serving skew; logs every prediction.
  Also serves `static/index.html` at the root URL.
- **static/index.html** — the frontend: a grouped input form (Personal,
  Job, Compensation, Tenure, Satisfaction) that submits to `/predict`
  and displays the returned probability as a risk score, with a plain
  Stay/Leave verdict.
- **monitor.py** — reads the prediction log and flags fields whose
  production-time mean has drifted from the training-time mean beyond a
  threshold, recommending retraining when triggered.


## Dashboard pages and data APIs

The redesigned UI is served at `/` and includes:
- **Overview** — dataset KPIs, attrition distribution, department counts and lifecycle pipeline.
- **Predictor** — submits a complete employee profile to the actual `/predict` endpoint.
- **Data Explorer** — dynamic distribution charts and attrition-by-department aggregates from the cleaned CSV.
- **Model Lab** — displays the saved evaluation metrics and compares logistic regression with XGBoost.
- **Monitoring** — recent prediction logs, response latency and predicted-label distribution.
- **ML Lifecycle / About** — code map, module ownership and local run instructions.

Dashboard JSON endpoints:
- `GET /api/overview`
- `GET /api/eda`
- `GET /api/predictions`
- `GET /health`

The frontend uses Chart.js via CDN, so an internet connection is needed for chart rendering and the optional Google font. Prediction and API endpoints remain local. The dataset has no meaningful geographic coordinates, so the EDA uses charts rather than inventing a map.
