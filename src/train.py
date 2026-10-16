"""
SkillMatch - Training & Evaluation Module
Lifecycle stages: training set -> trained model -> packaged artifact (M2-M6)

Trains and compares two models on the attrition-prediction task:
  1. Ridge-regularised Logistic Regression   (M2)
  2. Gradient-boosted trees (XGBoost)         (M3)

Also demonstrates:
  - K-means clustering used to derive an "attrition risk segment"
    feature that is fed back into the supervised models (M4).
  - Stratified train/val/test split + 5-fold stratified CV (M5).
  - Full classification metric suite + probability calibration (M5).
  - Model packaging as a single joblib artifact containing the
    preprocessing pipeline + trained model, so the exact same
    transform is used at serving time (M6 - avoids training/serving skew).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_ingestion import run as ingest_run
from feature_engineering import (
    engineer_features, build_preprocessor, build_tree_preprocessor,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RANDOM_STATE = 42


def add_cluster_feature(df: pd.DataFrame, n_clusters: int = 4, kmeans: KMeans = None):
    """M4: unsupervised segmentation used as an engineered feature.

    Clusters employees on tenure/engagement-style numeric signals into
    K segments, then feeds the cluster id back in as a categorical
    feature ('RiskSegment') for the supervised models in M2/M3.
    """
    cluster_cols = [
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "MonthlyIncome",
    ]
    X_cluster = df[cluster_cols].copy()
    # simple min-max scale for clustering distance sanity
    X_cluster = (X_cluster - X_cluster.min()) / (X_cluster.max() - X_cluster.min() + 1e-9)

    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
    else:
        labels = kmeans.predict(X_cluster)

    df = df.copy()
    df["RiskSegment"] = labels.astype(str)
    return df, kmeans


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))
    print("ROC-AUC:", metrics["roc_auc"])
    return metrics


def run():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # --- M1/M6: ingest ---
    df = ingest_run(save=True)

    # --- M2/M6: feature engineering (ratios etc.) ---
    df = engineer_features(df)

    # --- M4: unsupervised clustering feature ---
    df, kmeans_model = add_cluster_feature(df)

    categorical_features = CATEGORICAL_FEATURES + ["RiskSegment"]
    X = df[categorical_features + NUMERIC_FEATURES]
    y = df[TARGET]

    # --- M5: stratified split (train/val/test discipline) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # ---------------- Model 1: Ridge Logistic Regression (M2) ----------------
    logreg_pre = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    logreg_pipe = Pipeline(steps=[
        ("preprocess", logreg_pre),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_STATE,
        )),
    ])

    # 5-fold stratified CV on the training split (M5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logreg_cv_auc = cross_val_score(logreg_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"LogReg 5-fold CV ROC-AUC: {logreg_cv_auc.mean():.4f} +/- {logreg_cv_auc.std():.4f}")

    logreg_pipe.fit(X_train, y_train)

    # Calibrate probabilities on the validation split (M5)
    try:
        from sklearn.frozen import FrozenEstimator
        logreg_calibrated = CalibratedClassifierCV(FrozenEstimator(logreg_pipe), method="sigmoid")
    except ImportError:
        # older sklearn API
        logreg_calibrated = CalibratedClassifierCV(logreg_pipe, method="sigmoid", cv="prefit")
    logreg_calibrated.fit(X_val, y_val)

    logreg_metrics = evaluate(logreg_calibrated, X_test, y_test, "Logistic Regression (ridge, calibrated)")
    logreg_metrics["cv_roc_auc_mean"] = round(float(logreg_cv_auc.mean()), 4)
    logreg_metrics["cv_roc_auc_std"] = round(float(logreg_cv_auc.std()), 4)

    # ---------------- Model 2: Gradient-boosted trees (M3) ----------------
    tree_pre = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_pipe = Pipeline(steps=[
        ("preprocess", tree_pre),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )),
    ])

    xgb_cv_auc = cross_val_score(xgb_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"XGBoost 5-fold CV ROC-AUC: {xgb_cv_auc.mean():.4f} +/- {xgb_cv_auc.std():.4f}")

    xgb_pipe.fit(X_train, y_train)
    xgb_metrics = evaluate(xgb_pipe, X_test, y_test, "XGBoost (gradient-boosted trees)")
    xgb_metrics["cv_roc_auc_mean"] = round(float(xgb_cv_auc.mean()), 4)
    xgb_metrics["cv_roc_auc_std"] = round(float(xgb_cv_auc.std()), 4)

    # Feature importance from the tree model (M3)
    ohe = xgb_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = cat_names + NUMERIC_FEATURES
    importances = xgb_pipe.named_steps["clf"].feature_importances_
    fi = sorted(zip(all_feature_names, importances), key=lambda t: t[1], reverse=True)[:15]
    feature_importance = [{"feature": f, "importance": round(float(i), 4)} for f, i in fi]

    # ---------------- Pick champion model on test ROC-AUC ----------------
    champion_name = "xgboost" if xgb_metrics["roc_auc"] >= logreg_metrics["roc_auc"] else "logistic_regression"
    champion_model = xgb_pipe if champion_name == "xgboost" else logreg_calibrated

    # ---------------- M6: package artifact ----------------
    artifact = {
        "model": champion_model,
        "kmeans": kmeans_model,
        "categorical_features": categorical_features,
        "numeric_features": NUMERIC_FEATURES,
        "champion_name": champion_name,
    }
    joblib.dump(artifact, MODELS_DIR / "skillmatch_model.joblib")
    print(f"\nSaved packaged artifact -> {MODELS_DIR / 'skillmatch_model.joblib'}  (champion: {champion_name})")

    results = {
        "logistic_regression": logreg_metrics,
        "xgboost": xgb_metrics,
        "champion": champion_name,
        "feature_importance_top15": feature_importance,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "base_rate_attrition": round(float(y.mean()), 4),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {REPORTS_DIR / 'metrics.json'}")

    return results


if __name__ == "__main__":
    run()
