"""
SkillMatch - Feature Engineering Module
Lifecycle stage: cleaned dataset -> engineered features -> training set (M2, M6)

Builds the preprocessing pipeline (encoding + scaling) and a small set of
derived / engineered features on top of the raw HR attributes.

The transformer itself is packaged with the model artifact (see
train.py) so the exact same encoding is applied at serving time,
which is how this project avoids training-serving skew (M6).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "Attrition"
ID_COLUMN = "EmployeeNumber"

# Categorical columns -> one-hot encoding (M2)
CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Numeric columns used as-is (scaled for the linear model, passed through
# unscaled for tree models - handled inside train.py per model type)
NUMERIC_FEATURES_BASE = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Engineered (derived) features - simple ratio / tenure signals that are
# not present verbatim in the raw data but are well known attrition
# drivers in HR analytics.
ENGINEERED_FEATURES = [
    "TenureRatio",          # YearsAtCompany / TotalWorkingYears
    "PromotionGapRatio",    # YearsSinceLastPromotion / (YearsAtCompany + 1)
    "IncomePerJobLevel",    # MonthlyIncome / JobLevel
    "ManagerStability",     # YearsWithCurrManager / (YearsAtCompany + 1)
]

NUMERIC_FEATURES = NUMERIC_FEATURES_BASE + ENGINEERED_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns on top of the cleaned dataframe."""
    df = df.copy()
    df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)
    df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"].replace(0, 1)
    df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: one-hot for categoricals, scaling for numerics.

    Used for the linear (logistic regression) model. Tree-based models
    reuse the one-hot step but skip scaling (trees are scale-invariant).
    """
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline(steps=[
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", numeric_pipe, NUMERIC_FEATURES),
    ])
    return preprocessor


def build_tree_preprocessor() -> ColumnTransformer:
    """Preprocessor variant for tree models: one-hot categoricals, raw numerics."""
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    return preprocessor


def get_feature_target_split(df: pd.DataFrame):
    df = engineer_features(df)
    drop_cols = [TARGET]
    if ID_COLUMN in df.columns:
        drop_cols.append(ID_COLUMN)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]
    return X, y
