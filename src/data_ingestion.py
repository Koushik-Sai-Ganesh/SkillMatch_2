"""
SkillMatch - Data Ingestion Module
Lifecycle stage: raw data -> cleaned dataset (M1, M6)

Loads the IBM HR Analytics Employee Attrition & Performance dataset,
performs basic cleaning, and returns a tidy DataFrame ready for the
feature engineering stage.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "attrition_raw.csv"
CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "attrition_clean.csv"

# Columns that are constant across every row in this dataset and carry
# zero predictive signal (verified during EDA) -> dropped at ingestion.
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]

# Identifier column - not a feature, kept aside for traceability only.
ID_COLUMN = "EmployeeNumber"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning rules.

    - Drop constant / zero-variance columns.
    - Check for and report missing values (dataset has none, but the
      check is kept so the pipeline is safe if the source data changes).
    - Normalise the target column to a 0/1 integer.
    - Strip whitespace from object columns.
    """
    df = df.copy()

    # Drop known-constant columns if present
    df = df.drop(columns=[c for c in CONSTANT_COLUMNS if c in df.columns])

    # Whitespace cleanup for categorical/text columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Missing value report (informational - dataset is complete)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values detected:\n", missing)
    else:
        print("Missing value check: no missing values found.")

    # Target normalisation: Attrition Yes/No -> 1/0
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

    # De-duplicate on the employee identifier if present
    if ID_COLUMN in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[ID_COLUMN])
        after = len(df)
        if before != after:
            print(f"Dropped {before - after} duplicate rows on {ID_COLUMN}")

    return df.reset_index(drop=True)


def run(save: bool = True) -> pd.DataFrame:
    raw = load_raw()
    print(f"Loaded raw dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")
    cleaned = clean(raw)
    print(f"Cleaned dataset: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    if save:
        cleaned.to_csv(CLEAN_PATH, index=False)
        print(f"Saved cleaned dataset -> {CLEAN_PATH}")
    return cleaned


if __name__ == "__main__":
    run()
