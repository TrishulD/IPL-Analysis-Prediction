"""
=============================================================
 IPL Analysis System — Module 1: Data Loader & Cleaner
=============================================================
Handles loading, cleaning, and feature engineering for
both matches.csv and deliveries.csv datasets.
"""

import pandas as pd
import numpy as np
import os


# ──────────────────────────────────────────────
# 1. LOAD DATASETS
# ──────────────────────────────────────────────
def load_data(matches_path: str = "data/matches.csv",
              deliveries_path: str = "data/deliveries.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IPL datasets from CSV files."""
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"matches.csv not found at: {matches_path}")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"deliveries.csv not found at: {deliveries_path}")

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    print(f"✅ Loaded matches.csv     → {matches.shape[0]:,} rows × {matches.shape[1]} columns")
    print(f"✅ Loaded deliveries.csv  → {deliveries.shape[0]:,} rows × {deliveries.shape[1]} columns")
    return matches, deliveries


# ──────────────────────────────────────────────
# 2. CLEAN MATCHES
# ──────────────────────────────────────────────
def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the matches DataFrame:
      - Standardise column names
      - Fill / drop missing values
      - Normalise team name variants
      - Parse dates
      - Remove duplicate rows
    """
    df = matches.copy()

    # Lowercase + strip column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Team name normalisation map ──────────────────────────
    name_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiants": "Pune Warriors India",
        "Rising Pune Supergiant": "Pune Warriors India",
    }
    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].replace(name_map)

    # ── Date parsing ─────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Fill missing umpire names ────────────────────────────
    for col in ["umpire1", "umpire2"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── Fill missing player_of_match ────────────────────────
    if "player_of_match" in df.columns:
        df["player_of_match"] = df["player_of_match"].fillna("Not Awarded")

    # ── Drop rows where winner is NaN (no result / abandoned) ─
    if "winner" in df.columns:
        before = len(df)
        df = df.dropna(subset=["winner"])
        dropped = before - len(df)
        if dropped:
            print(f"  ⚠  Dropped {dropped} matches with no result (abandoned/NR)")

    # ── Remove duplicate match IDs ───────────────────────────
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    print(f"  ✅ Matches after cleaning: {len(df):,} rows")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. CLEAN DELIVERIES
# ──────────────────────────────────────────────
def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the deliveries DataFrame:
      - Standardise column names
      - Fill missing numeric fields with 0
      - Cast integer columns safely
    """
    df = deliveries.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Fill numeric NaNs with 0 ─────────────────────────────
    numeric_cols = ["batsman_runs", "extra_runs", "total_runs", "is_wicket"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Fill string NaNs ─────────────────────────────────────
    for col in ["player_dismissed", "dismissal_kind", "fielder", "extras_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    print(f"  ✅ Deliveries after cleaning: {len(df):,} rows")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING — MATCHES
# ──────────────────────────────────────────────
def engineer_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns:
      - toss_match_winner  : did the toss winner also win the match?
      - bat_first_win      : did the team batting first win?
      - high_scoring       : was the match total > 400 (both innings)?
      - result_type        : categorical bucket for result margin
    """
    df = matches.copy()

    # Toss winner == match winner?
    if {"toss_winner", "winner"}.issubset(df.columns):
        df["toss_match_winner"] = (df["toss_winner"] == df["winner"]).astype(int)

    # Did the team batting first win?
    if {"toss_decision", "toss_winner", "team1", "winner"}.issubset(df.columns):
        bat_first = np.where(df["toss_decision"] == "bat", df["toss_winner"],
                             np.where(df["toss_winner"] == df["team1"], df["team2"], df["team1"]))
        df["batting_first_team"] = bat_first
        df["bat_first_win"] = (df["batting_first_team"] == df["winner"]).astype(int)

    # High-scoring match flag (total runs > 380 in match)
    if {"team1_runs", "team2_runs"}.issubset(df.columns):
        df["match_total"] = df["team1_runs"] + df["team2_runs"]
        df["high_scoring"] = (df["match_total"] > 380).astype(int)

    # Season decade bucket
    if "season" in df.columns:
        df["season_era"] = pd.cut(df["season"],
                                  bins=[2007, 2012, 2017, 2024],
                                  labels=["Early (08-12)", "Mid (13-17)", "Recent (18-23)"])

    print("  ✅ Feature engineering complete")
    return df


# ──────────────────────────────────────────────
# 5. LABEL ENCODING — for ML module
# ──────────────────────────────────────────────
def encode_labels(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns used by the ML model.
    Returns encoded DataFrame + encoding dictionaries.
    """
    from sklearn.preprocessing import LabelEncoder
    df = matches.copy()
    encoders: dict = {}

    cat_cols = ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"  ✅ Label encoding applied to: {cat_cols}")
    return df, encoders


# ──────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ──────────────────────────────────────────────
def load_and_prepare(matches_path="data/matches.csv",
                     deliveries_path="data/deliveries.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single call to load, clean, and engineer all features."""
    print("\n📂 Loading data …")
    matches, deliveries = load_data(matches_path, deliveries_path)

    print("\n🧹 Cleaning …")
    matches = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)

    print("\n🔧 Engineering features …")
    matches = engineer_match_features(matches)

    return matches, deliveries
