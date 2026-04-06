"""
=============================================================
 IPL Analysis System — Module 3: ML Model
=============================================================
Builds a match-winner prediction model.
Algorithms: Random Forest + XGBoost (picks best).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ──────────────────────────────────────────────
# FEATURE PREPARATION
# ──────────────────────────────────────────────
def prepare_ml_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Select and encode features for the prediction model.

    Features used:
        team1, team2, toss_winner, toss_decision, venue
    Target:
        winner  (who won — encoded as int)
    """
    required = {"team1", "team2", "toss_winner", "toss_decision", "venue", "winner"}
    if not required.issubset(matches.columns):
        missing = required - set(matches.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = matches[list(required)].dropna().copy()

    # Add toss_win_flag: 1 if toss winner = team1
    df["toss_win_flag"] = (df["toss_winner"] == df["team1"]).astype(int)

    # Add bat_first_flag: 1 if toss winner chose to bat
    df["bat_first_flag"] = (df["toss_decision"] == "bat").astype(int)

    # Label encode categorical columns
    encoders: dict = {}
    cat_cols = ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = ["team1_enc", "team2_enc", "toss_winner_enc",
                    "toss_decision_enc", "venue_enc",
                    "toss_win_flag", "bat_first_flag"]

    X = df[feature_cols]
    y = df["winner_enc"]

    return X, y, encoders


# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
def train_model(matches: pd.DataFrame, test_size: float = 0.2,
                random_state: int = 42) -> dict:
    """
    Train Random Forest and Gradient Boosting models.
    Returns results dict with best model, accuracy, and encoders.
    """
    print("\n🤖 Preparing ML features …")
    X, y, encoders = prepare_ml_features(matches)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train size: {len(X_train):,}   Test size: {len(X_test):,}")

    # ── Random Forest ────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # ── Gradient Boosting (XGBoost-like sklearn) ─────────────
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                    learning_rate=0.1, random_state=random_state)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))

    # Try XGBoost if available
    xgb_acc = 0
    xgb_model = None
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.1, use_label_encoder=False,
                                       eval_metric="mlogloss", random_state=random_state,
                                       verbosity=0)
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    except ImportError:
        pass

    print(f"\n📊 Model Accuracies:")
    print(f"   Random Forest       : {rf_acc*100:.1f}%")
    print(f"   Gradient Boosting   : {gb_acc*100:.1f}%")
    if xgb_acc:
        print(f"   XGBoost             : {xgb_acc*100:.1f}%")

    # Pick best
    candidates = {"Random Forest": (rf, rf_acc),
                  "Gradient Boosting": (gb, gb_acc)}
    if xgb_acc:
        candidates["XGBoost"] = (xgb_model, xgb_acc)

    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model, best_acc = candidates[best_name]

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importances
    importances = pd.Series(best_model.feature_importances_,
                            index=X.columns).sort_values(ascending=False)

    print(f"\n🏆 Best model: {best_name}  →  Accuracy: {best_acc*100:.1f}%")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "best_accuracy": best_acc,
        "rf_model": rf, "rf_acc": rf_acc,
        "gb_model": gb, "gb_acc": gb_acc,
        "xgb_model": xgb_model, "xgb_acc": xgb_acc,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "feature_importances": importances,
        "confusion_matrix": cm,
        "classification_report": report,
        "encoders": encoders,
        "feature_cols": list(X.columns),
    }


# ──────────────────────────────────────────────
# PREDICT A SINGLE MATCH
# ──────────────────────────────────────────────
def predict_match(results: dict,
                  team1: str, team2: str,
                  toss_winner: str, toss_decision: str,
                  venue: str) -> str:
    """
    Predict winner for a hypothetical match.
    Returns predicted winning team name.
    """
    model = results["best_model"]
    encoders = results["encoders"]

    def safe_encode(le: LabelEncoder, value: str) -> int:
        classes = list(le.classes_)
        if value in classes:
            return le.transform([value])[0]
        # Unknown label → use most common class (index 0 after sort)
        return 0

    features = {
        "team1_enc": safe_encode(encoders["team1"], team1),
        "team2_enc": safe_encode(encoders["team2"], team2),
        "toss_winner_enc": safe_encode(encoders["toss_winner"], toss_winner),
        "toss_decision_enc": safe_encode(encoders["toss_decision"], toss_decision),
        "venue_enc": safe_encode(encoders["venue"], venue),
        "toss_win_flag": int(toss_winner == team1),
        "bat_first_flag": int(toss_decision == "bat"),
    }

    X_new = pd.DataFrame([features])[results["feature_cols"]]
    pred_enc = model.predict(X_new)[0]
    winner = encoders["winner"].inverse_transform([pred_enc])[0]
    return winner


# ──────────────────────────────────────────────
# SAMPLE PREDICTIONS TABLE
# ──────────────────────────────────────────────
def sample_predictions(results: dict, matches: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Generate a table of sample predictions vs actual results."""
    sample = matches.sample(min(n, len(matches)), random_state=99)[
        ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
    ].dropna()

    predictions = []
    for _, row in sample.iterrows():
        pred = predict_match(results, row["team1"], row["team2"],
                             row["toss_winner"], row["toss_decision"], row["venue"])
        predictions.append({
            "team1": row["team1"],
            "team2": row["team2"],
            "toss_winner": row["toss_winner"],
            "toss_decision": row["toss_decision"],
            "venue": row["venue"],
            "actual_winner": row["winner"],
            "predicted_winner": pred,
            "correct": "✅" if pred == row["winner"] else "❌",
        })
    return pd.DataFrame(predictions)
