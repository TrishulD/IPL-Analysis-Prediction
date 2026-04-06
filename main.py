"""
=============================================================
  IPL Cricket Analysis & Prediction System
  main.py  —  Run this file to execute the full pipeline
=============================================================

Usage:
  python main.py

Outputs:
  outputs/  ← all charts (PNG)
  A printed report in the terminal
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from modules.data_loader   import load_and_prepare
from modules.analytics     import (winner_by_year, season_champions,
                                   super_over_analysis, umpire_win_trends,
                                   high_scoring_matches, high_score_trend,
                                   team_performance, top_batsmen, top_bowlers,
                                   orange_cap_winners, purple_cap_winners,
                                   toss_impact, venue_analysis)
from modules.ml_model      import train_model, sample_predictions
from modules.visualizations import generate_all_charts


DIVIDER = "=" * 65


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN DATA
# ──────────────────────────────────────────────────────────────────
section("STEP 1 — DATA LOADING & CLEANING")
matches, deliveries = load_and_prepare()

print(f"\n  Seasons covered  : {matches['season'].min()} – {matches['season'].max()}")
print(f"  Total matches    : {len(matches):,}")
print(f"  Total deliveries : {len(deliveries):,}")
print(f"  Unique teams     : {matches['winner'].nunique()}")
print(f"  Unique venues    : {matches['venue'].nunique()}")


# ──────────────────────────────────────────────────────────────────
# STEP 2 — WINNER BY YEAR
# ──────────────────────────────────────────────────────────────────
section("STEP 2 — SEASON DOMINANCE")
champs = season_champions(matches)
print("\n  📅 Season Champions (most wins per season):")
print(champs.to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 3 — SUPER OVER
# ──────────────────────────────────────────────────────────────────
section("STEP 3 — SUPER OVER ANALYSIS")
so = super_over_analysis(matches)
print(f"\n  ⚡ Total Super Overs in dataset: {so['count']}")
if not so["per_season"].empty:
    print("\n  Super Overs per season:")
    print(so["per_season"].to_string(index=False))
if not so["winner_breakdown"].empty:
    print("\n  Teams that won most Super Overs:")
    print(so["winner_breakdown"].head(5).to_string())


# ──────────────────────────────────────────────────────────────────
# STEP 4 — UMPIRE TRENDS
# ──────────────────────────────────────────────────────────────────
section("STEP 4 — UMPIRE TRENDS")
ump = umpire_win_trends(matches)
print("\n  🧑‍⚖️  Top umpire–team win combinations:")
print(ump.head(10).to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 5 — HIGH-SCORING MATCHES
# ──────────────────────────────────────────────────────────────────
section("STEP 5 — HIGH-SCORING MATCHES (combined > 380)")
hs = high_scoring_matches(matches, threshold=380)
print(f"\n  🔥 Total high-scoring matches: {len(hs)}")
if not hs.empty:
    cols = [c for c in ["season","venue","team1","team1_runs","team2","team2_runs","match_total","winner"]
            if c in hs.columns]
    print(hs[cols].head(10).to_string(index=False))
hs_trend = high_score_trend(matches, threshold=380)


# ──────────────────────────────────────────────────────────────────
# STEP 6 — TEAM PERFORMANCE
# ──────────────────────────────────────────────────────────────────
section("STEP 6 — TEAM PERFORMANCE")
perf = team_performance(matches)
print("\n  📊 Team Win Statistics:")
print(perf.to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 7 — TOP BATSMEN & BOWLERS
# ──────────────────────────────────────────────────────────────────
section("STEP 7 — PLAYER ANALYSIS")
bat = top_batsmen(deliveries, top_n=15)
print("\n  🏏 Top Batsmen:")
print(bat[["batter","runs","balls","strike_rate"]].to_string(index=False))

bowl = top_bowlers(deliveries, top_n=15)
print("\n  🎳 Top Bowlers:")
print(bowl[["bowler","wickets","economy","bowling_avg"]].to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 8 — ORANGE & PURPLE CAP
# ──────────────────────────────────────────────────────────────────
section("STEP 8 — ORANGE & PURPLE CAP")
orange = orange_cap_winners(matches, deliveries)
purple = purple_cap_winners(matches, deliveries)

print("\n  🟠 Orange Cap Winners (Top scorer per season):")
print(orange.rename(columns={"batter":"player"}).to_string(index=False))

print("\n  🟣 Purple Cap Winners (Top wicket-taker per season):")
print(purple.rename(columns={"bowler":"player"}).to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 9 — TOSS & VENUE ANALYSIS
# ──────────────────────────────────────────────────────────────────
section("STEP 9 — TOSS & VENUE IMPACT")
toss = toss_impact(matches)
print(f"\n  🎲 Overall win % for toss winner: {toss['overall_toss_win_pct']}%")
print("\n  Win % by toss decision:")
print(toss["by_decision"].to_string(index=False))

venue = venue_analysis(matches)
print("\n  🏟  Venue Summary (top 8):")
print(venue.head(8).to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# STEP 10 — MACHINE LEARNING MODEL
# ──────────────────────────────────────────────────────────────────
section("STEP 10 — MATCH WINNER PREDICTION MODEL")
ml_results = train_model(matches)

print(f"\n  Model: {ml_results['best_model_name']}")
print(f"  Accuracy: {ml_results['best_accuracy']*100:.1f}%")

print("\n  Feature Importances:")
for feat, imp in ml_results["feature_importances"].items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:<22} {bar}  {imp:.4f}")

print("\n  📋 Sample Predictions:")
preds = sample_predictions(ml_results, matches, n=10)
print(preds.to_string(index=False))

correct = (preds["correct"] == "✅").sum()
print(f"\n  Correct on sample: {correct}/{len(preds)}")


# ──────────────────────────────────────────────────────────────────
# STEP 11 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────
section("STEP 11 — GENERATING CHARTS")
saved_charts = generate_all_charts(
    matches, deliveries, ml_results,
    bat, bowl, orange, purple,
    hs_trend, so
)
print(f"\n  ✅ {len(saved_charts)} charts saved to outputs/")


# ──────────────────────────────────────────────────────────────────
# FINAL INSIGHTS SUMMARY
# ──────────────────────────────────────────────────────────────────
section("FINAL INSIGHTS SUMMARY")
best_team = matches["winner"].value_counts().idxmax()
best_team_wins = matches["winner"].value_counts().max()

best_bat = bat.iloc[0]
best_bowl = bowl.iloc[0]

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │              📊 IPL DATA SCIENCE INSIGHTS               │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  🏆 Most Successful Team : {best_team:<28}│
  │     Total Wins           : {best_team_wins:<28}│
  │                                                         │
  │  🏏 Top Batsman  : {best_bat['batter']:<38}│
  │     Runs         : {best_bat['runs']:<38}│
  │     Strike Rate  : {best_bat['strike_rate']:<38}│
  │                                                         │
  │  🎳 Top Bowler   : {best_bowl['bowler']:<38}│
  │     Wickets      : {best_bowl['wickets']:<38}│
  │     Economy      : {best_bowl['economy']:<38}│
  │                                                         │
  │  🎲 Toss Win → Match Win : {toss['overall_toss_win_pct']}%{'':<30}│
  │  ⚡ Total Super Overs    : {so['count']:<28}│
  │  🔥 High-Scoring Matches : {len(hs):<28}│
  │                                                         │
  │  🤖 ML Model Accuracy    : {ml_results['best_accuracy']*100:.1f}%{'':<29}│
  │     Algorithm            : {ml_results['best_model_name']:<28}│
  │                                                         │
  └─────────────────────────────────────────────────────────┘
""")

print(f"\n✅ Analysis complete! All outputs in: outputs/\n")
