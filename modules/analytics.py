"""
=============================================================
 IPL Analysis System — Module 2: Analytics
=============================================================
Pure analysis functions — no plotting here.
Each function returns a clean DataFrame or Series ready for
either the reporting script or the Streamlit dashboard.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. WINNER BY YEAR
# ──────────────────────────────────────────────
def winner_by_year(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Count wins per team per season.
    Returns: MultiIndex DataFrame (season × team → wins).
    """
    df = (matches.groupby(["season", "winner"])
                 .size()
                 .reset_index(name="wins")
                 .sort_values(["season", "wins"], ascending=[True, False]))
    return df


def season_champions(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the team with the most wins each season (proxy for champion).
    Returns: DataFrame with season and dominant_team.
    """
    df = winner_by_year(matches)
    champs = (df.groupby("season")
                .apply(lambda g: g.loc[g["wins"].idxmax(), "winner"])
                .reset_index(name="dominant_team"))
    return champs


# ──────────────────────────────────────────────
# 2. SUPER OVER ANALYSIS
# ──────────────────────────────────────────────
def super_over_analysis(matches: pd.DataFrame) -> dict:
    """
    Analyse super-over matches.
    Returns dict with: count, per_season, winner_breakdown.
    """
    if "super_over" not in matches.columns:
        return {"count": 0, "per_season": pd.DataFrame(), "winner_breakdown": pd.Series()}

    so = matches[matches["super_over"] == 1].copy()
    per_season = so.groupby("season").size().reset_index(name="super_overs")
    winner_breakdown = so["winner"].value_counts()
    return {
        "count": len(so),
        "per_season": per_season,
        "winner_breakdown": winner_breakdown,
        "super_over_df": so,
    }


# ──────────────────────────────────────────────
# 3. UMPIRE TRENDS
# ──────────────────────────────────────────────
def umpire_win_trends(matches: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    For each umpire, calculate which teams won most under them.
    Returns DataFrame: umpire | team | wins | matches_officiated.
    """
    rows = []
    for ump_col in ["umpire1", "umpire2"]:
        if ump_col not in matches.columns:
            continue
        grp = matches.groupby([ump_col, "winner"]).size().reset_index(name="wins")
        grp = grp.rename(columns={ump_col: "umpire"})
        rows.append(grp)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows).groupby(["umpire", "winner"], as_index=False)["wins"].sum()

    # Umpires with most matches
    total_by_ump = combined.groupby("umpire")["wins"].sum().nlargest(top_n).index
    return combined[combined["umpire"].isin(total_by_ump)].sort_values("wins", ascending=False)


# ──────────────────────────────────────────────
# 4. HIGH-SCORING MATCHES
# ──────────────────────────────────────────────
def high_scoring_matches(matches: pd.DataFrame, threshold: int = 380) -> pd.DataFrame:
    """
    Return matches where combined score > threshold.
    Adds a 'match_total' column.
    """
    if "match_total" not in matches.columns:
        if {"team1_runs", "team2_runs"}.issubset(matches.columns):
            matches = matches.copy()
            matches["match_total"] = matches["team1_runs"] + matches["team2_runs"]
        else:
            return pd.DataFrame()

    hs = matches[matches["match_total"] > threshold].copy()
    return hs.sort_values("match_total", ascending=False)


def high_score_trend(matches: pd.DataFrame, threshold: int = 380) -> pd.DataFrame:
    """Count of high-scoring matches per season."""
    hs = high_scoring_matches(matches, threshold)
    if hs.empty:
        return pd.DataFrame()
    return hs.groupby("season").size().reset_index(name="high_score_count")


# ──────────────────────────────────────────────
# 5. TEAM PERFORMANCE
# ──────────────────────────────────────────────
def team_performance(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Overall wins, losses, win% per team.
    """
    # Count appearances (team1 + team2)
    t1 = matches["team1"].value_counts().rename("played_as_t1")
    t2 = matches["team2"].value_counts().rename("played_as_t2")
    played = (t1.add(t2, fill_value=0)).astype(int).rename("matches_played")

    wins = matches["winner"].value_counts().rename("wins")

    perf = pd.concat([played, wins], axis=1).fillna(0).astype(int)
    perf["losses"] = perf["matches_played"] - perf["wins"]
    perf["win_pct"] = (perf["wins"] / perf["matches_played"] * 100).round(1)
    perf = perf.reset_index().rename(columns={"index": "team"})
    return perf.sort_values("win_pct", ascending=False)


# ──────────────────────────────────────────────
# 6. PLAYER ANALYSIS — BATSMEN
# ──────────────────────────────────────────────
def top_batsmen(deliveries: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Top batsmen by total runs and strike rate.
    """
    bat = (deliveries.groupby("batter")
                     .agg(runs=("batsman_runs", "sum"),
                          balls=("batsman_runs", "count"))
                     .reset_index())
    bat = bat[bat["balls"] >= 100]  # Minimum balls faced
    bat["strike_rate"] = (bat["runs"] / bat["balls"] * 100).round(1)
    return bat.nlargest(top_n, "runs").reset_index(drop=True)


# ──────────────────────────────────────────────
# 7. PLAYER ANALYSIS — BOWLERS
# ──────────────────────────────────────────────
def top_bowlers(deliveries: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Top bowlers by wickets taken and economy rate.
    """
    # Wickets (exclude run-outs from bowler credit)
    wickets_df = deliveries[
        (deliveries["is_wicket"] == 1) &
        (~deliveries.get("dismissal_kind", pd.Series()).isin(["run out", "retired hurt", "obstructing the field"]))
    ]
    wickets = wickets_df.groupby("bowler")["is_wicket"].sum().rename("wickets")

    # Runs conceded & overs
    runs_given = deliveries.groupby("bowler")["total_runs"].sum().rename("runs_given")
    balls = deliveries.groupby("bowler")["total_runs"].count().rename("balls_bowled")

    bowl = pd.concat([wickets, runs_given, balls], axis=1).fillna(0)
    bowl["wickets"] = bowl["wickets"].astype(int)
    bowl = bowl[bowl["balls_bowled"] >= 120]  # Minimum deliveries
    bowl["overs"] = bowl["balls_bowled"] / 6
    bowl["economy"] = (bowl["runs_given"] / bowl["overs"]).round(2)
    bowl["bowling_avg"] = np.where(bowl["wickets"] > 0,
                                   (bowl["runs_given"] / bowl["wickets"]).round(1),
                                   np.inf)
    return bowl.nlargest(top_n, "wickets").reset_index().rename(columns={"index": "bowler"}).reset_index(drop=True)


# ──────────────────────────────────────────────
# 8. ORANGE CAP (top scorer per season)
# ──────────────────────────────────────────────
def orange_cap_winners(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Top run-scorer per IPL season = Orange Cap holder.
    """
    # Join season info via match_id
    season_map = matches.set_index("id")["season"].to_dict()
    del_with_season = deliveries.copy()
    del_with_season["season"] = del_with_season["match_id"].map(season_map)
    del_with_season = del_with_season.dropna(subset=["season"])
    del_with_season["season"] = del_with_season["season"].astype(int)

    orange = (del_with_season.groupby(["season", "batter"])["batsman_runs"]
                             .sum()
                             .reset_index(name="runs"))
    idx = orange.groupby("season")["runs"].idxmax()
    return orange.loc[idx].sort_values("season").reset_index(drop=True)


# ──────────────────────────────────────────────
# 9. PURPLE CAP (top wicket-taker per season)
# ──────────────────────────────────────────────
def purple_cap_winners(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Top wicket-taker per IPL season = Purple Cap holder.
    """
    season_map = matches.set_index("id")["season"].to_dict()
    del_with_season = deliveries.copy()
    del_with_season["season"] = del_with_season["match_id"].map(season_map)
    del_with_season = del_with_season.dropna(subset=["season"])
    del_with_season["season"] = del_with_season["season"].astype(int)

    wkts_df = del_with_season[
        (del_with_season["is_wicket"] == 1) &
        (~del_with_season.get("dismissal_kind", pd.Series()).isin(["run out", "retired hurt"]))
    ]
    purple = (wkts_df.groupby(["season", "bowler"])["is_wicket"]
                     .sum()
                     .reset_index(name="wickets"))
    idx = purple.groupby("season")["wickets"].idxmax()
    return purple.loc[idx].sort_values("season").reset_index(drop=True)


# ──────────────────────────────────────────────
# 10. TOSS IMPACT
# ──────────────────────────────────────────────
def toss_impact(matches: pd.DataFrame) -> dict:
    """
    Analyse whether winning the toss helps win the match.
    """
    if "toss_match_winner" not in matches.columns:
        matches = matches.copy()
        matches["toss_match_winner"] = (matches["toss_winner"] == matches["winner"]).astype(int)

    pct = matches["toss_match_winner"].mean() * 100
    by_decision = (matches.groupby("toss_decision")["toss_match_winner"]
                           .mean()
                           .mul(100)
                           .round(1)
                           .reset_index(name="win_pct_when_toss_won"))
    return {
        "overall_toss_win_pct": round(pct, 1),
        "by_decision": by_decision,
    }


# ──────────────────────────────────────────────
# 11. VENUE ANALYSIS
# ──────────────────────────────────────────────
def venue_analysis(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Matches played and avg score at each venue.
    """
    rows = []
    for venue, grp in matches.groupby("venue"):
        avg_t1 = grp["team1_runs"].mean() if "team1_runs" in grp else 0
        avg_t2 = grp["team2_runs"].mean() if "team2_runs" in grp else 0
        rows.append({
            "venue": venue,
            "matches": len(grp),
            "avg_first_innings": round(avg_t1, 1),
            "avg_second_innings": round(avg_t2, 1),
            "avg_total": round(avg_t1 + avg_t2, 1),
        })
    return pd.DataFrame(rows).sort_values("matches", ascending=False).reset_index(drop=True)
