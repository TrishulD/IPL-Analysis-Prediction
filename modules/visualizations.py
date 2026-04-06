"""
=============================================================
 IPL Analysis System — Module 4: Visualizations
=============================================================
All charts saved to outputs/ folder.
Uses Matplotlib + Seaborn for static charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────
IPL_COLORS = ["#E53935", "#F4A62A", "#1565C0", "#2E7D32",
              "#6A1B9A", "#00838F", "#EF6C00", "#AD1457",
              "#4527A0", "#00695C"]

def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 Saved → {path}")
    return path


# ──────────────────────────────────────────────
# 1. TOTAL WINS PER TEAM (all seasons)
# ──────────────────────────────────────────────
def plot_total_wins(matches: pd.DataFrame) -> str:
    wins = matches["winner"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    bars = ax.barh(wins.index[::-1], wins.values[::-1],
                   color=IPL_COLORS[:len(wins)], edgecolor="none", height=0.65)
    for bar, val in zip(bars, wins.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", color="white", fontsize=11, fontweight="bold")

    ax.set_xlabel("Total Wins", color="white", fontsize=12)
    ax.set_title("🏆  All-Time IPL Wins per Team", color="white", fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="white", alpha=0.1)
    fig.tight_layout()
    return _save(fig, "01_total_wins.png")


# ──────────────────────────────────────────────
# 2. WINS PER SEASON LINE GRAPH
# ──────────────────────────────────────────────
def plot_wins_per_season(matches: pd.DataFrame, top_teams: int = 5) -> str:
    top = matches["winner"].value_counts().head(top_teams).index.tolist()
    season_wins = (matches[matches["winner"].isin(top)]
                   .groupby(["season", "winner"])
                   .size()
                   .reset_index(name="wins"))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for i, team in enumerate(top):
        data = season_wins[season_wins["winner"] == team]
        ax.plot(data["season"], data["wins"], marker="o", linewidth=2.2,
                color=IPL_COLORS[i % len(IPL_COLORS)], label=team, markersize=6)

    ax.set_title("📈  Season-wise Win Trends — Top Teams", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", color="white"); ax.set_ylabel("Wins", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="white", alpha=0.1)
    legend = ax.legend(facecolor="#1C1C2E", labelcolor="white", framealpha=0.8)
    fig.tight_layout()
    return _save(fig, "02_wins_per_season.png")


# ──────────────────────────────────────────────
# 3. TOSS IMPACT
# ──────────────────────────────────────────────
def plot_toss_impact(matches: pd.DataFrame) -> str:
    toss_match = (matches["toss_winner"] == matches["winner"]).mean() * 100
    no_toss = 100 - toss_match

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0D1117")

    # Pie
    ax = axes[0]; ax.set_facecolor("#0D1117")
    wedges, texts, autotexts = ax.pie(
        [toss_match, no_toss], labels=["Won toss & match", "Lost toss but won"],
        colors=["#F4A62A", "#1565C0"], autopct="%1.1f%%",
        startangle=90, textprops={"color": "white", "fontsize": 11})
    for at in autotexts: at.set_fontsize(11)
    ax.set_title("Toss Winner → Match Winner?", color="white", fontsize=12, fontweight="bold")

    # By decision
    ax2 = axes[1]; ax2.set_facecolor("#0D1117")
    by_dec = (matches.groupby("toss_decision")
              .apply(lambda g: (g["toss_winner"] == g["winner"]).mean() * 100)
              .reset_index(name="win_pct"))
    bars = ax2.bar(by_dec["toss_decision"], by_dec["win_pct"],
                   color=["#E53935", "#2E7D32"], width=0.4, edgecolor="none")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=12)
    ax2.set_ylim(0, 100); ax2.set_ylabel("Win %", color="white")
    ax2.set_title("Win % by Toss Decision", color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white"); ax2.spines[:].set_visible(False)

    fig.suptitle("🎲  Toss Impact Analysis", color="white", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "03_toss_impact.png")


# ──────────────────────────────────────────────
# 4. TOP BATSMEN
# ──────────────────────────────────────────────
def plot_top_batsmen(bat_df: pd.DataFrame, top_n: int = 10) -> str:
    df = bat_df.head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    for ax, col, title, color in [
        (axes[0], "runs", "Total Runs", "#F4A62A"),
        (axes[1], "strike_rate", "Strike Rate", "#E53935"),
    ]:
        ax.set_facecolor("#0D1117")
        bars = ax.barh(df["batter"][::-1], df[col][::-1],
                       color=color, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, df[col][::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:,.1f}", va="center", color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color="white", alpha=0.1)

    fig.suptitle("🏏  Top Batsmen", color="white", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_top_batsmen.png")


# ──────────────────────────────────────────────
# 5. TOP BOWLERS
# ──────────────────────────────────────────────
def plot_top_bowlers(bowl_df: pd.DataFrame, top_n: int = 10) -> str:
    df = bowl_df.head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    for ax, col, title, color in [
        (axes[0], "wickets", "Wickets", "#6A1B9A"),
        (axes[1], "economy", "Economy Rate", "#00838F"),
    ]:
        ax.set_facecolor("#0D1117")
        d = df.sort_values(col, ascending=(col == "economy")).head(top_n)
        bars = ax.barh(d["bowler"], d[col], color=color, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, d[col]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val}", va="center", color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color="white", alpha=0.1)

    fig.suptitle("🎳  Top Bowlers", color="white", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_top_bowlers.png")


# ──────────────────────────────────────────────
# 6. ORANGE & PURPLE CAP TABLE
# ──────────────────────────────────────────────
def plot_caps(orange: pd.DataFrame, purple: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0D1117")

    for ax, df, title, col, color, icon in [
        (axes[0], orange, "Orange Cap — Top Run Scorer per Season",
         "runs", "#FF8C00", "🟠"),
        (axes[1], purple, "Purple Cap — Top Wicket Taker per Season",
         "wickets", "#9C27B0", "🟣"),
    ]:
        ax.set_facecolor("#0D1117")
        player_col = "batter" if "batter" in df.columns else "bowler"
        y = range(len(df))
        ax.barh(y, df[col], color=color, alpha=0.75, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{row['season']}  {row[player_col]}"
                            for _, row in df.iterrows()], color="white", fontsize=9)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row[col] + 0.3, i, str(row[col]), va="center", color="white", fontsize=8)
        ax.set_title(f"{icon}  {title}", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color="white", alpha=0.1)

    fig.tight_layout()
    return _save(fig, "06_caps.png")


# ──────────────────────────────────────────────
# 7. HIGH-SCORING MATCH TREND
# ──────────────────────────────────────────────
def plot_high_score_trend(trend_df: pd.DataFrame) -> str:
    if trend_df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
    ax.fill_between(trend_df["season"], trend_df["high_score_count"],
                    alpha=0.4, color="#E53935")
    ax.plot(trend_df["season"], trend_df["high_score_count"],
            marker="o", color="#E53935", linewidth=2.2)
    ax.set_title("🔥  High-Scoring Matches per Season (combined > 380 runs)",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Season", color="white"); ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="white", alpha=0.1)
    fig.tight_layout()
    return _save(fig, "07_high_score_trend.png")


# ──────────────────────────────────────────────
# 8. TEAM WIN % HEATMAP
# ──────────────────────────────────────────────
def plot_win_heatmap(matches: pd.DataFrame) -> str:
    """Head-to-head win % heatmap."""
    teams = matches["winner"].value_counts().head(8).index.tolist()
    df = matches[matches["team1"].isin(teams) & matches["team2"].isin(teams)]

    matrix = pd.DataFrame(0.0, index=teams, columns=teams)
    for _, row in df.iterrows():
        t1, t2, w = row["team1"], row["team2"], row["winner"]
        if t1 in teams and t2 in teams:
            matrix.loc[t1, t2] += 1 if w == t1 else 0
            matrix.loc[t2, t1] += 1 if w == t2 else 0

    # Normalise to win %
    totals = df.groupby(["team1", "team2"]).size().reset_index(name="n")
    for _, row in totals.iterrows():
        t1, t2 = row["team1"], row["team2"]
        if t1 in teams and t2 in teams and row["n"] > 0:
            matrix.loc[t1, t2] = matrix.loc[t1, t2] / row["n"] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="#0D1117",
                annot_kws={"size": 9}, ax=ax, cbar_kws={"label": "Win %"})
    ax.set_title("🔥  Head-to-Head Win % Heatmap", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, "08_hh_heatmap.png")


# ──────────────────────────────────────────────
# 9. FEATURE IMPORTANCES
# ──────────────────────────────────────────────
def plot_feature_importance(importances: pd.Series) -> str:
    nice_names = {
        "team1_enc": "Team 1", "team2_enc": "Team 2",
        "toss_winner_enc": "Toss Winner", "toss_decision_enc": "Toss Decision",
        "venue_enc": "Venue", "toss_win_flag": "Toss Win Flag",
        "bat_first_flag": "Bat First Flag",
    }
    imp = importances.rename(index=nice_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(imp.index, imp.values, color=colors, edgecolor="none")
    ax.set_title("🤖  Feature Importances — ML Model", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="white", alpha=0.1)
    fig.tight_layout()
    return _save(fig, "09_feature_importance.png")


# ──────────────────────────────────────────────
# 10. SUPER OVER TREND
# ──────────────────────────────────────────────
def plot_super_over(per_season: pd.DataFrame) -> str:
    if per_season.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
    ax.bar(per_season["season"], per_season["super_overs"],
           color="#00BCD4", alpha=0.85, edgecolor="none", width=0.7)
    ax.set_title("⚡  Super Overs per Season", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Season", color="white"); ax.set_ylabel("Super Overs", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="white", alpha=0.1)
    fig.tight_layout()
    return _save(fig, "10_super_over.png")


# ──────────────────────────────────────────────
# GENERATE ALL CHARTS
# ──────────────────────────────────────────────
def generate_all_charts(matches, deliveries, ml_results,
                        bat_df, bowl_df, orange, purple,
                        hs_trend, so_data) -> list[str]:
    """Call all plot functions and return list of saved paths."""
    print("\n🎨 Generating visualizations …")
    saved = []
    saved.append(plot_total_wins(matches))
    saved.append(plot_wins_per_season(matches))
    saved.append(plot_toss_impact(matches))
    saved.append(plot_top_batsmen(bat_df))
    saved.append(plot_top_bowlers(bowl_df))
    saved.append(plot_caps(orange, purple))
    saved.append(plot_high_score_trend(hs_trend))
    saved.append(plot_win_heatmap(matches))
    if ml_results:
        saved.append(plot_feature_importance(ml_results["feature_importances"]))
    saved.append(plot_super_over(so_data.get("per_season", pd.DataFrame())))
    return [p for p in saved if p]
