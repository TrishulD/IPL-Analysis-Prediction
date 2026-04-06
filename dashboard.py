"""
=============================================================
  IPL Cricket Analysis System — Streamlit Dashboard
  dashboard.py  —  Run with: streamlit run dashboard.py
=============================================================
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader   import load_and_prepare
from modules.analytics     import (winner_by_year, season_champions,
                                   super_over_analysis, high_scoring_matches,
                                   high_score_trend, team_performance,
                                   top_batsmen, top_bowlers,
                                   orange_cap_winners, purple_cap_winners,
                                   toss_impact, venue_analysis, umpire_win_trends)
from modules.ml_model      import train_model, predict_match

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analysis System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; }

  .main { background-color: #0A0E1A; }
  section[data-testid="stSidebar"] { background: #111827 !important; }

  .metric-card {
    background: linear-gradient(135deg, #1a1f35, #252b45);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-card h2 { font-size: 2rem; color: #F4A62A; margin: 0; }
  .metric-card p  { color: #8892b0; margin: 4px 0 0 0; font-size: 0.85rem; }

  .stSelectbox > div > div { background: #1a1f35 !important; color: white !important; }
  .stButton > button {
    background: linear-gradient(135deg, #E53935, #F4A62A);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.5rem 2rem;
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;
  }
  .section-header {
    background: linear-gradient(90deg, #E53935 0%, #F4A62A 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.6rem; font-family: 'Rajdhani', sans-serif; font-weight: 700;
  }
  div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD DATA (cached)
# ──────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_prepare()

@st.cache_resource
def get_ml(matches):
    return train_model(matches)

matches, deliveries = get_data()
IPL_COLORS = ["#E53935","#F4A62A","#1565C0","#2E7D32",
              "#6A1B9A","#00838F","#EF6C00","#AD1457","#4527A0","#00695C"]

BG = "#0A0E1A"
CARD = "#141824"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/200px-Indian_Premier_League_Official_Logo.svg.png",
                 width=160)
st.sidebar.title("🏏 IPL Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Team Analysis", "🏏 Player Analysis",
     "🎲 Match Insights", "🤖 Predict Winner", "🏆 Caps & Awards"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Seasons:** {matches['season'].min()} – {matches['season'].max()}")
st.sidebar.markdown(f"**Matches:** {len(matches):,}")
st.sidebar.markdown(f"**Deliveries:** {len(deliveries):,}")


def dark_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#2d3561")
    return fig, ax

def dark_fig2(w=13, h=5):
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(CARD); ax.tick_params(colors="white")
        ax.spines[:].set_color("#2d3561")
    return fig, axes


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<h1 style="text-align:center;color:#F4A62A;font-family:Rajdhani;">🏏 IPL Cricket Analysis & Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8892b0;">Complete Data Science project — 2008 to 2023</p>', unsafe_allow_html=True)
    st.markdown("---")

    # KPI cards
    col1,col2,col3,col4,col5 = st.columns(5)
    best_team = matches["winner"].value_counts().idxmax()
    best_wins = int(matches["winner"].value_counts().max())
    so = super_over_analysis(matches)
    hs = high_scoring_matches(matches)
    toss = toss_impact(matches)

    for col, h, p in [
        (col1, f"{matches['season'].nunique()}", "Seasons"),
        (col2, f"{len(matches):,}", "Matches"),
        (col3, f"{best_wins}", f"🏆 {best_team.split()[-1]}"),
        (col4, f"{so['count']}", "Super Overs"),
        (col5, f"{toss['overall_toss_win_pct']}%", "Toss→Win %"),
    ]:
        col.markdown(f'<div class="metric-card"><h2>{h}</h2><p>{p}</p></div>', unsafe_allow_html=True)

    st.markdown("### ")

    # Season champions table + wins chart side by side
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<p class="section-header">Season Champions</p>', unsafe_allow_html=True)
        champs = season_champions(matches)
        st.dataframe(champs, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<p class="section-header">Total Wins per Team</p>', unsafe_allow_html=True)
        wins = matches["winner"].value_counts().head(10)
        fig, ax = dark_fig(9, 5)
        bars = ax.barh(wins.index[::-1], wins.values[::-1],
                       color=IPL_COLORS[:len(wins)], edgecolor="none")
        for bar, v in zip(bars, wins.values[::-1]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    str(v), va="center", color="white", fontsize=9)
        ax.set_xlabel("Wins", color="white"); ax.xaxis.grid(True, color="white", alpha=0.1)
        ax.spines[:].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════
# PAGE 2 — TEAM ANALYSIS
# ═══════════════════════════════════════════════
elif page == "📊 Team Analysis":
    st.markdown('<h2 class="section-header">Team Performance Analysis</h2>', unsafe_allow_html=True)

    perf = team_performance(matches)
    st.dataframe(perf.style.background_gradient(subset=["win_pct"], cmap="YlOrRd"),
                 use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Win % by Team")
        fig, ax = dark_fig(7, 5)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(perf)))
        ax.barh(perf["team"][::-1], perf["win_pct"][::-1], color=colors, edgecolor="none")
        ax.set_xlabel("Win %", color="white"); ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color="white", alpha=0.1)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("#### Season Win Trends")
        top5 = matches["winner"].value_counts().head(5).index.tolist()
        sw = (matches[matches["winner"].isin(top5)]
              .groupby(["season","winner"]).size().reset_index(name="wins"))
        fig, ax = dark_fig(7, 5)
        for i, team in enumerate(top5):
            d = sw[sw["winner"]==team]
            ax.plot(d["season"], d["wins"], marker="o", linewidth=2,
                    color=IPL_COLORS[i], label=team, markersize=5)
        ax.set_xlabel("Season", color="white"); ax.set_ylabel("Wins", color="white")
        ax.legend(facecolor="#1a1f35", labelcolor="white", fontsize=8)
        ax.spines[:].set_visible(False); ax.yaxis.grid(True, color="white", alpha=0.1)
        st.pyplot(fig, use_container_width=True); plt.close()

    # H2H Heatmap
    st.markdown("#### Head-to-Head Win % Heatmap")
    teams = matches["winner"].value_counts().head(8).index.tolist()
    df_h2h = matches[matches["team1"].isin(teams) & matches["team2"].isin(teams)]
    matrix = pd.DataFrame(0.0, index=teams, columns=teams)
    for _, row in df_h2h.iterrows():
        t1, t2, w = row["team1"], row["team2"], row["winner"]
        if t1 in teams and t2 in teams:
            matrix.loc[t1, t2] += (1 if w==t1 else 0)
            matrix.loc[t2, t1] += (1 if w==t2 else 0)
    totals = df_h2h.groupby(["team1","team2"]).size().reset_index(name="n")
    for _, row in totals.iterrows():
        t1,t2 = row["team1"], row["team2"]
        if t1 in teams and t2 in teams and row["n"]>0:
            matrix.loc[t1,t2] = matrix.loc[t1,t2]/row["n"]*100
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.4, linecolor=BG, ax=ax, annot_kws={"size":8})
    ax.tick_params(colors="white", labelsize=8)
    st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════
# PAGE 3 — PLAYER ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🏏 Player Analysis":
    st.markdown('<h2 class="section-header">Player Analysis</h2>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🏏 Batsmen", "🎳 Bowlers"])

    with tab1:
        bat = top_batsmen(deliveries, top_n=15)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 by Runs**")
            df = bat.head(10)
            fig, ax = dark_fig(6, 5)
            ax.barh(df["batter"][::-1], df["runs"][::-1], color="#F4A62A", edgecolor="none")
            ax.set_xlabel("Runs", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            st.markdown("**Top 10 by Strike Rate**")
            df2 = bat.sort_values("strike_rate", ascending=False).head(10)
            fig, ax = dark_fig(6, 5)
            ax.barh(df2["batter"][::-1], df2["strike_rate"][::-1], color="#E53935", edgecolor="none")
            ax.set_xlabel("Strike Rate", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            st.pyplot(fig, use_container_width=True); plt.close()
        st.dataframe(bat, use_container_width=True, hide_index=True)

    with tab2:
        bowl = top_bowlers(deliveries, top_n=15)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 by Wickets**")
            df = bowl.head(10)
            fig, ax = dark_fig(6, 5)
            ax.barh(df["bowler"][::-1], df["wickets"][::-1], color="#6A1B9A", edgecolor="none")
            ax.set_xlabel("Wickets", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            st.markdown("**Best Economy (min 120 balls)**")
            df2 = bowl.sort_values("economy").head(10)
            fig, ax = dark_fig(6, 5)
            ax.barh(df2["bowler"][::-1], df2["economy"][::-1], color="#00838F", edgecolor="none")
            ax.set_xlabel("Economy", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            st.pyplot(fig, use_container_width=True); plt.close()
        st.dataframe(bowl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 4 — MATCH INSIGHTS
# ═══════════════════════════════════════════════
elif page == "🎲 Match Insights":
    st.markdown('<h2 class="section-header">Match Insights</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🎲 Toss", "🔥 High Scores", "🏟 Venues"])

    with tab1:
        toss = toss_impact(matches)
        col1, col2 = st.columns(2)
        with col1:
            pct = toss["overall_toss_win_pct"]
            fig, ax = plt.subplots(figsize=(5,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            wedges, texts, autotexts = ax.pie(
                [pct, 100-pct], labels=["Toss+Win","Lost Toss, Won"],
                colors=["#F4A62A","#1565C0"], autopct="%1.1f%%", startangle=90,
                textprops={"color":"white"})
            ax.set_title("Toss → Match Win?", color="white", fontsize=12, fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            by_dec = toss["by_decision"]
            fig, ax = dark_fig(5, 5)
            ax.bar(by_dec["toss_decision"], by_dec["win_pct_when_toss_won"],
                   color=["#E53935","#2E7D32"], width=0.4, edgecolor="none")
            ax.set_ylabel("Win %", color="white"); ax.set_ylim(0, 100)
            ax.set_title("Win % by Decision", color="white", fontsize=12, fontweight="bold")
            ax.spines[:].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        hs = high_scoring_matches(matches, 380)
        hs_t = high_score_trend(matches, 380)
        st.metric("Total High-Scoring Matches (>380)", len(hs))
        fig, ax = dark_fig(10, 4)
        ax.fill_between(hs_t["season"], hs_t["high_score_count"], alpha=0.4, color="#E53935")
        ax.plot(hs_t["season"], hs_t["high_score_count"], marker="o", color="#E53935", lw=2)
        ax.set_title("High-Scoring Matches per Season", color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Season", color="white"); ax.set_ylabel("Count", color="white")
        ax.spines[:].set_visible(False); ax.yaxis.grid(True, color="white", alpha=0.1)
        st.pyplot(fig, use_container_width=True); plt.close()
        if not hs.empty:
            cols = [c for c in ["season","venue","team1","team1_runs","team2","team2_runs","match_total","winner"] if c in hs.columns]
            st.dataframe(hs[cols].head(20), use_container_width=True, hide_index=True)

    with tab3:
        venue = venue_analysis(matches)
        st.dataframe(venue, use_container_width=True, hide_index=True)
        fig, ax = dark_fig(11, 5)
        v = venue.head(8)
        x = range(len(v))
        width = 0.35
        ax.bar([i-width/2 for i in x], v["avg_first_innings"], width, label="1st Innings", color="#F4A62A", edgecolor="none")
        ax.bar([i+width/2 for i in x], v["avg_second_innings"], width, label="2nd Innings", color="#1565C0", edgecolor="none")
        ax.set_xticks(list(x))
        ax.set_xticklabels([v.split(" Stadium")[0][:15] for v in v["venue"]], rotation=30, ha="right", color="white", fontsize=8)
        ax.set_ylabel("Avg Runs", color="white"); ax.set_title("Avg Innings Scores by Venue", color="white", fontsize=12, fontweight="bold")
        ax.legend(facecolor="#1a1f35", labelcolor="white"); ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color="white", alpha=0.1)
        st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════
# PAGE 5 — PREDICT WINNER
# ═══════════════════════════════════════════════
elif page == "🤖 Predict Winner":
    st.markdown('<h2 class="section-header">🤖 Match Winner Prediction</h2>', unsafe_allow_html=True)

    with st.spinner("Training ML model …"):
        ml_results = get_ml(matches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", ml_results["best_model_name"])
    col2.metric("Accuracy", f"{ml_results['best_accuracy']*100:.1f}%")
    col3.metric("Random Forest", f"{ml_results['rf_acc']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🎯 Predict a Match")

    all_teams = sorted(matches["team1"].unique().tolist())
    all_venues = sorted(matches["venue"].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams, index=0)
        toss_winner = st.selectbox("Toss Winner", [team1, "Team 2 (select below)"], index=0)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    with col2:
        team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1], index=0)
        venue = st.selectbox("Venue", all_venues)

    # Fix toss winner selection
    toss_winner_actual = team1 if toss_winner == team1 else team2

    if st.button("🏏 Predict Winner"):
        winner = predict_match(ml_results, team1, team2,
                               toss_winner_actual, toss_decision, venue)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1f35,#252b45);border:2px solid #F4A62A;
                    border-radius:16px;padding:32px;text-align:center;margin-top:16px;">
            <h1 style="color:#F4A62A;font-family:Rajdhani;font-size:2.5rem;">🏆 Predicted Winner</h1>
            <h2 style="color:white;font-family:Rajdhani;font-size:2rem;">{winner}</h2>
            <p style="color:#8892b0;">Based on team identities, toss outcome, and venue history</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    st.markdown("---")
    st.markdown("### 📊 Feature Importances")
    imp = ml_results["feature_importances"]
    nice = {"team1_enc":"Team 1","team2_enc":"Team 2","toss_winner_enc":"Toss Winner",
            "toss_decision_enc":"Toss Decision","venue_enc":"Venue",
            "toss_win_flag":"Toss Win Flag","bat_first_flag":"Bat First"}
    imp_renamed = imp.rename(index=nice).sort_values(ascending=True)
    fig, ax = dark_fig(8, 4)
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(imp_renamed)))
    ax.barh(imp_renamed.index, imp_renamed.values, color=colors, edgecolor="none")
    ax.set_xlabel("Importance", color="white"); ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="white", alpha=0.1)
    st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════
# PAGE 6 — CAPS & AWARDS
# ═══════════════════════════════════════════════
elif page == "🏆 Caps & Awards":
    st.markdown('<h2 class="section-header">Caps & Season Awards</h2>', unsafe_allow_html=True)

    orange = orange_cap_winners(matches, deliveries)
    purple = purple_cap_winners(matches, deliveries)

    tab1, tab2, tab3 = st.tabs(["🟠 Orange Cap", "🟣 Purple Cap", "⚡ Super Overs"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(orange.rename(columns={"batter":"Player","runs":"Runs"}),
                         use_container_width=True, hide_index=True)
        with col2:
            fig, ax = dark_fig(7, 6)
            y = range(len(orange))
            ax.barh(y, orange["runs"], color="#FF8C00", alpha=0.8, edgecolor="none")
            ax.set_yticks(y)
            ax.set_yticklabels([f"{row['season']}  {row['batter']}" for _,row in orange.iterrows()],
                               color="white", fontsize=8)
            ax.set_xlabel("Runs", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            ax.set_title("🟠 Orange Cap Winners", color="white", fontsize=12, fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(purple.rename(columns={"bowler":"Player","wickets":"Wickets"}),
                         use_container_width=True, hide_index=True)
        with col2:
            fig, ax = dark_fig(7, 6)
            y = range(len(purple))
            ax.barh(y, purple["wickets"], color="#9C27B0", alpha=0.8, edgecolor="none")
            ax.set_yticks(y)
            ax.set_yticklabels([f"{row['season']}  {row['bowler']}" for _,row in purple.iterrows()],
                               color="white", fontsize=8)
            ax.set_xlabel("Wickets", color="white"); ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="white", alpha=0.1)
            ax.set_title("🟣 Purple Cap Winners", color="white", fontsize=12, fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        so = super_over_analysis(matches)
        st.metric("Total Super Overs", so["count"])
        col1, col2 = st.columns(2)
        with col1:
            if not so["per_season"].empty:
                fig, ax = dark_fig(6, 4)
                ax.bar(so["per_season"]["season"], so["per_season"]["super_overs"],
                       color="#00BCD4", alpha=0.85, edgecolor="none", width=0.7)
                ax.set_title("Super Overs per Season", color="white", fontsize=11, fontweight="bold")
                ax.set_xlabel("Season", color="white"); ax.spines[:].set_visible(False)
                ax.yaxis.grid(True, color="white", alpha=0.1)
                st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            if not so["winner_breakdown"].empty:
                wb = so["winner_breakdown"].head(6)
                fig, ax = dark_fig(6, 4)
                ax.bar(wb.index, wb.values, color=IPL_COLORS[:len(wb)], edgecolor="none")
                ax.set_xticklabels([t.split()[-1] for t in wb.index], rotation=30, ha="right", color="white")
                ax.set_title("Super Over Wins by Team", color="white", fontsize=11, fontweight="bold")
                ax.spines[:].set_visible(False); ax.yaxis.grid(True, color="white", alpha=0.1)
                st.pyplot(fig, use_container_width=True); plt.close()


# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#3d4a6b;font-size:0.8rem;">IPL Cricket Analysis System | Built with Python, Pandas, Scikit-Learn & Streamlit</p>',
            unsafe_allow_html=True)
