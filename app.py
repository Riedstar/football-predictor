"""
=============================================================================
Title: Football Match Predictor & Analytics Dashboard (Modern Broadcast UI)
Description: Full-stack match prediction engine with glassmorphism team cards,
             3-way visual probability bars, scoreline heatbars, and a live
             multi-league Accumulator Slip.
Author: Riedwaan Kops
=============================================================================
"""
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="PitchPulse • Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "bet_slip" not in st.session_state:
  st.session_state.bet_slip = []

if "last_prediction" not in st.session_state:
  st.session_state.last_prediction = None

# ==========================================
# 2. MODERN BROADCAST & GLASSMORPHISM CSS
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #111c33 0%, #090e1a 55%, #060911 100%);
        color: #f8fafc;
    }

    /* Hero Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    /* Kickoff Pill */
    .kickoff-banner {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 999px;
        padding: 10px 20px;
        text-align: center;
        font-weight: 600;
        font-size: 13px;
        color: #93c5fd;
        margin: 10px 0 22px 0;
        letter-spacing: 0.3px;
    }

    /* Glass Team Card */
    .team-card {
        background: linear-gradient(160deg, rgba(30, 41, 59, 0.78) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 24px 18px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .team-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.45);
    }

    .crest-glow {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 96px;
        height: 96px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.18) 0%, rgba(15, 23, 42, 0) 72%);
        margin-bottom: 6px;
    }

    .vs-badge {
        width: 52px;
        height: 52px;
        margin: 115px auto 0 auto;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border: 2px solid rgba(147, 197, 253, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 16px;
        color: #ffffff;
        box-shadow: 0 0 24px rgba(37, 99, 235, 0.45);
    }

    .mono-stat {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Form Badges */
    .form-badge {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 11px;
        color: #ffffff;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }
    .form-W { background: linear-gradient(135deg, #10b981, #059669); }
    .form-D { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .form-L { background: linear-gradient(135deg, #ef4444, #dc2626); }

    /* Probability Bar Container */
    .prob-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .prob-bar-track {
        display: flex;
        width: 100%;
        height: 16px;
        border-radius: 999px;
        overflow: hidden;
        background: #1e293b;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 3. LOAD LEAGUE & FIXTURE DATABASE
# ==========================================
def load_league_database():
  with open("fixtures.json", "r", encoding="utf-8") as f:
    return json.load(f)


LEAGUE_DATABASE = load_league_database()


# ==========================================
# 4. HELPER FUNCTIONS & POISSON MODEL
# ==========================================
def convert_to_sast(time_str: str) -> str:
  if "SAST" in time_str:
    return time_str
  try:
    dt_utc = datetime.strptime(time_str.strip(), "%H:%M")
    dt_sast = dt_utc + timedelta(hours=2)
    return f"{dt_sast.strftime('%H:%M')} SAST"
  except Exception:
    return f"{time_str} SAST"


def render_form_badges(form_str: str) -> str:
  badges = []
  for char in form_str.upper():
    if char in ("W", "D", "L"):
      badges.append(f'<span class="form-badge form-{char}">{char}</span>')
  return "".join(badges)


def calculate_season_xg(team_data: dict, opp_data: dict, is_home: bool = True) -> float:
  form_str = team_data.get("full_form", team_data.get("last_5", "WWDWL"))
  weights = {"W": 0.34, "D": 0.16, "L": 0.05}
  clean = [c for c in form_str.upper() if c in weights]
  form_factor = (
      0.85 + (sum(weights[c] for c in clean) / len(clean)) if clean else 1.0
  )

  avg_gf = team_data.get("avg_gf", 1.35)
  opp_ga = opp_data.get("avg_ga", 1.25)
  base_xg = ((avg_gf + opp_ga) / 2.0) * form_factor

  if is_home:
    base_xg *= 1.12
  return round(max(0.45, min(3.20, base_xg)), 2)


def predict_match(home_xg: float, away_xg: float, max_goals: int = 7) -> dict:
  home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
  away_probs = [poisson.pmf(j, away_xg) for j in range(max_goals)]
  score_matrix = np.outer(home_probs, away_probs)

  home_win = float(np.sum(np.tril(score_matrix, -1))) * 100
  draw = float(np.sum(np.diag(score_matrix))) * 100
  away_win = float(np.sum(np.triu(score_matrix, 1))) * 100

  over_2_5 = 0.0
  under_2_5 = 0.0
  btts_yes = 0.0
  scorelines = []

  for h in range(max_goals):
    for a in range(max_goals):
      p = score_matrix[h, a] * 100
      if h + a > 2.5:
        over_2_5 += p
      else:
        under_2_5 += p
      if h > 0 and a > 0:
        btts_yes += p
      scorelines.append((f"{h} - {a}", round(p, 2)))

  scorelines.sort(key=lambda x: x[1], reverse=True)

  return {
      "home_win_prob": round(home_win, 1),
      "draw_prob": round(draw, 1),
      "away_win_prob": round(away_win, 1),
      "over_2_5_prob": round(over_2_5, 1),
      "under_2_5_prob": round(under_2_5, 1),
      "btts_yes_prob": round(btts_yes, 1),
      "top_scores": scorelines[:5],
  }


def render_team_card(team_name: str, t_data: dict, side_label: str) -> str:
  logo_html = (
      f'<div class="crest-glow"><img src="{t_data["logo"]}" width="78"></div>'
      if t_data.get("logo")
      else '<div class="crest-glow" style="font-size:42px;">🛡️</div>'
  )
  pos = t_data.get("position", "-")
  rec = t_data.get("record", "-")
  last_5 = t_data.get("last_5", t_data.get("form", "WWDWL"))
  badges = render_form_badges(last_5)

  return f"""
    <div class="team-card">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#60a5fa; font-weight:700; margin-bottom:6px;">
            {side_label}
        </div>
        {logo_html}
        <h2 style="margin:4px 0 14px 0; font-weight:800; letter-spacing:-0.4px;">{team_name}</h2>
        <div style="display:flex; justify-content:space-around; align-items:center;
                    border-top:1px solid rgba(148,163,184,0.16); padding-top:14px;">
            <div style="flex:0.8;">
                <div class="mono-stat" style="font-size:18px; font-weight:800; color:#ffffff;">#{pos}</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:2px;">League Pos</div>
            </div>
            <div style="width:1px; height:34px; background:rgba(148,163,184,0.2);"></div>
            <div style="flex:1.1;">
                <div class="mono-stat" style="font-size:15px; font-weight:800; color:#e2e8f0;">{rec}</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Season Record</div>
            </div>
            <div style="width:1px; height:34px; background:rgba(148,163,184,0.2);"></div>
            <div style="flex:1.5;">
                <div>{badges}</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:4px;">Last 5 (Oldest → Latest)</div>
            </div>
        </div>
    </div>
    """


# ==========================================
# 5. MAIN DASHBOARD HEADER & SELECTORS
# ==========================================
st.markdown(
    """
    <div class="hero-header">
        <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#60a5fa;">
            Live Poisson Analytics Engine
        </div>
        <h1 style="margin:4px 0 6px 0; font-size:32px; font-weight:800;">⚽ Football Match Predictor</h1>
        <p style="margin:0; color:#cbd5e1; font-size:14px;">
            Automated daily fixtures, live standings &amp; form-calibrated Expected Goals (xG) with multi-league Accumulator Builder.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not LEAGUE_DATABASE:
  st.error("No fixtures found in `fixtures.json`. Run `python fetch_fixtures.py` first.")
  st.stop()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
  selected_league = st.selectbox(
      "🏆 Competition / Tournament", list(LEAGUE_DATABASE.keys())
  )

league_data = LEAGUE_DATABASE[selected_league]
teams_info = league_data.get("teams", {})
fixtures_list = league_data.get("fixtures", [])

fixture_labels = [
    f"{fx['home']} vs {fx['away']} ({fx['date']} • {convert_to_sast(fx['time'])})"
    for fx in fixtures_list
]

with col_sel2:
  selected_idx = st.selectbox(
      "📅 Scheduled Match Fixture",
      range(len(fixture_labels)),
      format_func=lambda i: fixture_labels[i],
  )

current_fx = fixtures_list[selected_idx]
home_team = current_fx["home"]
away_team = current_fx["away"]
match_date = current_fx["date"]
match_time = convert_to_sast(current_fx["time"])

home_data = teams_info.get(home_team, {"logo": "", "last_5": "WWDWL"})
away_data = teams_info.get(away_team, {"logo": "", "last_5": "WDWLL"})

st.markdown(
    f"""
    <div class="kickoff-banner">
        🕒 KICK-OFF: <span class="mono-stat" style="color:#ffffff;">{match_date} @ {match_time}</span>
        &nbsp;&nbsp;•&nbsp;&nbsp;
        🏟️ <span style="color:#ffffff;">{selected_league}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 6. TEAM COMPARISON CARDS & xG INPUTS
# ==========================================
col_home, col_vs, col_away = st.columns([5, 1, 5])

with col_home:
  st.markdown(render_team_card(home_team, home_data, "🏠 Home Side"), unsafe_allow_html=True)
  default_home_xg = calculate_season_xg(home_data, away_data, is_home=True)
  home_xg = st.slider(
      f"{home_team} Expected Goals (xG)",
      min_value=0.30,
      max_value=3.50,
      value=default_home_xg,
      step=0.05,
      key=f"hxg_{selected_league}_{selected_idx}",
  )

with col_vs:
  st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

with col_away:
  st.markdown(render_team_card(away_team, away_data, "✈️ Away Side"), unsafe_allow_html=True)
  default_away_xg = calculate_season_xg(away_data, home_data, is_home=False)
  away_xg = st.slider(
      f"{away_team} Expected Goals (xG)",
      min_value=0.30,
      max_value=3.50,
      value=default_away_xg,
      step=0.05,
      key=f"axg_{selected_league}_{selected_idx}",
  )

# ==========================================
# 7. RUN PREDICTION & VISUAL PROBABILITY BAR
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run Match Prediction", type="primary", use_container_width=True):
  pred = predict_match(home_xg, away_xg)
  st.session_state.last_prediction = {
      "league": selected_league,
      "home_team": home_team,
      "away_team": away_team,
      "match_date": match_date,
      "match_time": match_time,
      "home_xg": home_xg,
      "away_xg": away_xg,
      "result": pred,
  }

saved = st.session_state.last_prediction
if (
    saved
    and saved["league"] == selected_league
    and saved["home_team"] == home_team
    and saved["away_team"] == away_team
):
  res = saved["result"]
  hw = res["home_win_prob"]
  dr = res["draw_prob"]
  aw = res["away_win_prob"]

  st.markdown("---")
  st.markdown(
      f"""
      <div class="prob-card">
          <div style="font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#93c5fd;">
              📊 1X2 Win Probability Distribution
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:12px; font-weight:800; font-size:15px;">
              <span style="color:#10b981;">🏠 {home_team}: <span class="mono-stat">{hw}%</span></span>
              <span style="color:#f59e0b;">🤝 Draw: <span class="mono-stat">{dr}%</span></span>
              <span style="color:#38bdf8;">✈️ {away_team}: <span class="mono-stat">{aw}%</span></span>
          </div>
          <div class="prob-bar-track">
              <div style="width:{hw}%; background:linear-gradient(90deg, #10b981, #059669);"></div>
              <div style="width:{dr}%; background:linear-gradient(90deg, #f59e0b, #d97706);"></div>
              <div style="width:{aw}%; background:linear-gradient(90deg, #0284c7, #38bdf8);"></div>
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )

  g1, g2, g3 = st.columns(3)
  g1.metric("🔥 Over 2.5 Goals", f"{res['over_2_5_prob']}%")
  g2.metric("🛡️ Under 2.5 Goals", f"{res['under_2_5_prob']}%")
  g3.metric("⚽ Both Teams to Score (Yes)", f"{res['btts_yes_prob']}%")

  st.markdown("#### 🎯 Most Likely Exact Scorelines")
  max_score_prob = max((p for _, p in res["top_scores"]), default=1.0)
  for score_str, prob_val in res["top_scores"]:
    bar_width = min(100, round((prob_val / max_score_prob) * 100, 1))
    st.markdown(
        f"""
        <div style="background:rgba(30,41,59,0.65); border:1px solid rgba(148,163,184,0.15);
                    border-radius:10px; padding:10px 14px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; font-weight:700; font-size:14px; margin-bottom:5px;">
                <span class="mono-stat" style="color:#ffffff;">{score_str}</span>
                <span class="mono-stat" style="color:#60a5fa;">{prob_val}%</span>
            </div>
            <div style="width:100%; height:7px; background:#0f172a; border-radius:999px; overflow:hidden;">
                <div style="width:{bar_width}%; height:100%; background:linear-gradient(90deg, #2563eb, #38bdf8);"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  # ==========================================
  # 8. ADD SELECTION TO ACCUMULATOR SLIP
  # ==========================================
  st.markdown("---")
  st.subheader("🎟️ Add Selection to Accumulator Bet Slip")

  market_options = {
      f"{home_team} Win": res["home_win_prob"],
      "Draw": res["draw_prob"],
      f"{away_team} Win": res["away_win_prob"],
      "Over 2.5 Goals": res["over_2_5_prob"],
      "Under 2.5 Goals": res["under_2_5_prob"],
      "BTTS - Yes": res["btts_yes_prob"],
  }

  col_add1, col_add2 = st.columns([3, 1])
  with col_add1:
    chosen_market = st.selectbox(
        "Select Market Leg",
        list(market_options.keys()),
        key=f"mkt_{selected_league}_{selected_idx}",
    )
  with col_add2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➕ Add to Slip", use_container_width=True):
      leg_prob = market_options[chosen_market]
      fixture_title = f"{home_team} vs {away_team}"

      st.session_state.bet_slip = [
          leg for leg in st.session_state.bet_slip if leg["Fixture"] != fixture_title
      ]
      st.session_state.bet_slip.append({
          "Competition": selected_league,
          "Fixture": fixture_title,
          "Kick-off": f"{match_date} {match_time}",
          "Selection": chosen_market,
          "Probability (%)": leg_prob,
          "Fair Odds": round(100.0 / leg_prob, 2) if leg_prob > 0 else 0.0,
      })
      st.toast(f"Added {chosen_market} ({leg_prob}%) to Accumulator Slip!", icon="🎟️")

# ==========================================
# 9. ACCUMULATOR BET SLIP SIDEBAR
# ==========================================
with st.sidebar:
  st.markdown("## 🎟️ Accumulator Slip")
  if not st.session_state.bet_slip:
    st.info(
        "Your slip is empty. Run a prediction on any match and click **➕ Add to Slip**"
        " to build a multi-bet across leagues."
    )
  else:
    combined_prob = 1.0
    for idx, leg in enumerate(st.session_state.bet_slip):
      combined_prob *= leg["Probability (%)"] / 100.0
      st.markdown(
          f"""
          <div style="background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                      border-left: 4px solid #10b981; border-top: 1px solid rgba(148,163,184,0.18);
                      border-right: 1px solid rgba(148,163,184,0.18); border-bottom: 1px solid rgba(148,163,184,0.18);
                      border-radius: 12px; padding: 12px; margin-bottom: 10px;">
              <div style="font-size: 11px; color: #93c5fd;">{leg['Competition']} • {leg['Kick-off']}</div>
              <div style="font-weight: 700; color: #ffffff; font-size: 14px; margin: 3px 0;">{leg['Fixture']}</div>
              <div class="mono-stat" style="color: #10b981; font-weight: 800; font-size: 13px;">
                  {leg['Selection']} • {leg['Probability (%)']}% ({leg['Fair Odds']}x)
              </div>
          </div>
          """,
          unsafe_allow_html=True,
      )
      if st.button("❌ Remove", key=f"rm_leg_{idx}"):
        st.session_state.bet_slip.pop(idx)
        st.rerun()

    combined_pct = round(combined_prob * 100, 2)
    fair_acc_odds = round(1.0 / combined_prob, 2) if combined_prob > 0 else 0.0

    st.markdown("---")
    stake = st.number_input(
        "Simulated Stake (ZAR)", min_value=10.0, value=50.0, step=10.0
    )
    projected_payout = round(stake * fair_acc_odds, 2)

    st.metric("Total Legs", len(st.session_state.bet_slip))
    st.metric("Combined Hit Probability", f"{combined_pct}%")
    st.metric("Fair Accumulator Odds", f"{fair_acc_odds}x")
    st.metric("Projected Return (ZAR)", f"R {projected_payout:,.2f}")

    slip_df = pd.DataFrame(st.session_state.bet_slip)
    st.download_button(
        "📥 Export Slip to CSV",
        data=slip_df.to_csv(index=False).encode("utf-8"),
        file_name="football_accumulator_slip.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("🗑️ Clear Entire Slip", use_container_width=True):
      st.session_state.bet_slip = []
      st.rerun()

# ==========================================
# 10. TECH STACK FOOTER
# ==========================================
st.markdown("---")
st.caption(
    "PitchPulse Analytics • Powered by Python, Streamlit, SciPy Poisson Model & Automated GitHub Actions Pipeline"
)