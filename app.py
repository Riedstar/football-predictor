"""
=============================================================================
Title: Football Match Predictor & Analytics Dashboard (Full Featured)
Description: A full-stack data science web application featuring enlarged 
             glass headers, visual form chips, H2H factors, and a CSV 
             Match Report Export feature.
Author: Riedwaan Kops
=============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Football Match Predictor", 
    page_icon="⚽", 
    layout="wide"
)

# --- 2. ADVANCED SPECTACULAR GLASSMORPHISM & MOBILE CSS STYLING ---
st.markdown("""
<style>
    /* Premium Deep Ambient Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a233a 0%, #0b0f19 70%);
        color: #f3f4f6;
    }
    
    /* High-End Multi-Layered Glassmorphism Card */
    .glass-card {
        background: rgba(20, 24, 33, 0.75);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.18);
        padding: 32px 24px;
        border-radius: 24px;
        margin-bottom: 20px;
        box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 60px 0 rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Sleek Streamlit Selectboxes */
    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
    }
    
    div[data-baseweb="popover"] div {
        background-color: #0f172a !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Glowing Primary Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-radius: 14px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.45);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.65);
        transform: translateY(-2px);
    }

    .metric-title {
        color: #93c5fd;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
        margin-top: 6px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Frosted Match Info Badge */
    .match-info-badge {
        background: rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 14px 22px;
        border-radius: 16px;
        text-align: center;
        color: #93c5fd;
        font-weight: 600;
        margin-bottom: 25px;
        font-size: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }

    /* Form Chip Badges */
    .form-chip-win {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        background-color: #10b981;
        color: #ffffff;
        font-weight: 800;
        font-size: 12px;
        border-radius: 6px;
        margin: 0 2px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.4);
    }
    .form-chip-draw {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        background-color: #f59e0b;
        color: #ffffff;
        font-weight: 800;
        font-size: 12px;
        border-radius: 6px;
        margin: 0 2px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.4);
    }
    .form-chip-loss {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        background-color: #ef4444;
        color: #ffffff;
        font-weight: 800;
        font-size: 12px;
        border-radius: 6px;
        margin: 0 2px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.4);
    }

    /* Enlarged Glassmorphism Header Icon Badge */
    .icon-glass-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 88px;
        height: 88px;
        background: rgba(20, 24, 33, 0.75);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        box-shadow: 0 16px 40px 0 rgba(0, 0, 0, 0.5), 0 0 25px rgba(59, 130, 246, 0.15);
        font-size: 42px;
        margin-bottom: 16px;
        transition: transform 0.3s ease;
    }

    .icon-glass-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.6), 0 0 35px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FULLY POPULATED LEAGUE & TOURNAMENT DATABASE ---
LEAGUE_DATABASE = {
    "England - Premier League": {
        "teams": {
            "Arsenal": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t3.png", "form": "WWWDWW"},
            "Newcastle United": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t4.png", "form": "DWWWDL"},
            "Liverpool": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t14.png", "form": "WWWWWW"},
            "Manchester City": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t43.png", "form": "WDWWLW"},
            "Chelsea": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t8.png", "form": "WDWLWW"},
            "Manchester United": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t1.png", "form": "LDWLWW"},
            "Tottenham Hotspur": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t6.png", "form": "WWLWDW"},
            "Aston Villa": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t7.png", "form": "WDWWWW"},
            "Brighton": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t36.png", "form": "WWDWLW"},
            "Nottingham Forest": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t17.png", "form": "DWLWWL"},
            "West Ham United": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t21.png", "form": "LLDWWW"},
            "Crystal Palace": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t31.png", "form": "WDDWLW"},
            "Brentford": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t94.png", "form": "WLWDLW"},
            "Fulham": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t54.png", "form": "DDWWDL"},
            "Everton": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t11.png", "form": "LDWDLL"},
            "Bournemouth": {"logo": "https://resources.premierleague.com/premierleague/badges/70/t91.png", "form": "WDLLWW"}
        },
        "fixtures": [
            {"home": "Arsenal", "away": "Newcastle United", "date": "2026-11-21", "time": "14:30 SAST"},
            {"home": "Liverpool", "away": "Manchester City", "date": "2026-11-21", "time": "17:00 SAST"},
            {"home": "Chelsea", "away": "Manchester United", "date": "2026-11-21", "time": "17:00 SAST"},
            {"home": "Tottenham Hotspur", "away": "Aston Villa", "date": "2026-11-21", "time": "17:00 SAST"},
            {"home": "Brighton", "away": "Nottingham Forest", "date": "2026-11-21", "time": "17:00 SAST"},
            {"home": "West Ham United", "away": "Crystal Palace", "date": "2026-11-21", "time": "19:30 SAST"},
            {"home": "Brentford", "away": "Fulham", "date": "2026-11-21", "time": "22:00 SAST"},
            {"home": "Everton", "away": "Bournemouth", "date": "2026-11-22", "time": "16:00 SAST"}
        ]
    },
    "Spain - La Liga": {
        "teams": {
            "Real Madrid": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg", "form": "WWWDWW"},
            "Barcelona": {"logo": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg", "form": "WWWWDL"},
            "Atletico Madrid": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/f4/Atletico_Madrid_2017_logo.svg", "form": "WDWWDW"},
            "Villarreal": {"logo": "https://upload.wikimedia.org/wikipedia/en/7/70/Villarreal_CF_logo.svg", "form": "LDWWLW"},
            "Real Sociedad": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/f1/Real_Sociedad_logo.svg", "form": "WWDLLW"},
            "Athletic Bilbao": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/98/Athletic_Club_Bilbao_%28crest%29.svg", "form": "WDWWDL"},
            "Real Betis": {"logo": "https://upload.wikimedia.org/wikipedia/en/1/13/Real_Betis_Balompi%C3%A9_Logo.svg", "form": "DWWLDW"},
            "Valencia": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/ce/Valencia_CF_logo.svg", "form": "LDLLWD"}
        },
        "fixtures": [
            {"home": "Real Madrid", "away": "Barcelona", "date": "2026-11-21", "time": "22:00 SAST"},
            {"home": "Atletico Madrid", "away": "Villarreal", "date": "2026-11-21", "time": "19:30 SAST"},
            {"home": "Real Sociedad", "away": "Athletic Bilbao", "date": "2026-11-21", "time": "17:15 SAST"},
            {"home": "Real Betis", "away": "Valencia", "date": "2026-11-22", "time": "19:00 SAST"}
        ]
    },
    "Germany - Bundesliga": {
        "teams": {
            "Bayern Munich": {"logo": "https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282002%29.svg", "form": "WWWWWW"},
            "Borussia Dortmund": {"logo": "https://upload.wikimedia.org/wikipedia/commons/6/67/Borussia_Dortmund_logo.svg", "form": "WDWWLW"},
            "Bayer Leverkusen": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/59/Bayer_04_Leverkusen_logo.svg", "form": "WWWDWW"},
            "RB Leipzig": {"logo": "https://upload.wikimedia.org/wikipedia/en/0/04/RB_Leipzig_2014_logo.svg", "form": "DWWLDW"},
            "Eintracht Frankfurt": {"logo": "https://upload.wikimedia.org/wikipedia/en/0/04/Eintracht_Frankfurt_Logo.svg", "form": "WDWLWW"},
            "VfB Stuttgart": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/eb/VfB_Stuttgart_1893_logo.svg", "form": "WWDLWW"}
        },
        "fixtures": [
            {"home": "Bayern Munich", "away": "Borussia Dortmund", "date": "2026-11-21", "time": "19:30 SAST"},
            {"home": "Bayer Leverkusen", "away": "RB Leipzig", "date": "2026-11-21", "time": "16:30 SAST"},
            {"home": "Eintracht Frankfurt", "away": "VfB Stuttgart", "date": "2026-11-21", "time": "16:30 SAST"}
        ]
    },
    "Italy - Serie A": {
        "teams": {
            "Inter Milan": {"logo": "https://upload.wikimedia.org/wikipedia/commons/0/05/FC_Internazionale_Milano_2021.svg", "form": "WWWDWW"},
            "AC Milan": {"logo": "https://upload.wikimedia.org/wikipedia/commons/d/d0/Logo_AC_Milan.svg", "form": "WDWWLW"},
            "Juventus": {"logo": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Juventus_FC_-_logo_black_%28Italy%2C_2017%29.svg", "form": "DDWWWW"},
            "Napoli": {"logo": "https://upload.wikimedia.org/wikipedia/commons/2/28/SSC_Neapel_Logo.svg", "form": "WWLWDW"},
            "AS Roma": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/f7/AS_Roma_Logo_%282013%29.svg", "form": "WDWWDL"},
            "Lazio": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/ce/SS_Lazio_1900_logo.svg", "form": "WWDWLW"}
        },
        "fixtures": [
            {"home": "Inter Milan", "away": "AC Milan", "date": "2026-11-21", "time": "21:45 SAST"},
            {"home": "Juventus", "away": "Napoli", "date": "2026-11-21", "time": "19:00 SAST"},
            {"home": "AS Roma", "away": "Lazio", "date": "2026-11-21", "time": "16:00 SAST"}
        ]
    },
    "France - Ligue 1": {
        "teams": {
            "Paris Saint-Germain": {"logo": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg", "form": "WWWWWW"},
            "Marseille": {"logo": "https://upload.wikimedia.org/wikipedia/en/d/d8/Olympique_Marseille_logo.svg", "form": "WDWDLW"},
            "Monaco": {"logo": "https://upload.wikimedia.org/wikipedia/en/b/ba/AS_Monaco_FC.svg", "form": "WWLWWW"},
            "Lille": {"logo": "https://upload.wikimedia.org/wikipedia/en/a/af/LOSC_Lille_logo_2018.svg", "form": "LDWWLW"}
        },
        "fixtures": [
            {"home": "Paris Saint-Germain", "away": "Marseille", "date": "2026-11-21", "time": "22:00 SAST"},
            {"home": "Monaco", "away": "Lille", "date": "2026-11-22", "time": "16:00 SAST"}
        ]
    },
    "Netherlands - Eredivisie": {
        "teams": {
            "Ajax": {"logo": "https://upload.wikimedia.org/wikipedia/en/7/79/Ajax_Amsterdam.svg", "form": "WWWDWW"},
            "PSV Eindhoven": {"logo": "https://upload.wikimedia.org/wikipedia/en/0/0e/PSV_Eindhoven_%28crest%29.svg", "form": "WWWWWW"},
            "Feyenoord": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/e5/Feyenoord_logo.svg", "form": "WDWWDL"}
        },
        "fixtures": [
            {"home": "Ajax", "away": "PSV Eindhoven", "date": "2026-11-21", "time": "15:30 SAST"},
            {"home": "Feyenoord", "away": "Ajax", "date": "2026-11-22", "time": "17:45 SAST"}
        ]
    },
    "Portugal - Primeira Liga": {
        "teams": {
            "Benfica": {"logo": "https://upload.wikimedia.org/wikipedia/en/a/a2/SL_Benfica_logo.svg", "form": "WWWWWW"},
            "FC Porto": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/f1/FC_Porto.svg", "form": "WDWWWD"},
            "Sporting CP": {"logo": "https://upload.wikimedia.org/wikipedia/en/3/3e/Sporting_Clube_de_Portugal_%28Logo%29.svg", "form": "WWLWWW"}
        },
        "fixtures": [
            {"home": "Benfica", "away": "FC Porto", "date": "2026-11-21", "time": "22:30 SAST"},
            {"home": "Sporting CP", "away": "Benfica", "date": "2026-11-22", "time": "20:00 SAST"}
        ]
    },
    "International - FIFA World Cup": {
        "teams": {
            "Brazil": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/99/Brazil_national_football_team_logo.svg", "form": "WWWDWW"}, 
            "France": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/c3/Fff_logo_2018.svg", "form": "WWWWWW"},
            "Argentina": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/c1/Argentina_national_football_team_logo.svg", "form": "WWWDW"}, 
            "England": {"logo": "https://upload.wikimedia.org/wikipedia/en/2/2c/England_national_football_team_crest.svg", "form": "WDWWLW"}
        },
        "fixtures": [
            {"home": "Brazil", "away": "Argentina", "date": "2026-11-21", "time": "21:00 SAST"},
            {"home": "France", "away": "England", "date": "2026-11-21", "time": "00:00 SAST"}
        ]
    },
    "International - UEFA European Championship (Euros)": {
        "teams": {
            "Spain": {"logo": "https://upload.wikimedia.org/wikipedia/en/6/6f/Spain_national_football_team_crest.svg", "form": "WWWWWW"}, 
            "Germany": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/f6/Germany_national_football_team_crest_%282021%29.svg", "form": "WDWWWD"},
            "France": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/c3/Fff_logo_2018.svg", "form": "WWWWWW"}, 
            "Portugal": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/5f/Portuguese_Football_Federation_crest_2021.svg", "form": "WWWDWW"}
        },
        "fixtures": [
            {"home": "Spain", "away": "Germany", "date": "2026-11-21", "time": "22:00 SAST"},
            {"home": "France", "away": "Portugal", "date": "2026-11-22", "time": "22:00 SAST"}
        ]
    },
    "International - Africa Cup of Nations (AFCON)": {
        "teams": {
            "Morocco": {"logo": "https://upload.wikimedia.org/wikipedia/en/3/36/Royal_Moroccan_Football_Federation_logo.svg", "form": "WWWWWW"}, 
            "Senegal": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/91/Senegal_Football_Federation.svg", "form": "WWWDWW"},
            "Nigeria": {"logo": "https://upload.wikimedia.org/wikipedia/en/6/6c/Nigeria_football_association_logo.svg", "form": "WDWWLW"}, 
            "Egypt": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/52/Egyptian_Football_Association_logo.svg", "form": "WWLWDW"},
            "South Africa": {"logo": "https://upload.wikimedia.org/wikipedia/en/3/36/South_Africa_national_football_team_seal.svg", "form": "WDWDW"}
        },
        "fixtures": [
            {"home": "Morocco", "away": "Senegal", "date": "2026-11-21", "time": "19:00 SAST"},
            {"home": "Nigeria", "away": "Egypt", "date": "2026-11-22", "time": "17:00 SAST"},
            {"home": "South Africa", "away": "Nigeria", "date": "2026-11-23", "time": "16:00 SAST"}
        ]
    },
    "Belgium - Belgian Pro League": {
        "teams": {
            "Anderlecht": {"logo": "https://upload.wikimedia.org/wikipedia/en/2/27/RSC_Anderlecht_logo.svg", "form": "WDWWL"}, 
            "Club Brugge": {"logo": "https://upload.wikimedia.org/wikipedia/en/b/b6/Club_Brugge_KV_logo.svg", "form": "WWWWDL"}
        },
        "fixtures": [{"home": "Club Brugge", "away": "Anderlecht", "date": "2026-11-21", "time": "19:00 SAST"}]
    },
    "Scotland - Scottish Premiership": {
        "teams": {
            "Celtic": {"logo": "https://upload.wikimedia.org/wikipedia/en/3/35/Celtic_FC.svg", "form": "WWWWWW"}, 
            "Rangers": {"logo": "https://upload.wikimedia.org/wikipedia/en/4/43/Rangers_FC.svg", "form": "WDWWLW"}
        },
        "fixtures": [{"home": "Celtic", "away": "Rangers", "date": "2026-11-21", "time": "14:30 SAST"}]
    },
    "Turkey - Süper Lig": {
        "teams": {
            "Galatasaray": {"logo": "https://upload.wikimedia.org/wikipedia/en/b/c3/Galatasaray_SK_Logo_%282020%29.svg", "form": "WWWWWW"}, 
            "Fenerbahçe": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/e9/Fenerbah%C3%A7e_SK.svg", "form": "WWWDWW"}
        },
        "fixtures": [{"home": "Galatasaray", "away": "Fenerbahçe", "date": "2026-11-21", "time": "20:00 SAST"}]
    },
    "Greece - Super League Greece": {
        "teams": {
            "Olympiacos": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/eb/Olympiacos_FC_logo.svg", "form": "WWWDW"}, 
            "Panathinaikos": {"logo": "https://upload.wikimedia.org/wikipedia/en/6/62/Panathinaikos_F.C._logo.svg", "form": "WDWDL"}
        },
        "fixtures": [{"home": "Olympiacos", "away": "Panathinaikos", "date": "2026-11-21", "time": "21:00 SAST"}]
    },
    "Switzerland - Swiss Super League": {
        "teams": {
            "Young Boys": {"logo": "https://upload.wikimedia.org/wikipedia/en/2/2f/BSC_Young_Boys_logo.svg", "form": "WWWDW"}, 
            "Basel": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/c5/FC_Basel_1893_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Young Boys", "away": "Basel", "date": "2026-11-21", "time": "17:30 SAST"}]
    },
    "Austria - Austrian Bundesliga": {
        "teams": {
            "Red Bull Salzburg": {"logo": "https://upload.wikimedia.org/wikipedia/en/7/77/FC_Red_Bull_Salzburg_logo.svg", "form": "WWWWWW"}, 
            "Sturm Graz": {"logo": "https://upload.wikimedia.org/wikipedia/en/2/21/SK_Sturm_Graz_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Red Bull Salzburg", "away": "Sturm Graz", "date": "2026-11-21", "time": "18:00 SAST"}]
    },
    "Denmark - Danish Superliga": {
        "teams": {
            "FC Copenhagen": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/95/F.C._Copenhagen_logo.svg", "form": "WWWDW"}, 
            "FC Midtjylland": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/e0/FC_Midtjylland_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "FC Copenhagen", "away": "FC Midtjylland", "date": "2026-11-21", "time": "17:00 SAST"}]
    },
    "Norway - Eliteserien": {
        "teams": {
            "Bodø/Glimt": {"logo": "https://upload.wikimedia.org/wikipedia/en/8/86/FK_Bod%C3%B8_Glimt_logo.svg", "form": "WWWWWW"}, 
            "Molde": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/52/Molde_FK_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Bodø/Glimt", "away": "Molde", "date": "2026-11-21", "time": "19:00 SAST"}]
    },
    "Sweden - Allsvenskan": {
        "teams": {
            "Malmö FF": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/9a/Malm%C3%B6_FF_logo.svg", "form": "WWWWWW"}, 
            "AIK": {"logo": "https://upload.wikimedia.org/wikipedia/en/9/92/AIK_Fotboll_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Malmö FF", "away": "AIK", "date": "2026-11-21", "time": "16:00 SAST"}]
    },
    "Poland - Ekstraklasa": {
        "teams": {
            "Legia Warsaw": {"logo": "https://upload.wikimedia.org/wikipedia/en/1/18/Legia_Warsaw_crest.svg", "form": "WWWDW"}, 
            "Lech Poznań": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/5d/Lech_Pozna%C5%84_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Legia Warsaw", "away": "Lech Poznań", "date": "2026-11-21", "time": "21:30 SAST"}]
    },
    "Czech Republic - Czech First League": {
        "teams": {
            "Slavia Prague": {"logo": "https://upload.wikimedia.org/wikipedia/en/f/fb/SK_Slavia_Prague_logo.svg", "form": "WWWWWW"}, 
            "Sparta Prague": {"logo": "https://upload.wikimedia.org/wikipedia/en/5/5b/AC_Sparta_Prague_logo.svg", "form": "WWWDW"}
        },
        "fixtures": [{"home": "Slavia Prague", "away": "Sparta Prague", "date": "2026-11-21", "time": "19:00 SAST"}]
    },
    "Serbia - Serbian SuperLiga": {
        "teams": {
            "Red Star Belgrade": {"logo": "https://upload.wikimedia.org/wikipedia/en/1/14/FK_Crvena_Zvezda.svg", "form": "WWWWWW"}, 
            "Partizan Belgrade": {"logo": "https://upload.wikimedia.org/wikipedia/en/7/74/FK_Partizan_logo.svg", "form": "WDWW_"}
        },
        "fixtures": [{"home": "Red Star Belgrade", "away": "Partizan Belgrade", "date": "2026-11-21", "time": "20:00 SAST"}]
    },
    "Croatia - HNL": {
        "teams": {
            "Dinamo Zagreb": {"logo": "https://upload.wikimedia.org/wikipedia/en/c/cc/GNK_Dinamo_Zagreb_logo.svg", "form": "WWWWWW"}, 
            "Hajduk Split": {"logo": "https://upload.wikimedia.org/wikipedia/en/e/ea/HNK_Hajduk_Split_logo.svg", "form": "WDWWL"}
        },
        "fixtures": [{"home": "Dinamo Zagreb", "away": "Hajduk Split", "date": "2026-11-21", "time": "18:30 SAST"}]
    }
}

# --- 4. HEAD-TO-HEAD (H2H) HISTORICAL RIVALRY DATABASE ---
H2H_DATABASE = {
    ("Arsenal", "Newcastle United"): (1.1, 0.9, "Arsenal holds a strong historical H2H advantage at home against Newcastle."),
    ("Liverpool", "Manchester City"): (1.0, 1.0, "Historically tightly contested heavyweight fixture with high-scoring encounters."),
    ("Chelsea", "Manchester United"): (1.0, 1.0, "Classic tactical matchup with frequent tight draws or single-goal margins."),
    ("Real Madrid", "Barcelona"): (1.05, 1.05, "El Clásico fixtures historically yield high offensive output from both giants."),
    ("Bayern Munich", "Borussia Dortmund"): (1.1, 0.95, "Der Klassiker heavily favors Bayern historically, especially at home.")
}

# --- 5. AUTOMATED HISTORICAL xG ENGINE WITH H2H FACTOR ---
def calculate_historical_xg(home_name, away_name, teams_dictionary):
    home_team_info = teams_dictionary.get(home_name, {"form": "DDWWDW"})
    away_team_info = teams_dictionary.get(away_name, {"form": "DDWWDW"})
    
    def form_multiplier(form_str):
        if not form_str or form_str == "N/A":
            return 1.0
        score = 0
        for char in form_str.upper():
            if char == 'W': score += 3
            elif char == 'D': score += 1
            elif char == 'L': score += 0
        max_possible = len(form_str) * 3
        if max_possible == 0:
            return 1.0
        ratio = score / max_possible
        return 0.7 + (ratio * 0.6)

    base_home_xg = 1.55
    base_away_xg = 1.15
    
    home_calculated_xg = base_home_xg * form_multiplier(home_team_info.get("form", ""))
    away_calculated_xg = base_away_xg * form_multiplier(away_team_info.get("form", ""))
    
    h2h_note = "Standard form-based projection (No specific H2H historical weight anomaly recorded)."
    if (home_name, away_name) in H2H_DATABASE:
        h2h_adj = H2H_DATABASE[(home_name, away_name)]
        home_calculated_xg *= h2h_adj[0]
        away_calculated_xg *= h2h_adj[1]
        h2h_note = h2h_adj[2]
        
    return round(max(0.5, min(4.5, home_calculated_xg)), 2), round(max(0.5, min(4.5, away_calculated_xg)), 2), h2h_note

# Helper function to convert form string into HTML chips
def render_form_chips(form_str):
    if not form_str or form_str == "N/A":
        return "N/A"
    chips_html = ""
    for char in form_str.upper():
        if char == 'W':
            chips_html += '<span class="form-chip-win">W</span>'
        elif char == 'D':
            chips_html += '<span class="form-chip-draw">D</span>'
        elif char == 'L':
            chips_html += '<span class="form-chip-loss">L</span>'
    return chips_html

# --- 6. MAIN DASHBOARD HEADER WITH ENLARGED GLASS ICON ---
st.markdown("""
    <div style="text-align: left;">
        <div class="icon-glass-badge">⚽</div>
    </div>
""", unsafe_allow_html=True)

st.title("Football Match Predictor")
st.markdown("Select a competition below to automatically load scheduled fixtures, view automated historical form-based Expected Goals (`xG`), and click **Run Match Prediction**.")

# --- 7. CLEAN SELECTION CONTROLS ---
col_c1, col_c2 = st.columns(2)
with col_c1:
    selected_league = st.selectbox("Select Competition / Tournament", list(LEAGUE_DATABASE.keys()))

league_data = LEAGUE_DATABASE[selected_league]
teams_info = league_data["teams"]
fixtures_list = league_data["fixtures"]

fixture_options = [
    f"{f['home']} vs {f['away']} ({f['date']} @ {f['time']})" 
    for f in fixtures_list
]

with col_c2:
    selected_fixture_str = st.selectbox("Select Match Fixture", fixture_options)

selected_index = fixture_options.index(selected_fixture_str)
active_match = fixtures_list[selected_index]

home_team = active_match["home"]
away_team = active_match["away"]
match_date = active_match["date"]
match_time = active_match["time"]

home_data = teams_info.get(home_team, {"form": "N/A"})
away_data = teams_info.get(away_team, {"form": "N/A"})

# Automatically compute xG with H2H Factor integration
home_xg, away_xg, h2h_insight = calculate_historical_xg(home_team, away_team, teams_info)

# --- 8. STATIC MATCH DATE & TIME BADGE (SAST) ---
st.markdown(f"""
<div class="match-info-badge" style="margin-top: 10px;">
    📅 Kick-off Scheduled: <b>{match_date} at {match_time}</b> | 🏆 Competition: <b>{selected_league}</b>
</div>
""", unsafe_allow_html=True)

# --- 9. UNIFIED SPECTACULAR TEAM GLASS PODS WITH FORM CHIPS ---
col_h, col_vs, col_a = st.columns([5, 1.2, 5])

with col_h:
    home_form_chips = render_form_chips(home_data.get('form', 'N/A'))
    st.markdown(f"""
        <div class="glass-card">
            <img src="{home_data.get('logo', '')}" width="75" style="margin-bottom: 15px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5);">
            <div style="font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px;">{home_team}</div>
            <div style="font-size: 13px; color: #93c5fd; margin-top: 8px; font-weight: 600;">Recent Form: <div style="margin-top: 6px;">{home_form_chips}</div></div>
            <div style="margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 20px;">
                <div class="metric-title">Expected Goals (xG)</div>
                <div style="font-size: 38px; font-weight: 900; color: #60a5fa; margin-top: 6px; text-shadow: 0 2px 12px rgba(96,165,250,0.3);">{home_xg}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_vs:
    st.markdown("<div style='text-align: center; padding-top: 130px; font-size: 26px; font-weight: 900; color: #3b82f6; text-shadow: 0 2px 15px rgba(59,130,246,0.5);'>VS</div>", unsafe_allow_html=True)

with col_a:
    away_form_chips = render_form_chips(away_data.get('form', 'N/A'))
    st.markdown(f"""
        <div class="glass-card">
            <img src="{away_data.get('logo', '')}" width="75" style="margin-bottom: 15px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5);">
            <div style="font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px;">{away_team}</div>
            <div style="font-size: 13px; color: #fca5a5; margin-top: 8px; font-weight: 600;">Recent Form: <div style="margin-top: 6px;">{away_form_chips}</div></div>
            <div style="margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 20px;">
                <div class="metric-title">Expected Goals (xG)</div>
                <div style="font-size: 38px; font-weight: 900; color: #ef4444; margin-top: 6px; text-shadow: 0 2px 12px rgba(239,68,68,0.3);">{away_xg}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 10. POISSON PREDICTION ENGINE ---
def calculate_match_probabilities(home_expected_goals, away_expected_goals, max_goals=6):
    home_probs = [poisson.pmf(i, home_expected_goals) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(j, away_expected_goals) for j in range(max_goals + 1)]
    
    score_matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diagonal(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    
    over_2_5 = 0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if (h + a) > 2.5:
                over_2_5 += score_matrix[h, a]
                
    under_2_5 = 1 - over_2_5
    
    best_score_idx = np.argmax(score_matrix)
    most_likely_h, most_likely_a = divmod(best_score_idx, max_goals + 1)
    
    max_prob = max(home_win, draw, away_win)
    confidence = round(max_prob * 100, 1)

    return {
        "home_win_prob": round(home_win * 100, 1),
        "draw_prob": round(draw * 100, 1),
        "away_win_prob": round(away_win * 100, 1),
        "over_2_5_prob": round(over_2_5 * 100, 1),
        "under_2_5_prob": round(under_2_5 * 100, 1),
        "most_likely_score": f"{most_likely_h} - {most_likely_a}",
        "confidence": confidence
    }

# --- 11. EXPLICIT PREDICTION BUTTON & RESULTS DASHBOARD ---
if st.button("Run Match Prediction", type="primary"):
    result = calculate_match_probabilities(home_xg, away_xg)
    
    st.markdown("### 📊 Match Prediction Results")
    st.info(f"📅 **Kick-off:** {match_date} at {match_time} | 🏆 **Fixture:** {home_team} vs {away_team}")
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: left; padding: 24px 32px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <div class="metric-title">Most Likely Exact Score</div>
                <div class="metric-value" style="color: #10b981;">{result['most_likely_score']}</div>
            </div>
            <div style="text-align: right;">
                <div class="metric-title">Model Confidence</div>
                <div class="metric-value" style="color: #3b82f6;">{result['confidence']}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Win Probabilities Breakdown")
    col_d1, col_d2, col_d3 = st.columns(3)
    
    def render_donut_chart(title, percentage, color):
        deg = (percentage / 100.0) * 360
        return f'<div class="glass-card" style="text-align: center; padding: 25px;"><div class="metric-title" style="margin-bottom: 15px; min-height: 35px; display: flex; align-items: center; justify-content: center;">{title}</div><div style="display: flex; justify-content: center; align-items: center;"><div style="width: 130px; height: 130px; border-radius: 50%; background: conic-gradient({color} 0deg {deg}deg, rgba(255, 255, 255, 0.05) {deg}deg 360deg); display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5);"><div style="width: 90px; height: 90px; background-color: #111522; border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center;"><span style="font-size: 18px; font-weight: 800; color: {color};">{percentage}%</span></div></div></div></div>'

    with col_d1:
        st.markdown(render_donut_chart(f"{home_team} Win", result['home_win_prob'], "#3b82f6"), unsafe_allow_html=True)
    with col_d2:
        st.markdown(render_donut_chart("Draw", result['draw_prob'], "#eab308"), unsafe_allow_html=True)
    with col_d3:
        st.markdown(render_donut_chart(f"{away_team} Win", result['away_win_prob'], "#ef4444"), unsafe_allow_html=True)
    
    st.markdown("### 🧠 Statistical Reasoning & Breakdown")
    st.markdown(f"""
    <div class="glass-card" style="text-align: left; padding: 25px 32px;">
        <ul style="color: #d1d5db; line-height: 1.8; margin: 0; padding-left: 20px;">
            <li><b>{home_team} Win ({result['home_win_prob']}%):</b> Driven by an automated Expected Goals rating (xG = {home_xg}) derived from recent form, calculating a higher likelihood of home goals exceeding away goals.</li>
            <li><b>Draw ({result['draw_prob']}%):</b> Represents the probability that both teams score an equal number of goals based on their simulated scoring outputs.</li>
            <li><b>{away_team} Win ({result['away_win_prob']}%):</b> Computed when visiting team goal projections surpass home team projections (xG = {away_xg} based on form).</li>
            <li><b>Goals Market ({result['over_2_5_prob']}% Over 2.5):</b> Calculated by combining all scoreline possibilities where total match goals exceed 2.5.</li>
            <li><b>Head-to-Head Factor:</b> <i>{h2h_insight}</i></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🥅 Goals Market (Over / Under 2.5)")
    st.progress(result['over_2_5_prob'] / 100.0)
    st.write(f"**Over 2.5 Goals:** {result['over_2_5_prob']}%  |  **Under 2.5 Goals:** {result['under_2_5_prob']}%")

    # --- 12. CSV MATCH REPORT EXPORT ---
    report_data = pd.DataFrame([{
        "Competition": selected_league,
        "Fixture": f"{home_team} vs {away_team}",
        "Kick-off": f"{match_date} {match_time}",
        f"{home_team} xG": home_xg,
        f"{away_team} xG": away_xg,
        "Most Likely Score": result['most_likely_score'],
        "Model Confidence (%)": result['confidence'],
        f"{home_team} Win (%)": result['home_win_prob'],
        "Draw (%)": result['draw_prob'],
        f"{away_team} Win (%)": result['away_win_prob'],
        "Over 2.5 Goals (%)": result['over_2_5_prob'],
        "Under 2.5 Goals (%)": result['under_2_5_prob'],
        "H2H Factor Note": h2h_insight
    }])

    csv_data = report_data.to_csv(index=False).encode('utf-8')
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Full Match Prediction Report (CSV)",
        data=csv_data,
        file_name=f"{home_team}_vs_{away_team}_prediction_report.csv",
        mime="text/csv",
        type="secondary"
    )