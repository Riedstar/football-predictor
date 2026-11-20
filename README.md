# ⚽ Global Football Match Predictor & Match Analytics Dashboard

A full-stack data science web application built with **Python**, **Streamlit**, and **SciPy** to predict football match outcomes, exact scorelines, and goal markets across top-tier European leagues and international tournaments using the **Poisson distribution** statistical model.

---

## 🚀 Live Demo & Features
* **Multi-League & International Coverage:** Supports 20+ top-flight European domestic leagues (Premier League, La Liga, Bundesliga, Serie A, Eredivisie, etc.) as well as major international tournaments (FIFA World Cup, Euros, AFCON).
* **Automated Match Day Scheduling:** Chronologically auto-populates fixtures, dates, and kick-off times for selected competitions.
* **Recent Form Guides:** Displays team momentum indicators (e.g., `WWWDWW`) right beside club crests and badges.
* **Poisson Probability Engine:** Computes discrete probability distributions for team goal expectancy (`xG`) to evaluate 1X2 win probabilities, Over/Under 2.5 goal markets, and most likely exact scorelines.
* **Transparent Statistical Reasoning:** Automatically breaks down the math behind each prediction so users understand the underlying probability matrix.

---

## 🛠️ Tech Stack & Libraries
* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Statistical Modeling & Math:** [SciPy (Poisson PMF)](https://scipy.org/) & [NumPy](https://numpy.org/) (Matrix outer products & probability summation)
* **API Integration:** Python `requests` for live fixture fetching (API-Football)

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Riedstar/football-predictor.git](https://github.com/Riedstar/football-predictor.git)
   cd football-predictor