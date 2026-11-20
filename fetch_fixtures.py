import json
import requests

LEAGUES = {
    "England - Premier League": "eng.1",
    "England - EFL Championship": "eng.2",
    "England - FA Cup": "eng.fa",
    "Spain - La Liga": "esp.1",
    "Germany - Bundesliga": "ger.1",
    "Italy - Serie A": "ita.1",
    "France - Ligue 1": "fra.1",
    "UEFA - Champions League": "uefa.champions",
    "UEFA - Europa League": "uefa.europa",
    "UEFA - Conference League": "uefa.europa.conf",
    "South Africa - Premiership (PSL)": "rsa.1",
    "Netherlands - Eredivisie": "ned.1",
    "Portugal - Primeira Liga": "por.1",
    "Scotland - Premiership": "sco.1",
    "Turkey - Süper Lig": "tur.1",
    "Brazil - Série A": "bra.1",
    "Argentina - Liga Profesional": "arg.1",
    "USA - MLS": "usa.1",
    "International - Friendly / Nations": "fifa.friendly",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_standings_table(slug: str) -> dict:
  """Fetches League Position and W-D-L record from ESPN's standings API."""
  table = {}
  for host in ("site.web.api.espn.com", "site.api.espn.com"):
    url = f"https://{host}/apis/v2/sports/soccer/{slug}/standings"
    try:
      resp = requests.get(url, headers=HEADERS, timeout=10)
      if resp.status_code != 200:
        continue

      data = resp.json()
      groups = data.get("children", [])
      if not groups and "standings" in data:
        groups = [data]

      for group in groups:
        entries = group.get("standings", {}).get("entries", [])
        for idx, entry in enumerate(entries, start=1):
          team_obj = entry.get("team", {})
          t_id = str(team_obj.get("id", ""))
          t_name = team_obj.get("displayName", "")

          stats_map = {}
          for s in entry.get("stats", []):
            key = s.get("name", "")
            val = s.get("displayValue") if s.get("displayValue") is not None else s.get("value")
            if key and val is not None:
              stats_map[key] = str(val)

          pos = stats_map.get("rank") or str(idx)
          wins = stats_map.get("wins", "0")
          ties = stats_map.get("ties", "0")
          losses = stats_map.get("losses", "0")
          gp = float(stats_map.get("gamesPlayed", 0) or 0)
          gf = float(stats_map.get("pointsFor", 0) or 0)
          ga = float(stats_map.get("pointsAgainst", 0) or 0)

          info = {
              "position": str(pos),
              "record": f"{wins}W {ties}D {losses}L",
              "avg_gf": round(gf / gp, 2) if gp > 0 else 1.40,
              "avg_ga": round(ga / gp, 2) if gp > 0 else 1.25,
          }
          if t_id:
            table[t_id] = info
          if t_name:
            table[t_name] = info

      if table:
        return table
    except Exception:
      continue
  return table


def extract_scoreboard_record(competitor: dict, fallback_form: str) -> str:
  """Extracts W-D-L directly from the scoreboard competitor's records array."""
  for r in competitor.get("records", []):
    summary = r.get("summary", "")  # e.g. "7-0-3"
    parts = summary.split("-")
    if len(parts) == 3:
      return f"{parts[0]}W {parts[1]}D {parts[2]}L"
  return f"{fallback_form.count('W')}W {fallback_form.count('D')}D {fallback_form.count('L')}L"


def fetch_all_leagues():
  database = {}

  for display_name, slug in LEAGUES.items():
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
    try:
      resp = requests.get(url, headers=HEADERS, timeout=10)
      if resp.status_code != 200:
        continue

      events = resp.json().get("events", [])
      if not events:
        continue

      standings = fetch_standings_table(slug)
      teams_dict = {}
      fixtures_list = []

      for event in events:
        comp = event["competitions"][0]
        competitors = comp["competitors"]

        home_comp = next((c for c in competitors if c["homeAway"] == "home"), None)
        away_comp = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home_comp or not away_comp:
          continue

        home_id = str(home_comp["team"]["id"])
        away_id = str(away_comp["team"]["id"])
        home_name = home_comp["team"]["displayName"]
        away_name = away_comp["team"]["displayName"]

        home_logo = home_comp["team"].get("logo", "")
        away_logo = away_comp["team"].get("logo", "")

        home_form = "".join(
            [c for c in (home_comp.get("form") or "WWDWL").upper() if c in ("W", "D", "L")]
        )[:5]
        away_form = "".join(
            [c for c in (away_comp.get("form") or "WDWLL").upper() if c in ("W", "D", "L")]
        )[:5]

        if home_name not in teams_dict:
          st_info = standings.get(home_id) or standings.get(home_name) or {}
          teams_dict[home_name] = {
              "logo": home_logo,
              "position": st_info.get("position", "-"),
              "record": st_info.get("record") or extract_scoreboard_record(home_comp, home_form),
              "last_5": home_form,
              "full_form": home_form,
              "avg_gf": st_info.get("avg_gf", 1.40),
              "avg_ga": st_info.get("avg_ga", 1.25),
          }

        if away_name not in teams_dict:
          st_info = standings.get(away_id) or standings.get(away_name) or {}
          teams_dict[away_name] = {
              "logo": away_logo,
              "position": st_info.get("position", "-"),
              "record": st_info.get("record") or extract_scoreboard_record(away_comp, away_form),
              "last_5": away_form,
              "full_form": away_form,
              "avg_gf": st_info.get("avg_gf", 1.40),
              "avg_ga": st_info.get("avg_ga", 1.25),
          }

        raw_date = event["date"]
        dt_part, tm_part = raw_date.split("T")
        tm_clean = tm_part.replace("Z", "")[:5]

        fixtures_list.append({
            "home": home_name,
            "away": away_name,
            "date": dt_part,
            "time": tm_clean,
        })

      if fixtures_list:
        database[display_name] = {
            "teams": teams_dict,
            "fixtures": fixtures_list,
        }
        sample = next(iter(teams_dict))
        s = teams_dict[sample]
        print(
            f"✅ {display_name} | Sample ({sample}): Pos {s['position']} | {s['record']} | Last 5: {s['last_5']}"
        )

    except Exception as e:
      print(f"⚠️ Error fetching {display_name}: {e}")

  if database:
    with open("fixtures.json", "w", encoding="utf-8") as f:
      json.dump(database, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 Saved {len(database)} competitions to fixtures.json!")


if __name__ == "__main__":
  fetch_all_leagues()