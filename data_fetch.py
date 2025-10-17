"""
Fetch game-level data from Pro-Football-Reference using pandas.read_html.
Saves CSV files for seasons you supply.
"""
import pandas as pd
from datetime import datetime
import time
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_season(season):
    url = f"https://www.pro-football-reference.com/years/{season}/games.htm"
    tables = pd.read_html(url)
    # The first table is the schedule/games table
    games = tables[0].copy()
    # Drop rows that are headers inside the table (they contain 'Week' in NaNs)
    games = games[games['Week'] != 'Week']
    # Rename columns to safe names
    games = games.rename(columns=lambda c: c.strip().replace(' ', '_').replace('%', 'pct'))
    # Keep relevant columns and convert types
    keep = ['Week', 'Date', 'Time', 'Day', 'Boxscore', 'Visitor', 'Pts', 'Home', 'Pts.1', 'Note']
    # Some seasons may have slightly different column names; adapt
    cols = [c for c in keep if c in games.columns]
    games = games[cols]
    # Standardize columns: visitor_team, visitor_pts, home_team, home_pts, date
    games = games.rename(columns={
        'Visitor': 'visitor_team',
        'Home': 'home_team',
        'Pts': 'visitor_pts',
        'Pts.1': 'home_pts',
        'Date': 'date',
        'Week': 'week'
    })
    # Convert points to numeric; some future weeks have blank points
    for col in ['visitor_pts', 'home_pts']:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors='coerce')
    # Parse date where possible; PFR uses formats like 'Sep 10'
    def parse_date(row):
        try:
            d = row['date']
            if isinstance(d, str):
                # PFR date strings often lack year; append season year
                return pd.to_datetime(f"{d} {season}")
            return pd.to_datetime(d)
        except Exception:
            return pd.NaT
    games['date'] = games.apply(parse_date, axis=1)
    games['season'] = season
    # Save CSV
    out = f"{OUTPUT_DIR}/games_{season}.csv"
    games.to_csv(out, index=False)
    print(f"Saved {out} ({len(games)} rows)")
    time.sleep(1)  # be polite
    return games

if __name__ == "__main__":
    seasons = list(range(2010, datetime.now().year + 1))  # adjust start year as desired
    for s in seasons:
        fetch_season(s)
