from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
MODEL_DIR = "models"
DATA_DIR = "data"

# Load models
model_home = joblib.load(os.path.join(MODEL_DIR, "model_home.pkl"))
model_away = joblib.load(os.path.join(MODEL_DIR, "model_away.pkl"))

# Load latest game CSVs to construct recent team stats for predictions
def load_history():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("games_") and f.endswith(".csv")])
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=['date'], infer_datetime_format=True))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['home_pts', 'visitor_pts'])
    return df

HISTORY_DF = load_history()

def compute_team_recent_stats(history_df, team, window=5):
    # collect last 'window' games for team
    # Each game produces either home or away row
    if history_df.empty:
        return {'avg_scored': np.nan, 'avg_allowed': np.nan, 'games': 0}
    # home games
    h = history_df[history_df['home_team'] == team][['date','home_pts','visitor_pts']].rename(
        columns={'home_pts':'scored','visitor_pts':'allowed'})
    # away games
    a = history_df[history_df['visitor_team'] == team][['date','visitor_pts','home_pts']].rename(
        columns={'visitor_pts':'scored','home_pts':'allowed'})
    all_games = pd.concat([h,a]).sort_values('date')
    last_games = all_games.tail(window)
    if last_games.empty:
        return {'avg_scored': np.nan, 'avg_allowed': np.nan, 'games': 0}
    return {'avg_scored': float(last_games['scored'].mean()), 'avg_allowed': float(last_games['allowed'].mean()), 'games': int(len(last_games))}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    teams = sorted(pd.unique(HISTORY_DF[['home_team','visitor_team']].values.ravel('K')))
    if request.method == 'POST':
        home = request.form.get('home_team')
        away = request.form.get('away_team')
        if home == away:
            error = "Home and away teams must be different."
        else:
            # compute features
            home_stats = compute_team_recent_stats(HISTORY_DF, home, window=5)
            away_stats = compute_team_recent_stats(HISTORY_DF, away, window=5)
            # If no recent history, fall back to neutral average values from dataset
            def safe(x, field, default=20.0):
                v = x.get(field)
                return default if (v is None or (isinstance(v,float) and np.isnan(v))) else v
            X = pd.DataFrame([{
                'home_avg_scored_lastN': safe(home_stats,'avg_scored'),
                'home_avg_allowed_lastN': safe(home_stats,'avg_allowed'),
                'home_games_lastN': home_stats.get('games',0),
                'away_avg_scored_lastN': safe(away_stats,'avg_scored'),
                'away_avg_allowed_lastN': safe(away_stats,'avg_allowed'),
                'away_games_lastN': away_stats.get('games',0),
                'home_field': 1
            }])
            pred_home = model_home.predict(X)[0]
            pred_away = model_away.predict(X)[0]
            # Round to nearest integer and ensure non-negative
            pred_home = max(0, int(round(pred_home)))
            pred_away = max(0, int(round(pred_away)))
            prediction = {'home': pred_home, 'away': pred_away, 'home_team': home, 'away_team': away}
    return render_template('index.html', teams=teams, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
