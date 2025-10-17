"""
Load CSVs created by data_fetch.py, build simple features, train two regressors:
 - home_team_points
 - visitor_team_points

Feature design (simple, extendable):
 - home_team average points scored last N games
 - away_team average points scored last N games
 - home_team average points allowed last N games
 - away_team average points allowed last N games
 - home_field indicator
 - season and week as numeric features

This is intentionally simple so you can iterate and improve.
"""
import pandas as pd
import numpy as np
from glob import glob
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

DATA_GLOB = "data/games_*.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_all():
    files = sorted(glob(DATA_GLOB))
    df_list = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['date'], infer_datetime_format=True)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # Drop rows missing scores (future or unplayed games)
    df = df.dropna(subset=['home_pts', 'visitor_pts'])
    # Standardize team names to strings
    df['home_team'] = df['home_team'].astype(str)
    df['visitor_team'] = df['visitor_team'].astype(str)
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    return df

def rolling_team_stats(df, window=5):
    # Build per-team rolling averages of points scored and allowed
    df_expanded = []
    teams = pd.unique(df[['home_team','visitor_team']].values.ravel('K'))
    # initialize empty history
    history = {t: {'dates': [], 'scored': [], 'allowed': []} for t in teams}
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['visitor_team']
        # compute last N averages for both teams
        def last_stats(team):
            rec = history.get(team, {'scored': [], 'allowed': []})
            s = np.array(rec['scored'][-window:]) if len(rec['scored'])>0 else np.array([])
            a = np.array(rec['allowed'][-window:]) if len(rec['allowed'])>0 else np.array([])
            return {
                'avg_scored': float(s.mean()) if s.size>0 else np.nan,
                'avg_allowed': float(a.mean()) if a.size>0 else np.nan,
                'games_played': len(s)
            }
        home_stats = last_stats(home)
        away_stats = last_stats(away)
        rec = {
            'date': row.get('date'),
            'season': row.get('season'),
            'week': row.get('week'),
            'home_team': home,
            'visitor_team': away,
            'home_pts': row['home_pts'],
            'visitor_pts': row['visitor_pts'],
            'home_avg_scored_lastN': home_stats['avg_scored'],
            'home_avg_allowed_lastN': home_stats['avg_allowed'],
            'home_games_lastN': home_stats['games_played'],
            'away_avg_scored_lastN': away_stats['avg_scored'],
            'away_avg_allowed_lastN': away_stats['avg_allowed'],
            'away_games_lastN': away_stats['games_played'],
            'is_home_game': 1
        }
        df_expanded.append(rec)
        # update history after the game (use actual scores)
        history[home]['scored'].append(row['home_pts'])
        history[home]['allowed'].append(row['visitor_pts'])
        history[away]['scored'].append(row['visitor_pts'])
        history[away]['allowed'].append(row['home_pts'])
    feat_df = pd.DataFrame(df_expanded)
    return feat_df

def prepare_training(feat_df):
    # Drop rows where either team has no history (NaN averages), keep them for prediction later as NaN-handling
    train_df = feat_df.dropna(subset=[
        'home_avg_scored_lastN', 'home_avg_allowed_lastN',
        'away_avg_scored_lastN', 'away_avg_allowed_lastN'
    ]).copy()
    # Features
    X = train_df[[
        'home_avg_scored_lastN', 'home_avg_allowed_lastN', 'home_games_lastN',
        'away_avg_scored_lastN', 'away_avg_allowed_lastN', 'away_games_lastN'
    ]].fillna(0)
    # Add a simple home-field feature: home team tends to score more
    X['home_field'] = 1
    y_home = train_df['home_pts']
    y_away = train_df['visitor_pts']
    return X, y_home, y_away, train_df

def train_and_save(X, y_home, y_away):
    X_train, X_test, hy_train, hy_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, ay_train, ay_test = train_test_split(X, y_away, test_size=0.2, random_state=42)
    model_home = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_away = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_home.fit(X_train, hy_train)
    model_away.fit(X_train, ay_train)
    # Metrics
    pred_h = model_home.predict(X_test)
    pred_a = model_away.predict(X_test)
    mae_h = mean_absolute_error(hy_test, pred_h)
    mae_a = mean_absolute_error(ay_test, pred_a)
    print(f"Home MAE: {mae_h:.3f}, Away MAE: {mae_a:.3f}")
    joblib.dump(model_home, os.path.join(MODEL_DIR, "model_home.pkl"))
    joblib.dump(model_away, os.path.join(MODEL_DIR, "model_away.pkl"))
    print(f"Saved models to {MODEL_DIR}")
    return model_home, model_away

if __name__ == "__main__":
    df = load_all()
    feat_df = rolling_team_stats(df, window=5)
    X, y_home, y_away, train_df = prepare_training(feat_df)
    train_and_save(X, y_home, y_away)
