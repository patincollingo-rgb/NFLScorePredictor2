# NFL Score Predictor

Quick setup:

1. Open this repository in GitHub Codespaces or any Python 3.11 environment.
2. The devcontainer will install dependencies from requirements.txt.
3. Fetch historical game data:
   - python data_fetch.py
4. Train the model:
   - python train_model.py
5. Run the web app:
   - python app.py
6. In Codespaces, open the forwarded port 5000 or preview the web app.

Notes:
- This project uses a simple rolling-average feature set. Improve by adding ELO ratings, injuries, weather, play-by-play features, advanced stats, or external APIs.
- Respect Pro-Football-Reference terms of service and rate-limit your requests when fetching many seasons.
