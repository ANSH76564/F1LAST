# F1 Race Strategy Simulator

## Files required
- `app.py`
- `f1_merged_dataset.csv` (the merged dataset created earlier in this notebook)
- `requirements.txt`

## Run locally
Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

The app trains its prediction models from the merged historical data when it starts. It estimates finishing position and podium probability; it does not model tyre wear, pit stops, weather, or safety cars.
