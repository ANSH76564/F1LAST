from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report,
)

st.set_page_config(page_title="F1 Race Strategy Simulator", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator")
st.caption(
    "Uses the pre-merged F1 dataset to estimate finishing position and podium probability. "
    "It does not optimize tyre or pit-stop strategy."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "f1_merged_dataset.csv"


@st.cache_data(show_spinner="Loading and preparing the merged F1 dataset...")
def load_merged_data(csv_path: str):
    """Load one merged CSV and prepare historical features for the ML models."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path.name}. Place f1_merged_dataset.csv in the same folder as app.py."
        )

    df = pd.read_csv(path, na_values=[r"\N"])

    aliases = {
        "positionOrder": ["positionOrder", "position_order", "finish_position", "position"],
        "year": ["year", "season"],
        "round": ["round", "race_round"],
        "grid": ["grid", "grid_position", "starting_grid"],
        "driver_name": ["driver_name", "driverName", "full_name", "driver"],
        "constructor_name": ["constructor_name", "constructorName", "team_name", "constructor", "team"],
        "driverId": ["driverId", "driver_id"],
        "constructorId": ["constructorId", "constructor_id", "teamId", "team_id"],
        "raceId": ["raceId", "race_id"],
        "circuitId": ["circuitId", "circuit_id"],
        "points": ["points", "driver_points"],
        "date": ["date", "race_date"],
        "race_name": ["race_name", "raceName", "name", "race"],
    }
    for standard_name, candidates in aliases.items():
        if standard_name not in df.columns:
            found = next((col for col in candidates if col in df.columns), None)
            if found is not None:
                df = df.rename(columns={found: standard_name})

    if "driver_name" not in df.columns and {"forename", "surname"}.issubset(df.columns):
        df["driver_name"] = (
            df["forename"].fillna("").astype(str) + " " + df["surname"].fillna("").astype(str)
        ).str.strip()

    required = ["year", "round", "grid", "positionOrder", "driver_name", "constructor_name"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The merged CSV is missing required columns: " + ", ".join(missing)
            + ". Check that you are using f1_merged_dataset.csv created after merging races, results, drivers, and constructors."
        )

    if "driverId" not in df.columns:
        df["driverId"] = pd.factorize(df["driver_name"].astype(str))[0] + 1
    if "constructorId" not in df.columns:
        df["constructorId"] = pd.factorize(df["constructor_name"].astype(str))[0] + 1
    if "raceId" not in df.columns:
        df["raceId"] = pd.factorize(
            df["year"].astype(str) + "_" + df["round"].astype(str)
        )[0] + 1
    if "circuitId" not in df.columns:
        df["circuitId"] = 0
    if "points" not in df.columns:
        df["points"] = 0
    if "date" not in df.columns:
        df["date"] = pd.NaT

    numeric_columns = [
        "year", "round", "grid", "positionOrder", "points", "raceId",
        "driverId", "constructorId", "circuitId",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["year", "round", "raceId", "positionOrder"]).reset_index(drop=True)

    model_df = df.dropna(subset=["year", "round", "grid", "positionOrder"]).copy()
    model_df = model_df[model_df["grid"] > 0].copy()
    model_df = model_df.sort_values(["year", "round", "raceId", "positionOrder"]).reset_index(drop=True)

    model_df["driver_previous_finish"] = model_df.groupby("driverId")["positionOrder"].transform(lambda s: s.shift(1))
    model_df["driver_avg_finish_before"] = model_df.groupby("driverId")["positionOrder"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["driver_avg_points_before"] = model_df.groupby("driverId")["points"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["team_previous_finish"] = model_df.groupby("constructorId")["positionOrder"].transform(lambda s: s.shift(1))
    model_df["team_avg_finish_before"] = model_df.groupby("constructorId")["positionOrder"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["team_avg_points_before"] = model_df.groupby("constructorId")["points"].transform(lambda s: s.shift(1).expanding().mean())

    features = [
        "year", "round", "circuitId", "grid", "driverId", "constructorId",
        "driver_previous_finish", "driver_avg_finish_before", "driver_avg_points_before",
        "team_previous_finish", "team_avg_finish_before", "team_avg_points_before",
    ]
    races = model_df[["year", "round", "circuitId"]].drop_duplicates().copy()
    return df, races, model_df, features


try:
    df, races, model_df, feature_columns = load_merged_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the merged F1 dataset. Details: {exc}")
    st.markdown(
        "**Required project structure**\n\n"
        "```text\nF1_Race_Strategy_Simulator/\n"
        "├── app.py\n"
        "└── f1_merged_dataset.csv\n```"
    )
    st.stop()

st.success(f"Loaded {len(df):,} rows from f1_merged_dataset.csv.")
years = sorted(model_df["year"].dropna().astype(int).unique())
if len(years) < 2:
    st.error("At least two different seasons are needed to train and test the model.")
    st.stop()

split_index = min(max(1, int(len(years) * 0.8)), len(years) - 1)
test_start_year = years[split_index]
train_data = model_df[model_df["year"] < test_start_year].copy()
test_data = model_df[model_df["year"] >= test_start_year].copy()
if train_data.empty or test_data.empty:
    st.error("Not enough data after the chronological train/test split.")
    st.stop()


@st.cache_resource
def train_models(train_x, train_y, train_podium_y):
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    ])
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    reg.fit(train_x, train_y)
    clf.fit(train_x, train_podium_y)
    return reg, clf


with st.spinner("Training the models on the merged historical dataset..."):
    train_podium = train_data["positionOrder"].between(1, 3).astype(int)
    regressor, podium_model = train_models(train_data[feature_columns], train_data["positionOrder"], train_podium)

test_pred = regressor.predict(test_data[feature_columns])
test_podium_true = test_data["positionOrder"].between(1, 3).astype(int)
test_podium_pred = podium_model.predict(test_data[feature_columns])
mae = mean_absolute_error(test_data["positionOrder"], test_pred)
rmse = np.sqrt(mean_squared_error(test_data["positionOrder"], test_pred))
accuracy = accuracy_score(test_podium_true, test_podium_pred)

st.subheader("Model evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test seasons", f"{int(test_data['year'].min())}–{int(test_data['year'].max())}")
m2.metric("Finishing-position MAE", f"{mae:.2f} places")
m3.metric("Finishing-position RMSE", f"{rmse:.2f} places")
m4.metric("Podium classification accuracy", f"{accuracy:.1%}")
with st.expander("View classification report"):
    report = classification_report(test_podium_true, test_podium_pred, labels=[0, 1], target_names=["Not podium", "Podium"], zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.caption("Accuracy can be misleading when podium results are less common. Check podium precision, recall, and F1 too.")

st.divider()
st.subheader("Predict a hypothetical race")
drivers = sorted(df["driver_name"].dropna().astype(str).loc[lambda s: s.str.strip().ne("")].unique().tolist())
constructors = sorted(df["constructor_name"].dropna().astype(str).unique().tolist())
if not drivers or not constructors:
    st.error("No driver or constructor names were found in the merged dataset.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    driver_name = st.selectbox("Driver", drivers)
with c2:
    constructor_name = st.selectbox("Constructor / team", constructors)
with c3:
    prediction_year = st.selectbox("Season", list(range(int(max(years)), int(min(years)) - 1, -1)))

available_rounds = sorted(races.loc[races["year"] == prediction_year, "round"].dropna().astype(int).unique().tolist())
if not available_rounds:
    available_rounds = sorted(model_df["round"].dropna().astype(int).unique().tolist())
round_number = st.selectbox("Round", available_rounds, index=0)
grid_position = st.slider("Hypothetical starting grid position", min_value=1, max_value=20, value=5)


def simulate(driver, team, year, rnd, grid):
    earlier = model_df[(model_df["year"] < year) | ((model_df["year"] == year) & (model_df["round"] < rnd))]
    dh = earlier[earlier["driver_name"].astype(str).str.casefold() == driver.casefold()]
    th = earlier[earlier["constructor_name"].astype(str).str.casefold() == team.casefold()]
    race_match = races[(races["year"] == year) & (races["round"] == rnd)]
    circuit_id = race_match["circuitId"].iloc[0] if len(race_match) else np.nan
    driver_rows = df[df["driver_name"].astype(str).str.casefold() == driver.casefold()]
    team_rows = df[df["constructor_name"].astype(str).str.casefold() == team.casefold()]
    row = {
        "year": year,
        "round": rnd,
        "circuitId": circuit_id,
        "grid": grid,
        "driverId": driver_rows["driverId"].iloc[0] if len(driver_rows) else np.nan,
        "constructorId": team_rows["constructorId"].iloc[0] if len(team_rows) else np.nan,
        "driver_previous_finish": dh["positionOrder"].iloc[-1] if len(dh) else np.nan,
        "driver_avg_finish_before": dh["positionOrder"].mean() if len(dh) else np.nan,
        "driver_avg_points_before": dh["points"].mean() if len(dh) else np.nan,
        "team_previous_finish": th["positionOrder"].iloc[-1] if len(th) else np.nan,
        "team_avg_finish_before": th["positionOrder"].mean() if len(th) else np.nan,
        "team_avg_points_before": th["points"].mean() if len(th) else np.nan,
    }
    X = pd.DataFrame([row])[feature_columns]
    predicted_position = float(regressor.predict(X)[0])
    probabilities = podium_model.predict_proba(X)[0]
    podium_class_index = list(podium_model.classes_).index(1) if 1 in podium_model.classes_ else 0
    podium_probability = float(probabilities[podium_class_index])
    return predicted_position, podium_probability


if st.button("Run prediction", type="primary"):
    predicted_position, podium_probability = simulate(driver_name, constructor_name, int(prediction_year), int(round_number), int(grid_position))
    a, b = st.columns(2)
    a.metric("Estimated finishing position", f"{predicted_position:.1f}")
    b.metric("Estimated podium probability", f"{podium_probability:.1%}")
    st.caption("These are estimates based on historical results, not guarantees. This version does not model tyre wear, pit stops, weather, or safety cars.")

st.subheader("Compare starting-grid scenarios")
if st.button("Compare grid positions"):
    rows = []
    for grid in [1, 3, 5, 10, 15, 20]:
        pos, prob = simulate(driver_name, constructor_name, int(prediction_year), int(round_number), grid)
        rows.append({"Grid position": grid, "Estimated finishing position": round(pos, 1), "Podium probability": f"{prob:.1%}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
