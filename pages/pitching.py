# pages/pitching.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from glob import glob

st.set_page_config(page_title="Pitching", layout="wide")
st.title("Pitching")

# -----------------------------
# Load all merged CSVs
# -----------------------------
data_folder = "data"
merged_files = glob(os.path.join(data_folder, "merged_*.csv"))

if not merged_files:
    st.warning("No merged files found in 'data/'")
    st.stop()

dfs = []
for f in merged_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        st.error(f"Failed to read {f}: {e}")

df = pd.concat(dfs, ignore_index=True)

# Ensure Trackman columns exist
if "batter_name_tm" not in df.columns or "pitcher_name_tm" not in df.columns:
    st.error("Merged files do not contain Trackman batter/pitcher columns")
    st.stop()

# -----------------------------
# Filters (unique values only)
# -----------------------------
year_options = ["Select all"] + sorted(df["game_id_tm"].dropna().astype(str).str[:4].unique())
year = st.selectbox("Year", year_options)

# Filter games based on year
if year == "Select all":
    games_options = sorted(df["game_id_tm"].dropna().unique())
else:
    games_options = sorted(df.loc[df["game_id_tm"].str.startswith(year), "game_id_tm"].dropna().unique())
games_options = ["Select all"] + list(games_options)
game = st.selectbox("Game", games_options)

# Filter opponent based on game
if game == "Select all":
    opponent_options = sorted(df["opponent_px"].dropna().unique()) if "opponent_px" in df.columns else []
else:
    opponent_options = sorted(df.loc[df["game_id_tm"] == game, "opponent_px"].dropna().unique()) \
        if "opponent_px" in df.columns else []
opponent_options = ["Select all"] + list(opponent_options)
opponent = st.selectbox("Opponent", opponent_options)

# Filter batter based on game
if game == "Select all":
    batter_options = sorted(df["batter_name_tm"].dropna().unique())
else:
    batter_options = sorted(df.loc[df["game_id_tm"] == game, "batter_name_tm"].dropna().unique())
batter_options = ["Select all"] + list(batter_options)
batter = st.selectbox("Batter", batter_options)

# Filter pitcher based on game
if game == "Select all":
    pitcher_options = sorted(df["pitcher_name_tm"].dropna().unique())
else:
    pitcher_options = sorted(df.loc[df["game_id_tm"] == game, "pitcher_name_tm"].dropna().unique())
pitcher_options = ["Select all"] + list(pitcher_options)
pitcher = st.selectbox("Pitcher", pitcher_options)

batter_hand_options = ["Select all"] + (sorted(df["BatterSide_px"].dropna().unique()) if "BatterSide_px" in df.columns else [])
batter_hand = st.selectbox("Batter Hand", batter_hand_options)

pitcher_hand_options = ["Select all"] + (sorted(df["PitcherSide_px"].dropna().unique()) if "PitcherSide_px" in df.columns else [])
pitcher_hand = st.selectbox("Pitcher Hand", pitcher_hand_options)

# -----------------------------
# Filter dataframe
# -----------------------------
df_plot = df.copy()

if game != "Select all":
    df_plot = df_plot[df_plot["game_id_tm"] == game]

if batter != "Select all":
    df_plot = df_plot[df_plot["batter_name_tm"] == batter]

if pitcher != "Select all":
    df_plot = df_plot[df_plot["pitcher_name_tm"] == pitcher]

if opponent != "Select all" and "opponent_px" in df_plot.columns:
    df_plot = df_plot[df_plot["opponent_px"] == opponent]

if batter_hand != "Select all" and "BatterSide_px" in df_plot.columns:
    df_plot = df_plot[df_plot["BatterSide_px"] == batter_hand]

if pitcher_hand != "Select all" and "PitcherSide_px" in df_plot.columns:
    df_plot = df_plot[df_plot["PitcherSide_px"] == pitcher_hand]

if df_plot.empty:
    st.info("No data for this selection")
    st.stop()

# -----------------------------
# Strike zone chart
# -----------------------------
# Use Trackman PlateLocHeight / PlateLocSide columns
if "PlateLocHeight_tm" not in df_plot.columns or "PlateLocSide_tm" not in df_plot.columns:
    st.warning("Trackman PlateLocHeight_tm / PlateLocSide_tm columns not found")
    st.stop()

fig = go.Figure()

# Draw strike zone rectangle (from top of knees ~1.5 ft to mid-chest ~3.5 ft, width 17 inches)
# Trackman uses inches, so approximate: top=42, bottom=24, left=-8.5, right=8.5
fig.add_shape(
    type="rect",
    x0=-8.5, x1=8.5,
    y0=24, y1=42,
    line=dict(color="black", width=2),
    fillcolor="rgba(0,0,0,0)"
)

# Add pitches as circles (2.9 inch diameter)
for _, row in df_plot.iterrows():
    fig.add_shape(
        type="circle",
        x0=(row["PlateLocSide_tm"]*12) - 1.45,
        x1=(row["PlateLocSide_tm"]*12) + 1.45,
        y0=(row["PlateLocHeight_tm"]*12) - 1.45,
        y1=(row["PlateLocHeight_tm"]*12) + 1.45,
        line=dict(color="blue"),
        fillcolor="rgba(0,0,255,0.5)"
    )

fig.update_xaxes(title="PlateLocSide (inches)", range=[-20, 20], scaleanchor="y", scaleratio=1)
fig.update_yaxes(title="PlateLocHeight (inches)", range=[0, 60], scaleanchor="x", scaleratio=1)
fig.update_layout(height=600, width=50, title=f"Strike Zone: {batter} vs {pitcher}")

st.plotly_chart(fig)
