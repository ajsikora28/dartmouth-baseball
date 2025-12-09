# pages/pitching.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from glob import glob
from typing import List
from align import safe_col

st.set_page_config(page_title="Pitching", layout="wide")
st.title("Pitching")

# -----------------------------
# Load all merged CSVs
# -----------------------------
DATA_FOLDER = "data"
merged_files = glob(os.path.join(DATA_FOLDER, "merged_*.csv"))

if not merged_files:
    st.warning("No merged files found in 'data/' (expected pattern: merged_*.csv).")
    st.stop()

dfs = []
for f in merged_files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        st.error(f"Failed to read {f}: {e}")

if not dfs:
    st.error("No valid merged CSVs could be loaded.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Default pitching filter: only Dartmouth pitchers
# -----------------------------
pitcher_team_col = safe_col(df, ["PitcherTeam", "PitcherTeam_px", "pitcher_team_px", "Pitcher_Team_px"])
if pitcher_team_col in df.columns:
    df = df[df[pitcher_team_col].str.lower().isin(["dartmouth", "intrasquad", "intersquad","big green","dar"])]

# Display-formatted names
def title_case_col(tm_col, px_col):
    if tm_col in df.columns:
        return df[tm_col].str.title()
    elif px_col in df.columns:
        return df[px_col].str.title()
    else:
        return pd.Series(["Unknown"] * len(df))
    
# Pitch types
PITCH_TYPE_MAPPING = {
    1: "Four-Seam Fastball",
    2: "Curveball",
    3: "Slider",
    4: "Changeup",
    5: "Cutter",
    6: "Sinker",
    7: "Other"
}

df["batter_name_display"] = title_case_col("batter_name_tm", "batter_name_px")
df["pitcher_name_display"] = title_case_col("pitcher_name_tm", "pitcher_name_px")

# Lowercase-to-original column map
col_map = {c.lower(): c for c in df.columns}
def has_col(lower_name: str) -> bool:
    return lower_name in col_map
def col(lower_name: str):
    return col_map.get(lower_name)

# -----------------------------
# Multiselect helper with Select all button only
# -----------------------------
def ms_with_select_all(label: str, options: List[str], key_prefix: str) -> List[str]:
    options = list(options)
    sel_key = f"{key_prefix}_sel"
    ms_key = f"{key_prefix}_ms"

    if sel_key not in st.session_state:
        st.session_state[sel_key] = list(options)

    selection = st.multiselect(label, options, default=st.session_state[sel_key], key=ms_key)
    st.session_state[sel_key] = selection

    # Select all button
    if st.button("Select all", key=f"{key_prefix}_select_all"):
        st.session_state[sel_key] = list(options)

    return st.session_state[sel_key]

# -----------------------------
# Column definitions with fallback
# -----------------------------
def fallback_col(tm_col, px_col):
    if tm_col in df.columns:
        return tm_col
    elif px_col in df.columns:
        return px_col
    else:
        return None

game_col = fallback_col("game_id_tm", "game_id_px")
opp_col = fallback_col("Opponent_px", "Opponent_tm")
batter_col = "batter_name_display"
pitcher_col = "pitcher_name_display"
plh_col = fallback_col("PlateLocHeight_tm", "PlateLocHeight_px")
pls_col = fallback_col("PlateLocSide_tm", "PlateLocSide_px")
bhand_col = fallback_col("BatterSide_px", "BatterSide_tm")
phand_col = fallback_col("PitcherThrows_px", "PitcherThrows_tm")

balls_col = fallback_col("Balls_px", "Balls_tm")
strikes_col = fallback_col("Strikes_px", "Strikes_tm")
outs_col = fallback_col("Outs_px", "Outs_tm")
pitch_type_col = fallback_col("PitchType_tm", "PitchType_px")
vel_col = fallback_col("RelSpeed_tm", "RelSpeed_px")

# -----------------------------
# Prepare option lists
# -----------------------------
years = sorted(df[game_col].dropna().astype(str).str[:4].unique()) if game_col else []
games = sorted(df[game_col].dropna().unique()) if game_col else []
opponents = sorted(df[opp_col].dropna().unique()) if opp_col else []
batters_all = sorted(df[batter_col].dropna().unique())
pitchers_all = sorted(df[pitcher_col].dropna().unique())
batter_hands_all = sorted(df[bhand_col].dropna().unique()) if bhand_col else []
pitcher_hands_all = sorted(df[phand_col].dropna().unique()) if phand_col else []

counts_all = sorted([f"{b}-{s}" for b in sorted(df[balls_col].dropna().unique()) for s in sorted(df[strikes_col].dropna().unique())]) if balls_col and strikes_col else []
outs_all = sorted(df[outs_col].dropna().astype(int).astype(str).unique()) if outs_col else []
if pitch_type_col in df.columns:
    pitch_types_all_raw = df[pitch_type_col].dropna()
    def map_pitch_type_sidebar(x):
        try:
            return PITCH_TYPE_MAPPING.get(int(float(x)), str(x))
        except (ValueError, TypeError):
            return str(x)
    pitch_types_all = sorted(pitch_types_all_raw.apply(map_pitch_type_sidebar).unique())
else:
    pitch_types_all = []


vel_min = float(df[vel_col].min()) if vel_col else 0
vel_max = float(df[vel_col].max()) if vel_col else 100

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters (multiselects)")

selected_years = ms_with_select_all("Year", years, "year") if years else []
selected_games = ms_with_select_all("Game", games, "game") if games else []
selected_opponents = ms_with_select_all("Opponent", opponents, "opponent") if opponents else []
selected_batters = ms_with_select_all("Batter", batters_all, "batter") if batters_all else []
selected_pitchers = ms_with_select_all("Pitcher", pitchers_all, "pitcher") if pitchers_all else []
selected_batter_hands = ms_with_select_all("Batter Hand", batter_hands_all, "batter_hand") if batter_hands_all else []
selected_pitcher_hands = ms_with_select_all("Pitcher Hand", pitcher_hands_all, "pitcher_hand") if pitcher_hands_all else []
selected_outs = ms_with_select_all("Outs", outs_all, "outs") if outs_all else []
selected_pitch_types = ms_with_select_all("Pitch Type", pitch_types_all, "pitch_type") if pitch_types_all else []
selected_vel = st.sidebar.slider("Velocity (mph)", float(vel_min), float(vel_max), (float(vel_min), float(vel_max)))

# Counts (Balls-Strikes) — need to build count_str on df first
df_temp = df.copy()
balls_col_actual = fallback_col("Balls_tm", "Balls_px")
strikes_col_actual = fallback_col("Strikes_tm", "Strikes_px")

if balls_col_actual and strikes_col_actual and balls_col_actual in df_temp.columns and strikes_col_actual in df_temp.columns:
    df_temp["count_str"] = df_temp.apply(
        lambda r: f"{int(r[balls_col_actual])}-{int(r[strikes_col_actual])}"
        if pd.notna(r[balls_col_actual]) and pd.notna(r[strikes_col_actual])
        else "NA",
        axis=1
    )
    counts_all = sorted(df_temp["count_str"].dropna().unique())
    selected_counts = ms_with_select_all("Count", counts_all, "count") if counts_all else []
else:
    counts_all = []
    selected_counts = []

# -----------------------------
# Apply filters
# -----------------------------
df_plot = df.copy()

# Create a string column for counts like "1-1" instead of "1.0-1.0"
if balls_col_actual and strikes_col_actual:
    df_plot["count_str"] = df_plot.apply(
        lambda r: f"{int(r[balls_col_actual])}-{int(r[strikes_col_actual])}"
        if pd.notna(r[balls_col_actual]) and pd.notna(r[strikes_col_actual])
        else "NA",
        axis=1
    )
else:
    df_plot["count_str"] = "NA"

# Map pitch types to display names (to match selected_pitch_types)
if pitch_type_col in df_plot.columns:
    def map_pitch_type(x):
        try:
            # Convert numeric strings or floats to int
            return PITCH_TYPE_MAPPING.get(int(float(x)), str(x))
        except (ValueError, TypeError):
            return str(x)
    df_plot["pitch_type_display"] = df_plot[pitch_type_col].apply(map_pitch_type)
else:
    df_plot["pitch_type_display"] = "NA"

# Convert outs to string for filtering (to match selected_outs which are strings)
if outs_col:
    df_plot["outs_display"] = df_plot[outs_col].astype(int).astype(str)
else:
    df_plot["outs_display"] = "NA"


if game_col and selected_games:
    df_plot = df_plot[df_plot[game_col].isin(selected_games)]
if opp_col and selected_opponents:
    df_plot = df_plot[df_plot[opp_col].isin(selected_opponents)]
if selected_batters:
    df_plot = df_plot[df_plot[batter_col].isin(selected_batters)]
if selected_pitchers:
    df_plot = df_plot[df_plot[pitcher_col].isin(selected_pitchers)]
if bhand_col and selected_batter_hands:
    df_plot = df_plot[df_plot[bhand_col].isin(selected_batter_hands)]
if phand_col and selected_pitcher_hands:
    df_plot = df_plot[df_plot[phand_col].isin(selected_pitcher_hands)]
if selected_counts:
    df_plot = df_plot[df_plot["count_str"].isin(selected_counts)]
if selected_outs and outs_col:
    df_plot = df_plot[df_plot["outs_display"].isin(selected_outs)]
if selected_pitch_types and pitch_type_col:
    df_plot = df_plot[df_plot["pitch_type_display"].isin(selected_pitch_types)]
if vel_col:
    vel_mask = df_plot[vel_col].between(selected_vel[0], selected_vel[1])
    if selected_vel != (vel_min, vel_max):
        df_plot = df_plot[vel_mask]
    else:
        df_plot = df_plot[vel_mask | df_plot[vel_col].isna()]

if df_plot.empty:
    st.info("No data for this selection")
    st.stop()

# -----------------------------
# Convert plate locations to inches
# -----------------------------
if plh_col and pls_col:
    df_plot["platelocheight_in"] = df_plot[plh_col].astype(float) * 12.0
    df_plot["platelocside_in"] = df_plot[pls_col].astype(float) * 12.0
else:
    st.warning("Plate location columns not found.")
    df_plot["platelocheight_in"] = np.nan
    df_plot["platelocside_in"] = np.nan

# -----------------------------
# Basic stats
# -----------------------------
total_pitches = len(df_plot)
st.sidebar.metric("Pitches", total_pitches)

# -----------------------------
# Main chart area
# -----------------------------
col_left, col_right = st.columns([1, 2])
with col_right:
    fig = go.Figure()

    # strike zone rectangle (white outline)
    zone_left, zone_right = -17/2, 17/2
    zone_bottom, zone_top = 18, 42
    fig.add_shape(
        type="rect",
        x0=zone_left, x1=zone_right,
        y0=zone_bottom, y1=zone_top,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    # draw pitch circles (2.9" diameter)
    radius = 2.9 / 2.0
    for _, r in df_plot.iterrows():
        x, y = r["platelocside_in"], r["platelocheight_in"]
        if pd.isna(x) or pd.isna(y):
            continue
        fig.add_shape(
            type="circle",
            x0=x - radius, x1=x + radius,
            y0=y - radius, y1=y + radius,
            line=dict(color="blue", width=1),
            fillcolor="rgba(0,0,255,0.45)"
        )

    # axis ranges
    x_min_data, x_max_data = df_plot["platelocside_in"].min() - radius, df_plot["platelocside_in"].max() + radius
    y_min_data, y_max_data = df_plot["platelocheight_in"].min() - radius, df_plot["platelocheight_in"].max() + radius
    x_min, x_max = min(zone_left - 6, x_min_data), max(zone_right + 6, x_max_data)
    y_min, y_max = min(zone_bottom - 6, y_min_data), max(zone_top + 6, y_max_data)

    fig.update_xaxes(title_text="PlateLocSide (inches)", range=[x_min, x_max], scaleanchor="y", scaleratio=1, zeroline=False)
    fig.update_yaxes(title_text="PlateLocHeight (inches)", range=[y_min, y_max], zeroline=False)
    fig.update_layout(title_text="Strike Zone (circles = 2.9 inches)", height=800, width=800, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

with col_left:
    st.subheader("Summary")
    st.write(f"Showing {total_pitches} pitches")
    if pitcher_col in df_plot.columns:
        top_pitchers = df_plot[pitcher_col].value_counts().head(10)
        st.write("Top pitchers (by pitch count)")
        st.table(top_pitchers.rename_axis("pitcher").reset_index(name="count"))
    if batter_col in df_plot.columns:
        top_batters = df_plot[batter_col].value_counts().head(10)
        st.write("Top batters (by pitch count)")
        st.table(top_batters.rename_axis("batter").reset_index(name="count"))

# -----------------------------
# Optional histogram of plate height
# -----------------------------
st.subheader("Pitch location distributions")
if "platelocheight_in" in df_plot.columns:
    h_fig = go.Figure()
    h_fig.add_trace(go.Histogram(x=df_plot["platelocheight_in"], nbinsx=30))
    h_fig.update_layout(title_text="Distribution of PlateLocHeight (inches)", xaxis_title="Height (in)", yaxis_title="Count")
    st.plotly_chart(h_fig, use_container_width=True)
