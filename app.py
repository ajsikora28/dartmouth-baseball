import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Custom color scheme
PRIMARY_COLOR = "#00693e"
SECONDARY_COLOR = "#12312b"

# Apply minimal custom CSS styling: only set the main page background
st.markdown(f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --secondary-color: {SECONDARY_COLOR};
    }}

    /* Only color the main app background; leave all other styles default */
    .stApp {{
        background-color: {SECONDARY_COLOR} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Dartmouth Baseball Analytics")
st.header("Schedule")

# Initialize state
if "years" not in st.session_state:
    st.session_state.years = [2023, 2024, 2025]

if "selected_year" not in st.session_state:
    st.session_state.selected_year = st.session_state.years[0]

if "adding_year" not in st.session_state:
    st.session_state.adding_year = False

# Prepare dropdown options
year_options = st.session_state.years + ["Add new year..."]

def on_year_select():
    selected = st.session_state.year_select
    if selected == "Add new year...":
        st.session_state.adding_year = True
    else:
        st.session_state.selected_year = selected
        st.session_state.adding_year = False

# Show dropdown, set index to selected_year if it exists
index = year_options.index(st.session_state.selected_year) if st.session_state.selected_year in year_options else 0
st.selectbox("Select Season", year_options, index=index, key="year_select", on_change=on_year_select)

# Show modal dialog for adding a new year
@st.dialog("Add New Year")
def add_year_dialog():
    new_year = st.number_input("Enter new year", min_value=2000, max_value=2100, step=1, key="new_year_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add", use_container_width=True):
            if new_year not in st.session_state.years:
                st.session_state.years.append(new_year)
                st.session_state.years.sort()  # insert in chronological order
                st.session_state.selected_year = new_year  # select new year
            st.session_state.adding_year = False  # hide input
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.adding_year = False
            st.rerun()

# Trigger the dialog if adding_year is True
if st.session_state.adding_year:
    add_year_dialog()

# Initialize schedule
if "schedule" not in st.session_state:
    # keep a Year column so we can filter by season when DB is added
    st.session_state.schedule = pd.DataFrame(columns=["Date", "Opponent", "Trackman CSV", "PxP CSV", "Year"])

# Helper to extract year from a Date-like value
def _get_year(d):
    try:
        return d.year
    except Exception:
        try:
            return pd.to_datetime(d).year
        except Exception:
            return None

# Display schedule (only show games for the selected season)
filtered_schedule = st.session_state.schedule[st.session_state.schedule["Year"] == st.session_state.selected_year]
st.dataframe(filtered_schedule[["Date", "Opponent", "Trackman CSV", "PxP CSV"]])

# Add a game manually
st.subheader("Add a Game")
# Set default date to January 1st of the selected year
default_date = datetime(year=st.session_state.selected_year, month=1, day=1).date()
date = st.date_input("Date", value=default_date)
opponent = st.text_input("Opponent")
if st.button("Add Game"):
    new_row = {"Date": date, "Opponent": opponent, "Trackman CSV": "", "PxP CSV": "", "Year": _get_year(date)}
    st.session_state.schedule = pd.concat([
        st.session_state.schedule,
        pd.DataFrame([new_row])
    ], ignore_index=True)
    st.success("Game added!")
    st.rerun()

# Manage Games with dialog
if "managing_game_idx" not in st.session_state:
    st.session_state.managing_game_idx = None

st.subheader("Manage Games")

if len(st.session_state.schedule) > 0:
    # Only offer games in the selected season
    filtered = st.session_state.schedule[st.session_state.schedule["Year"] == st.session_state.selected_year].reset_index()
    game_options = [f"{row['Date']} vs {row['Opponent']}" for _, row in filtered.iterrows()]
    orig_indices = list(filtered['index'])

    def on_game_select():
        selected = st.session_state.game_select
        if selected != "Select a game...":
            idx_in_options = game_options.index(selected)
            # Map the selected option back to the original schedule index
            st.session_state.managing_game_idx = orig_indices[idx_in_options]

    # If a previous run requested the game_select to be cleared, do that
    # before instantiating the selectbox widget to avoid Streamlit errors
    if st.session_state.get("clear_game_select", False):
        st.session_state.game_select = "Select a game..."
        st.session_state.pop("clear_game_select", None)

    st.selectbox("Select a game to manage", ["Select a game..."] + game_options, key="game_select", on_change=on_game_select)

    # Show modal dialog for managing a selected game
    @st.dialog("Manage Game")
    def manage_game_dialog():
        idx = st.session_state.managing_game_idx
        row = st.session_state.schedule.loc[idx]

        st.write(f"**{row['Date']} vs {row['Opponent']}**")

        # Upload Trackman
        trackman_file = st.file_uploader(f"Trackman CSV for {row['Opponent']}", type="csv", key=f"trackman_{idx}")
        if trackman_file:
            os.makedirs("data", exist_ok=True)
            trackman_path = os.path.join("data", f"trackman_{st.session_state.selected_year}_{row['Opponent']}.csv")
            with open(trackman_path, "wb") as f:
                f.write(trackman_file.getbuffer())
            st.session_state.schedule.at[idx, "Trackman CSV"] = trackman_path
            st.success("Trackman file saved!")
            st.rerun()

        # Upload PxP
        pxp_file = st.file_uploader(f"PxP CSV for {row['Opponent']}", type="csv", key=f"pxp_{idx}")
        if pxp_file:
            os.makedirs("data", exist_ok=True)
            pxp_path = os.path.join("data", f"pxp_{st.session_state.selected_year}_{row['Opponent']}.csv")
            with open(pxp_path, "wb") as f:
                f.write(pxp_file.getbuffer())
            st.session_state.schedule.at[idx, "PxP CSV"] = pxp_path
            st.success("PxP file saved!")
            st.rerun()

        # Delete game button
        if st.button(f"Delete Game: {row['Opponent']}", use_container_width=True):
            st.session_state.schedule = st.session_state.schedule.drop(idx).reset_index(drop=True)
            st.session_state.managing_game_idx = None
            st.success(f"Game vs {row['Opponent']} deleted!")
            st.rerun()

    # Trigger the dialog if a game is selected. Show it once and then
    # reset the managing index and request that the selectbox be cleared
    # on the next run. Do NOT call st.rerun() here — calling rerun
    # immediately would interrupt the dialog lifecycle.
    if st.session_state.managing_game_idx is not None:
        manage_game_dialog()
        st.session_state.managing_game_idx = None
        st.session_state.clear_game_select = True
else:
    st.write("No games added yet.")
