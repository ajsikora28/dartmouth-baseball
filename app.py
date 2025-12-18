# app.py
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# ---------------------------
# Simple login
# ---------------------------

# Stored in Streamlit secrets
USERNAME = st.secrets["USERNAME"]
PASSWORD = st.secrets["PASSWORD"]

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login form only if not logged in
if not st.session_state["logged_in"]:
    st.sidebar.title("Login")
    user_input = st.sidebar.text_input("Username")
    pw_input = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if login_btn:
        if user_input == USERNAME and pw_input == PASSWORD:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.sidebar.error("Incorrect username or password")

# Stop the app if not logged in
if not st.session_state["logged_in"]:
    st.stop()


# --------- config ----------
DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.json")
PRIMARY_COLOR = "#00693e"
SECONDARY_COLOR = "#12312b"
# ---------------------------

# Minimal CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {SECONDARY_COLOR} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Dartmouth Baseball Analytics")
st.header("Schedule")

os.makedirs(DATA_DIR, exist_ok=True)

# ----- helpers for persistence -----
def load_schedule() -> pd.DataFrame:
    if os.path.exists(SCHEDULE_FILE):
        try:
            with open(SCHEDULE_FILE, "r", encoding="utf8") as fh:
                data = json.load(fh)
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame(columns=["Date", "Opponent", "Trackman CSV", "PxP CSV", "Year"])
    else:
        return pd.DataFrame(columns=["Date", "Opponent", "Trackman CSV", "PxP CSV", "Year"])

def save_schedule(df: pd.DataFrame):
    # convert datelike objects to ISO strings for JSON
    copy = df.copy()
    if "Date" in copy.columns:
        copy["Date"] = copy["Date"].astype(str)
    with open(SCHEDULE_FILE, "w", encoding="utf8") as fh:
        json.dump(copy.to_dict(orient="records"), fh, indent=2, ensure_ascii=False)

# ----- session state initialization -----
if "years" not in st.session_state:
    st.session_state.years = [2023, 2024, 2025]

if "selected_year" not in st.session_state:
    st.session_state.selected_year = st.session_state.years[0]

# load schedule into session state on first run
if "schedule" not in st.session_state:
    st.session_state.schedule = load_schedule()

# helper to get year from date-like values
def _get_year(d):
    try:
        return d.year
    except Exception:
        try:
            return pd.to_datetime(d).year
        except Exception:
            return None

# ------ Year selection + inline add-year form ------
year_options = st.session_state.years + ["Add new year..."]

# show selectbox; choose index of selected_year if present
index = year_options.index(st.session_state.selected_year) if st.session_state.selected_year in year_options else 0
choice = st.selectbox("Select Season", year_options, index=index)

if choice == "Add new year...":
    # Using a form so submit triggers a clean rerun
    with st.form("add_year_form", clear_on_submit=True):
        new_year = st.number_input("Enter new year", min_value=2000, max_value=2100, step=1, value=datetime.now().year)
        submitted = st.form_submit_button("Add year")
        if submitted:
            if new_year not in st.session_state.years:
                st.session_state.years.append(int(new_year))
                st.session_state.years.sort()
            st.session_state.selected_year = int(new_year)
            st.success(f"Added and selected {new_year}")
            # persist nothing else needed here — the rerun will update UI
else:
    st.session_state.selected_year = choice

st.write(f"Current selected year: **{st.session_state.selected_year}**")

# ------ Display schedule filtered by year ------
df_schedule = st.session_state.schedule.copy()
# normalize Date column to string for display
if not df_schedule.empty and "Year" in df_schedule.columns:
    filtered_schedule = df_schedule[df_schedule["Year"] == int(st.session_state.selected_year)]
else:
    filtered_schedule = pd.DataFrame(columns=["Date", "Opponent", "Trackman CSV", "PxP CSV", "Year"])

st.subheader("Season Schedule")
st.dataframe(filtered_schedule[["Date", "Opponent", "Trackman CSV", "PxP CSV"]], width='stretch')

# ------ Add a game form ------
st.subheader("Add a Game")
with st.form("add_game_form", clear_on_submit=True):
    default_date = datetime(year=int(st.session_state.selected_year), month=1, day=1).date()
    date_input = st.date_input("Date", value=default_date)
    opponent_input = st.text_input("Opponent")
    add_game_btn = st.form_submit_button("Add Game")
    if add_game_btn:
        if not opponent_input:
            st.warning("Please enter an opponent name.")
        else:
            new_row = {
                "Date": str(date_input),  # store as string for JSON
                "Opponent": opponent_input,
                "Trackman CSV": "",
                "PxP CSV": "",
                "Year": int(_get_year(date_input))
            }
            st.session_state.schedule = pd.concat([st.session_state.schedule, pd.DataFrame([new_row])], ignore_index=True)
            save_schedule(st.session_state.schedule)
            st.success("Game added!")

# ------ Manage existing games (one at a time) ------
st.subheader("Manage Games")
# Only show games for the selected season
season_games = st.session_state.schedule[st.session_state.schedule["Year"] == int(st.session_state.selected_year)].reset_index()
if season_games.empty:
    st.info("No games added yet for this season.")
else:
    # build label and original index mapping
    labels = [f"{row['Date']} — {row['Opponent']}" for _, row in season_games.iterrows()]
    sel = st.selectbox("Choose a game to manage", ["(select)"] + labels)
    if sel != "(select)":
        sel_idx = labels.index(sel)
        orig_idx = int(season_games.loc[sel_idx, "index"])
        row = st.session_state.schedule.loc[orig_idx]

        # Use expander to show the manage UI
        with st.expander(f"Manage: {row['Date']} — {row['Opponent']}", expanded=True):
            st.write(f"Trackman file: {row.get('Trackman CSV','') or 'None'}")
            st.write(f"PxP file: {row.get('PxP CSV','') or 'None'}")

            # upload Trackman
            with st.form(f"upload_trackman_{orig_idx}", clear_on_submit=True):
                trackman_file = st.file_uploader("Upload Trackman CSV", type=["csv"], key=f"tm_{orig_idx}")
                submit_tm = st.form_submit_button("Save Trackman")
                if submit_tm and trackman_file:
                    safe_op = "".join(c for c in row['Opponent'] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                    filename = f"trackman_{st.session_state.selected_year}_{safe_op}.csv"
                    path = os.path.join(DATA_DIR, filename)
                    with open(path, "wb") as fh:
                        fh.write(trackman_file.getbuffer())
                    st.session_state.schedule.at[orig_idx, "Trackman CSV"] = path
                    save_schedule(st.session_state.schedule)
                    st.success("Saved Trackman CSV")

            # upload PxP
            with st.form(f"upload_pxp_{orig_idx}", clear_on_submit=True):
                pxp_file = st.file_uploader("Upload play-by-play (PxP) CSV", type=["csv"], key=f"pxp_{orig_idx}")
                submit_pxp = st.form_submit_button("Save PxP")
                if submit_pxp and pxp_file:
                    safe_op = "".join(c for c in row['Opponent'] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                    filename = f"pxp_{st.session_state.selected_year}_{safe_op}.csv"
                    path = os.path.join(DATA_DIR, filename)
                    with open(path, "wb") as fh:
                        fh.write(pxp_file.getbuffer())
                    st.session_state.schedule.at[orig_idx, "PxP CSV"] = path
                    save_schedule(st.session_state.schedule)
                    st.success("Saved PxP CSV")

                    # ----- Run alignment (attempt even if Trackman is missing) -----
                    try:
                        import align  # ensure align.py is importable
                        df_px = pd.read_csv(path)

                        # try to load Trackman if we have a path, else use None
                        trackman_path = row.get("Trackman CSV", "") or ""
                        if trackman_path and os.path.exists(trackman_path):
                            df_tm = pd.read_csv(trackman_path)
                            tm_msg = "Trackman file found; running full alignment."
                        else:
                            df_tm = None
                            tm_msg = "No Trackman file found; running alignment with PxP only (Trackman side will be empty/unmatched)."

                        game_id = f"{st.session_state.selected_year}_{safe_op}"

                        # Run alignment. align_game handles None/empty Trackman (it will mark px rows as unmatched).
                        merged = align.align_game(df_tm, df_px, game_id)

                        # Save merged CSV (this will overwrite an existing merged_{game_id}.csv if present)
                        merged_filename = f"merged_{game_id}.csv"
                        merged_path = os.path.join(DATA_DIR, merged_filename)
                        merged.to_csv(merged_path, index=False)
                        st.success(f"{tm_msg} Merged file saved to {merged_path}")

                    except ValueError as e:
                        # align.align_game raises ValueError with formatted problems when anomalies are found
                        st.error(f"Alignment failed with errors:\n{e}")
                    except Exception as e:
                        st.error(f"Unexpected error during alignment:\n{e}")

            # delete button
            if st.button("Delete game", key=f"del_{orig_idx}"):
                st.session_state.schedule = st.session_state.schedule.drop(orig_idx).reset_index(drop=True)
                save_schedule(st.session_state.schedule)
                st.success("Game deleted")
                # After deletion, selecting the default in the next run is fine.

