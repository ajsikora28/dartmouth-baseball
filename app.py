# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from supabase import create_client

# Get credentials from Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]

# Create Supabase client
supabase = create_client(supabase_url, supabase_key)


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

@st.cache_data(ttl=60)
def load_seasons():
    resp = (
        supabase
        .table("games")
        .select("season")
        .execute()
    )

    if resp.status_code != 200 or not resp.data:
        return []

    seasons = sorted(
        {row["season"] for row in resp.data if row.get("season") is not None}
    )
    return seasons

# ----- session state initialization -----
if "years" not in st.session_state:
    st.session_state.years = load_seasons()

if "selected_year" not in st.session_state:
    if st.session_state.years:
        st.session_state.selected_year = st.session_state.years[-1]  # most recent
    else:
        st.session_state.selected_year = datetime.now().year

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
if st.session_state.selected_year in st.session_state.years:
    index = st.session_state.years.index(st.session_state.selected_year)
else:
    index = 0
choice = st.selectbox("Select Season", year_options, index=index)

if choice == "Add new year...":
    st.info("Seasons are created automatically when you add a game.")
    st.stop()
else:
    st.session_state.selected_year = choice

st.write(f"Current selected year: **{st.session_state.selected_year}**")

# ------ Load schedule from Supabase ------
resp = (
    supabase
    .table("games")
    .select("*")
    .eq("season", int(st.session_state.selected_year))
    .order("date")
    .execute()
)

# Use model_dump() to access the data
resp_dict = resp.model_dump()  # converts Pydantic object to dict

# resp_dict contains keys: "data", "count", etc.
if resp_dict.get("status_code") != 200  or not resp.data:  # or check if "data" is None
    err_msg = resp_dict.get("message", "Unknown error")
    st.error(f"Error loading schedule: {err_msg}")
    df_schedule = pd.DataFrame()
else:
    df_schedule = pd.DataFrame(resp_dict.get("data", []))


st.subheader("Season Schedule")

if df_schedule.empty:
    st.info("No games added yet for this season.")
else:
    st.dataframe(
        df_schedule[["date", "opponent"]],
        width="stretch"
    )

# ------ Add a game form ------
st.subheader("Add a Game")
with st.form("add_game_form", clear_on_submit=True):
    default_date = datetime(year=int(st.session_state.selected_year), month=1, day=1).date()
    date_input = st.date_input("Date", value=default_date)
    opponent_input = st.text_input("Opponent")
    add_game_btn = st.form_submit_button("Add Game")
    if add_game_btn:
        if not opponent_input:
            st.warning("Please enter an opponent name (for intrasquad, enter \"Intrasquad\").")
        existing = (
            supabase
            .table("games")
            .select("id")
            .eq("season", int(_get_year(date_input)))
            .eq("date", str(date_input))
            .eq("opponent", opponent_input)
            .execute()
        )
        if existing.data:
            st.warning("This game already exists.")
        else:
            supabase.table("games").insert({
                "season": int(_get_year(date_input)),
                "date": str(date_input),
                "opponent": opponent_input
            }).execute()

            load_seasons.clear()
            st.session_state.years = load_seasons()
            st.session_state.selected_year = int(_get_year(date_input))

            st.success("Game added!")
            st.rerun()


# ------ Manage existing games ------
st.subheader("Manage Games")

if df_schedule.empty:
    st.info("No games to manage.")
else:
    labels = [
        f"{row['date']} — {row['opponent']}"
        for _, row in df_schedule.iterrows()
    ]

    sel = st.selectbox("Choose a game to manage", ["(select)"] + labels)

    if sel != "(select)":
        sel_idx = labels.index(sel)
        game = df_schedule.iloc[sel_idx]
        game_id = game["id"]
        opponent = game["opponent"]
        season = game["season"]

        safe_op = (
            "".join(c for c in opponent if c.isalnum() or c in (" ", "-", "_"))
            .strip()
            .replace(" ", "_")
        )

        with st.expander(f"Manage: {game['date']} — {opponent}", expanded=True):

            # ---- Upload Trackman ----
            with st.form(f"upload_trackman_{game_id}", clear_on_submit=True):
                trackman_file = st.file_uploader(
                    "Upload Trackman CSV",
                    type=["csv"]
                )
                submit_tm = st.form_submit_button("Save Trackman")

                if submit_tm and trackman_file:
                    storage_path = f"{season}/{safe_op}/trackman.csv"

                    supabase.storage.from_("game-data").upload(
                        storage_path,
                        trackman_file.getvalue(),
                        file_options={
                            "content-type": "text/csv",
                            "upsert": True
                        }
                    )


                    supabase.table("game_files") \
                        .delete() \
                        .eq("game_id", game_id) \
                        .eq("file_type", "trackman") \
                        .execute()

                    supabase.table("game_files").insert({
                        "game_id": game_id,
                        "file_type": "trackman",
                        "storage_path": storage_path
                    }).execute()


                    st.success("Trackman uploaded")


            # ---- Upload PxP ----
            with st.form(f"upload_pxp_{game_id}", clear_on_submit=True):
                pxp_file = st.file_uploader(
                    "Upload play-by-play (PxP) CSV",
                    type=["csv"]
                )
                submit_pxp = st.form_submit_button("Save PxP")

                if submit_pxp and pxp_file:
                    storage_path = f"{season}/{safe_op}/pxp.csv"

                    supabase.storage.from_("game-data").upload(
                        storage_path,
                        pxp_file.getvalue(),
                        file_options={
                            "content-type": "text/csv",
                            "upsert": True
                        }
                    )


                    supabase.table("game_files") \
                        .delete() \
                        .eq("game_id", game_id) \
                        .eq("file_type", "pxp") \
                        .execute()

                    supabase.table("game_files").insert({
                        "game_id": game_id,
                        "file_type": "pxp",
                        "storage_path": storage_path
                    }).execute()


                    st.success("PxP uploaded")


            # ---- Delete game ----
            if st.button("Delete game"):
                supabase.table("game_files").delete().eq("game_id", game_id).execute()
                supabase.table("games").delete().eq("id", game_id).execute()
                st.success("Game deleted")
                st.rerun()
