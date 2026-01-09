# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from supabase import create_client

# ---------------------------
# Supabase setup
# ---------------------------
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

# ---------------------------
# Simple login
# ---------------------------
USERNAME = st.secrets["USERNAME"]
PASSWORD = st.secrets["PASSWORD"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("Login")
    user_input = st.sidebar.text_input("Username")
    pw_input = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if login_btn:
        if user_input == USERNAME and pw_input == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Incorrect username or password")

if not st.session_state.logged_in:
    st.stop()

# ---------------------------
# Config + styling
# ---------------------------
PRIMARY_COLOR = "#00693e"
SECONDARY_COLOR = "#12312b"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {SECONDARY_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dartmouth Baseball Analytics")
st.header("Schedule")

# ---------------------------
# Helpers
# ---------------------------
def _safe_data(resp):
    """Safely extract data from supabase-py v2 response."""
    return resp.model_dump().get("data") or []

def _get_year(d):
    try:
        return d.year
    except Exception:
        try:
            return pd.to_datetime(d).year
        except Exception:
            return None

# ---------------------------
# Load seasons
# ---------------------------
@st.cache_data(ttl=60)
def load_seasons():
    resp = supabase.table("games").select("season").execute()
    data = _safe_data(resp)
    seasons = sorted({row["season"] for row in data if row.get("season") is not None})
    return seasons

# ---------------------------
# Session state init
# ---------------------------
if "years" not in st.session_state:
    st.session_state.years = load_seasons()

if "selected_year" not in st.session_state:
    st.session_state.selected_year = (
        st.session_state.years[-1] if st.session_state.years else None
    )

# ---------------------------
# Season selector
# ---------------------------
year_options = st.session_state.years + ["Add new year..."]

index = (
    st.session_state.years.index(st.session_state.selected_year)
    if st.session_state.selected_year in st.session_state.years
    else 0
)

choice = st.selectbox("Select Season", year_options, index=index)

if choice == "Add new year...":
    st.info("Seasons are created automatically when you add a game.")
    st.session_state.selected_year = None
else:
    st.session_state.selected_year = int(choice)

st.write(f"Current selected year: **{st.session_state.selected_year}**")

# ---------------------------
# Load schedule
# ---------------------------
if st.session_state.selected_year is not None:
    resp = (
        supabase
        .table("games")
        .select("*")
        .eq("season", st.session_state.selected_year)
        .order("date")
        .execute()
    )
    df_schedule = pd.DataFrame(_safe_data(resp))
else:
    df_schedule = pd.DataFrame()

# ---------------------------
# Display schedule
# ---------------------------
st.subheader("Season Schedule")

if df_schedule.empty:
    st.info("No games added yet for this season.")
else:
    st.dataframe(df_schedule[["date", "opponent"]], width="stretch")

# ---------------------------
# Add game
# ---------------------------
st.subheader("Add a Game")

with st.form("add_game_form", clear_on_submit=True):
    default_year = st.session_state.selected_year or datetime.now().year
    default_date = datetime(year=default_year, month=1, day=1).date()
    date_input = st.date_input("Date", value=default_date)
    opponent_input = st.text_input("Opponent")
    submit = st.form_submit_button("Add Game")

    if submit:
        if not opponent_input:
            st.warning('Please enter an opponent name (use "Intrasquad" if applicable).')
        else:
            season_year = _get_year(date_input)
            existing = (
                supabase
                .table("games")
                .select("id")
                .eq("season", season_year)
                .eq("date", str(date_input))
                .eq("opponent", opponent_input)
                .execute()
            )

            if _safe_data(existing):
                st.warning("This game already exists.")
            else:
                supabase.table("games").insert({
                    "season": season_year,
                    "date": str(date_input),
                    "opponent": opponent_input,
                }).execute()

                load_seasons.clear()
                st.session_state.years = load_seasons()
                st.session_state.selected_year = season_year

                st.success("Game added!")
                st.rerun()

# ---------------------------
# Manage games
# ---------------------------
st.subheader("Manage Games")

if df_schedule.empty:
    st.info("No games to manage.")
else:
    labels = [f"{row['date']} — {row['opponent']}" for _, row in df_schedule.iterrows()]
    sel = st.selectbox("Choose a game to manage", ["(select)"] + labels)

    if sel != "(select)":
        game = df_schedule.iloc[labels.index(sel)]
        game_id = game["id"]
        opponent = game["opponent"]
        season = game["season"]

        safe_op = "".join(c for c in opponent if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

        with st.expander(f"Manage: {game['date']} — {opponent}", expanded=True):

            # ---- Trackman upload ----
            with st.form(f"upload_trackman_{game_id}", clear_on_submit=True):
                file = st.file_uploader("Upload Trackman CSV", type=["csv"])
                submit = st.form_submit_button("Save Trackman")

                if submit and file:
                    path = f"{season}/{safe_op}/trackman.csv"

                    supabase.storage.from_("game-data").upload(
                        path,
                        file.getvalue(),
                        file_options={"content-type": "text/csv", "upsert": True},
                    )

                    supabase.table("game_files").delete() \
                        .eq("game_id", game_id) \
                        .eq("file_type", "trackman") \
                        .execute()

                    supabase.table("game_files").insert({
                        "game_id": game_id,
                        "file_type": "trackman",
                        "storage_path": path,
                    }).execute()

                    st.success("Trackman uploaded")

            # ---- PxP upload ----
            with st.form(f"upload_pxp_{game_id}", clear_on_submit=True):
                file = st.file_uploader("Upload play-by-play CSV", type=["csv"])
                submit = st.form_submit_button("Save PxP")

                if submit and file:
                    path = f"{season}/{safe_op}/pxp.csv"

                    supabase.storage.from_("game-data").upload(
                        path,
                        file.getvalue(),
                        file_options={"content-type": "text/csv", "upsert": True},
                    )

                    supabase.table("game_files").delete() \
                        .eq("game_id", game_id) \
                        .eq("file_type", "pxp") \
                        .execute()

                    supabase.table("game_files").insert({
                        "game_id": game_id,
                        "file_type": "pxp",
                        "storage_path": path,
                    }).execute()

                    st.success("PxP uploaded")

            # ---- Delete game ----
            if st.button("Delete game"):
                supabase.table("game_files").delete().eq("game_id", game_id).execute()
                supabase.table("games").delete().eq("id", game_id).execute()
                st.success("Game deleted")
                st.rerun()
