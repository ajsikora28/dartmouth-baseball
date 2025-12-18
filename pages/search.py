import streamlit as st
import pandas as pd
import os

def check_login():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("Please log in first")
        st.stop()

check_login()

st.title("Search Games / Players")

# Example: load all PxP CSVs
pxp_files = [f for f in os.listdir("data") if f.startswith("pxp_")]
selected_file = st.selectbox("Select a PxP file", pxp_files)

if selected_file:
    df = pd.read_csv(os.path.join("data", selected_file))
    st.write("First few rows:")
    st.dataframe(df)
    
    # Example filter by player
    if "player_name" in df.columns:
        player = st.text_input("Filter by Player Name")
        if player:
            filtered = df[df["player_name"].str.contains(player, case=False, na=False)]
            st.dataframe(filtered)
