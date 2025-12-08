# run_align.py
import pandas as pd
from align import align_game

pxp = pd.read_csv("Feb21Texas.csv")
tm = pd.read_csv("2-21-25_Texas1_Trackman.csv")
try:
    merged = align_game(tm, pxp, game_id="2025-02-21-TEX-1")
    print("Merged mapping rows:", len(merged))
    print(merged[["trackman_idx","pxp_idx","match_type","confidence","comment"]].to_string(index=False))
    merged.to_csv("mapping_debug.csv", index=False)
    print("Saved mapping_debug.csv")
except ValueError as e:
    print("Anomalies detected:")
    print(e)
