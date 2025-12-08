# align.py
"""
Align Trackman <-> Play-by-Play (PxP) with anomaly detection.

If anomalies are detected in the input files (missing PA/pitch IDs, duplicates,
gaps, missing names, mismatched batters per PA, duplicate PitchNo), align_game()
raises a ValueError with a readable list of problems and suggestions.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import re

# -----------------------
# Utilities / normalizers
# -----------------------
def normalize_name_last_first(s: Optional[str]) -> str:
    if pd.isna(s) or s is None:
        return ""
    s2 = str(s).strip()
    s2 = " ".join(s2.split())
    # Fix spacing around commas (e.g., "Last , First" → "Last, First")
    s2 = re.sub(r"\s*,\s*", ", ", s2)
    return s2.lower()

def safe_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _tb_token(row) -> str:
    """Return 'T' for Top, 'B' for Bottom, 'U' for unknown. Accepts many column name variants."""
    # prefer already-normalized 'top_bottom' if present, else check common column names
    tb_val = None
    for c in ("top_bottom", "Top/Bottom", "TopBottom", "Top_Bottom", "Top/Bottom"):
        if c in row.index:
            tb_val = row[c]
            break

    if tb_val is None or pd.isna(tb_val):
        return "U"

    s = str(tb_val).strip().lower()
    if s.startswith("t"):
        return "T"
    if s.startswith("b"):
        return "B"
    if s in ("top",):
        return "T"
    if s in ("bottom", "bot"):
        return "B"
    return "U"

# -----------------------
# Preprocessing
# -----------------------
def preprocess_trackman(df_tm: pd.DataFrame, game_id: str) -> pd.DataFrame:
    df = df_tm.copy().reset_index(drop=True).reset_index().rename(columns={"index": "trackman_idx"})

    # --- DROP EMPTY ROWS BEFORE CREATING pa_id ---
    # Consider a row empty if all columns except trackman_idx are blank/NaN
    cols_to_check = [c for c in df.columns if c != "trackman_idx"]
    df = df[df[cols_to_check].apply(lambda row: row.notna() & (row.astype(str).str.strip() != "")).any(axis=1)]

    # optional timestamp
    ts_col = safe_col(df, ["Timestamp", "Time", "timestamp", "time"])
    if ts_col:
        try:
            df["ts"] = pd.to_datetime(df[ts_col], errors="coerce", format="%Y-%m-%dT%H:%M:%S")
            if df["ts"].isna().all():
                df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        except Exception:
            df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        df["ts"] = pd.NaT

    # inning
    inning_col = safe_col(df, ["Inning", "InningNo", "inning"])
    df["inning"] = pd.to_numeric(df[inning_col], errors="coerce").fillna(0).astype(int) if inning_col else pd.NA

    # PAofInning
    pa_col = safe_col(df, ["PAofInning", "PaOfInning", "PA_of_Inning"])
    df["pa_of_inning"] = pd.to_numeric(df[pa_col], errors="coerce").fillna(np.nan).astype("Int64") if pa_col else pd.NA

    # PitchofPA
    ppa_col = safe_col(df, ["PitchofPA", "PitchOfPA", "Pitch_of_PA"])
    df["pitch_of_pa"] = pd.to_numeric(df[ppa_col], errors="coerce").fillna(np.nan).astype("Int64") if ppa_col else pd.NA

    # PitchNo (global)
    pno_col = safe_col(df, ["PitchNo", "PitchNoTotal", "Pitch_No", "PitchNumber"])
    if pno_col:
        df["pitch_no"] = pd.to_numeric(df[pno_col], errors="coerce")
    else:
        df["pitch_no"] = (df["trackman_idx"] + 1).astype(int)

    # names
    batter_col = safe_col(df, ["Batter", "BatterName", "batter", "batter_name", "Hitter"])
    df["batter_name"] = df[batter_col].astype(str).apply(normalize_name_last_first) if batter_col else ""

    pitcher_col = safe_col(df, ["Pitcher", "PitcherName", "pitcher"])
    df["pitcher_name"] = df[pitcher_col].astype(str).apply(normalize_name_last_first) if pitcher_col else ""

    # synthesize pa_of_inning if missing globally by change-of-batter (game-scoped)
    if (pa_col is None) or df["pa_of_inning"].isna().all():
        df = df.sort_values(by=["pitch_no"]).reset_index(drop=True)
        df["pa_of_inning"] = (df["batter_name"] != df["batter_name"].shift(1)).cumsum().astype("Int64")

    # compose pa_id
    def make_pa_id_tm(row, game_id: str):
        """Create unique pa_id for a Trackman row; robust to missing inning/pa_of_inning."""
        tb = _tb_token(row)
        inning = row.get("inning", None)
        pa = row.get("pa_of_inning", None)

        # if both inning and pa are present and not NA -> use full form
        if (inning is not None and not pd.isna(inning)) and (pa is not None and not pd.isna(pa)):
            return f"{game_id}_{tb}_inning{int(inning)}_pa{int(pa)}"

        # fallback: try to use trackman_idx if available
        tm_idx = row.get("trackman_idx", None)
        if tm_idx is not None and not pd.isna(tm_idx):
            return f"{game_id}_{tb}_tmidx{int(tm_idx)}"

        # final fallback: generic unknown
        return f"{game_id}_{tb}_paUNKNOWN"

    df["pa_id"] = df.apply(lambda r: make_pa_id_tm(r, game_id), axis=1)

    df["game_id"] = game_id
    return df

def preprocess_pxp(df_px: pd.DataFrame, game_id: str) -> pd.DataFrame:
    df = df_px.copy()
    # DROP columns with no title or named "ROSTER"
    df = df[[c for c in df.columns if c.strip() != "" and c.upper() != "ROSTER"]]
    df = df.reset_index(drop=True).reset_index().rename(columns={"index": "pxp_idx"})


    # --- DROP EMPTY ROWS BEFORE CREATING pa_id ---
    # Consider a row empty if all columns except pxp_idx are blank/NaN
    cols_to_check = [c for c in df.columns if c != "pxp_idx"]
    df = df[df[cols_to_check].apply(lambda row: row.notna() & (row.astype(str).str.strip() != "")).any(axis=1)]

    # inning
    inning_col = safe_col(df, ["Inning", "InningNo", "inning"])
    df["inning"] = pd.to_numeric(df[inning_col], errors="coerce").fillna(np.nan).astype("Int64") if inning_col else pd.NA

    # top/bottom
    topbot_col = safe_col(df, ["Top/Bottom", "TopBottom", "Top_Bottom", "TopBottom"])
    df["top_bottom"] = df[topbot_col].astype(str) if topbot_col else ""

    # PAofInning & PitchofPA
    pa_col = safe_col(df, ["PAofInning", "PA", "PaOfInning"])
    df["pa_of_inning"] = pd.to_numeric(df[pa_col], errors="coerce").fillna(np.nan).astype("Int64") if pa_col else pd.NA

    ppa_col = safe_col(df, ["PitchofPA", "PitchOfPA", "Pitch_of_PA"])
    df["pitch_of_pa"] = pd.to_numeric(df[ppa_col], errors="coerce").fillna(np.nan).astype("Int64") if ppa_col else pd.NA

    # global pitch number if present (else synthesize)
    if "PitchNo" in df.columns:
        df["pitch_no"] = pd.to_numeric(df["PitchNo"], errors="coerce")
    else:
        df["pitch_no"] = (df.reset_index().index + 1).astype(int)

    # names
    batter_col = safe_col(df, ["Batter", "BatterName", "batter", "batter_name"])
    df["batter_name"] = df[batter_col].astype(str).apply(normalize_name_last_first) if batter_col else ""

    pitcher_col = safe_col(df, ["Pitcher", "PitcherName", "pitcher"])
    df["pitcher_name"] = df[pitcher_col].astype(str).apply(normalize_name_last_first) if pitcher_col else ""

    # synthesize pa_of_inning if missing
    if (pa_col is None) or df["pa_of_inning"].isna().all():
        if df["inning"].notna().any():
            df = df.sort_values(by=["inning"]).reset_index(drop=True)
            df["pa_of_inning"] = (df["batter_name"] != df["batter_name"].shift(1)).cumsum().astype("Int64")
        else:
            df["pa_of_inning"] = (df["batter_name"] != df["batter_name"].shift(1)).cumsum().astype("Int64")

    # pitch_of_pa if missing
    if (ppa_col is None) or df["pitch_of_pa"].isna().all():
        df["pitch_of_pa"] = df.groupby("pa_of_inning").cumcount() + 1
        df["pitch_of_pa"] = df["pitch_of_pa"].astype("Int64")

    def make_pa_id_px(row, game_id: str):
        """Create unique pa_id for a PxP row; robust to missing inning/pa_of_inning."""
        tb = _tb_token(row)
        inning = row.get("inning", None)
        pa = row.get("pa_of_inning", None)

        if (inning is not None and not pd.isna(inning)) and (pa is not None and not pd.isna(pa)):
            return f"{game_id}_{tb}_inning{int(inning)}_pa{int(pa)}"

        # fallback: use pxp_idx if present
        px_idx = row.get("pxp_idx", None)
        if px_idx is not None and not pd.isna(px_idx):
            return f"{game_id}_{tb}_pxidx{int(px_idx)}"

        return f"{game_id}_{tb}_paUNKNOWN"

    df["pa_id"] = df.apply(lambda r: make_pa_id_px(r, game_id), axis=1)


    df["game_id"] = game_id
    df["pxp_pitch_seq"] = df.reset_index().index + 1
    return df

# -----------------------
# Anomaly detection
# -----------------------
def _row_info(side: str, idx_col: str, row: pd.Series) -> str:
    # side = 'tm' or 'px'
    inning = row.get("inning", None)
    pa = row.get("pa_of_inning", None)
    pitch = row.get("pitch_of_pa", None)
    pitch_no = row.get("pitch_no", None)
    idx = row.get(idx_col, "<unknown>")
    return f"[{side}] row_index={idx}, inning={inning}, PA={pa}, pitch_of_PA={pitch}, pitch_no={pitch_no}"

def find_anomalies(tm: pd.DataFrame, px: pd.DataFrame) -> List[Dict]:
    """
    Return list of anomaly dicts with keys:
      side ('tm' or 'px'), row_index, inning, pa_of_inning, pitch_of_pa, pitch_no, issue, suggestion
    """
    issues = []

    # Helper to add issue
    def add(side, row_idx, inning, pa, pitch, pno, issue, suggestion):
        issues.append({
            "side": side,
            "row_index": int(row_idx) if pd.notna(row_idx) else None,
            "inning": int(inning) if pd.notna(inning) else None,
            "pa_of_inning": int(pa) if pd.notna(pa) else None,
            "pitch_of_pa": int(pitch) if pd.notna(pitch) else None,
            "pitch_no": int(pno) if pd.notna(pno) else None,
            "issue": issue,
            "suggestion": suggestion
        })

    # 1) Missing batter names
    if not tm.empty:
        for _, r in tm.iterrows():
            if (r.get("batter_name", "") == ""):
                add("tm", r.get("trackman_idx"), r.get("inning"), r.get("pa_of_inning"), r.get("pitch_of_pa"), r.get("pitch_no"),
                    "Missing batter name in Trackman row",
                    "Add 'Batter' column or populate batter names in Trackman export.")
    if not px.empty:
        for _, r in px.iterrows():
            if (r.get("batter_name", "") == ""):
                add("pxp", r.get("pxp_idx"), r.get("inning"), r.get("pa_of_inning"), r.get("pitch_of_pa"), r.get("pitch_no"),
                    "Missing batter name in PxP row",
                    "Fill 'Batter' column in the play-by-play CSV.")

    # 2) Duplicate (inning, pa_of_inning, pitch_of_pa) within same file
    def check_dups(df, side, idx_col):
        if df.empty:
            return
        # consider rows with non-null pa/pitch
        mask = df["pa_of_inning"].notna() & df["pitch_of_pa"].notna()
        grouped = df[mask].groupby(["pa_id", "pitch_of_pa"])
        for name, g in grouped:
            if len(g) > 1:
                for _, r in g.iterrows():
                    add(side, r.get(idx_col), r.get("inning"), r.get("pa_of_inning"), r.get("pitch_of_pa"), r.get("pitch_no"),
                        "Duplicate PA + pitch_of_PA in same file",
                        "Remove duplicate rows or correct PA/Pitch numbering for these rows.")
    check_dups(tm, "tm", "trackman_idx")
    check_dups(px, "px", "pxp_idx")

    # 3) Non-consecutive pitch_of_pa within same PA (gaps)
    def check_gaps(df, side, idx_col):
        if df.empty:
            return
        for pa, g in df.groupby("pa_id"):
            # ignore if only one row
            seq = g["pitch_of_pa"].dropna().astype(int).tolist()
            if not seq:
                continue
            seq_sorted = sorted(seq)
            # check gaps > 1
            for a, b in zip(seq_sorted, seq_sorted[1:]):
                if b - a > 1:
                    # find a representative row to report (first missing gap)
                    # report the row just before gap
                    row_before = g[g["pitch_of_pa"] == a].iloc[0]
                    add(side, row_before.get(idx_col), row_before.get("inning"), row_before.get("pa_of_inning"), row_before.get("pitch_of_pa"), row_before.get("pitch_no"),
                        f"Gap in pitch_of_PA sequence in PA '{pa}': found {a} then {b}",
                        "Check missing pitch rows or correct PitchofPA so sequence increments by 1.")
    check_gaps(tm, "tm", "trackman_idx")
    check_gaps(px, "px", "pxp_idx")

    # 4) Duplicate global pitch_no inside a file
    def check_dup_pitchno(df, side, idx_col):
        if df.empty:
            return
        if "pitch_no" in df.columns:
            dupes = df[df["pitch_no"].duplicated(keep=False) & df["pitch_no"].notna()]
            for _, r in dupes.iterrows():
                add(side, r.get(idx_col), r.get("inning"), r.get("pa_of_inning"), r.get("pitch_of_pa"), r.get("pitch_no"),
                    "Duplicate global pitch number (PitchNo) in file",
                    "Ensure PitchNo is unique per game; remove duplicates or fix numbering.")
    check_dup_pitchno(tm, "tm", "trackman_idx")
    check_dup_pitchno(px, "px", "pxp_idx")

    # 5) Batter mismatch between tm and px for same PA (majority mismatch)
    if (not tm.empty) and (not px.empty):
        # build map pa -> set of batter names
        pa_batters_tm = tm.groupby("pa_id")["batter_name"].agg(lambda s: set([x for x in s if x])).to_dict()
        pa_batters_px = px.groupby("pa_id")["batter_name"].agg(lambda s: set([x for x in s if x])).to_dict()
        common_pas = set(pa_batters_tm.keys()) & set(pa_batters_px.keys())
        for pa in common_pas:
            tset = pa_batters_tm.get(pa, set())
            pset = pa_batters_px.get(pa, set())
            # if both sets non-empty and disjoint (or little intersection), flag
            if tset and pset and len(tset.intersection(pset)) == 0:
                # pick one representative row from each side to report
                tm_row = tm[tm["pa_id"] == pa].iloc[0]
                px_row = px[px["pa_id"] == pa].iloc[0]
                add("tm", tm_row.get("trackman_idx"), tm_row.get("inning"), tm_row.get("pa_of_inning"), tm_row.get("pitch_of_pa"), tm_row.get("pitch_no"),
                    f"Batter mismatch for PA {pa}: Trackman batters={sorted(list(tset))}; PxP batters={sorted(list(pset))}",
                    "Standardize batter name formatting (Last, First) and ensure both files use the same roster names.")
                add("pxp", px_row.get("pxp_idx"), px_row.get("inning"), px_row.get("pa_of_inning"), px_row.get("pitch_of_pa"), px_row.get("pitch_no"),
                    f"Batter mismatch for PA {pa}: Trackman batters={sorted(list(tset))}; PxP batters={sorted(list(pset))}",
                    "Standardize batter name formatting (Last, First) and ensure both files use the same roster names.")

    return issues

# -----------------------
# Alignment logic (unchanged)
# -----------------------
def align_by_pa(df_tm: pd.DataFrame, df_px: pd.DataFrame) -> pd.DataFrame:
    mappings = []
    pa_ids = sorted(set(df_tm["pa_id"].unique()) | set(df_px["pa_id"].unique()))

    for pa in pa_ids:
        tm_pa = df_tm[df_tm["pa_id"] == pa].copy().reset_index(drop=True)
        px_pa = df_px[df_px["pa_id"] == pa].copy().reset_index(drop=True)

        if tm_pa.empty and px_pa.empty:
            continue

        if (not tm_pa.empty) and (not px_pa.empty):
            px_by_pitch = {int(r["pitch_of_pa"]): int(r["pxp_idx"]) for _, r in px_pa.iterrows() if not pd.isna(r["pitch_of_pa"])}
            tm_by_pitch = {int(r["pitch_of_pa"]): int(r["trackman_idx"]) for _, r in tm_pa.iterrows() if not pd.isna(r["pitch_of_pa"])}

            for pitch_no, tm_idx in tm_by_pitch.items():
                if pitch_no in px_by_pitch:
                    mappings.append({
                        "trackman_idx": int(tm_idx),
                        "pxp_idx": int(px_by_pitch[pitch_no]),
                        "match_type": "exact_pa_pitch_of_pa",
                        "confidence": 1.0,
                        "comment": "exact pa/pitch_of_pa"
                    })

            matched_tm = {m["trackman_idx"] for m in mappings if m["match_type"] == "exact_pa_pitch_of_pa"}
            matched_px = {m["pxp_idx"] for m in mappings if m["match_type"] == "exact_pa_pitch_of_pa"}

            px_global_map = {}
            if "PitchNo" in px_pa.columns:
                px_global_map = {int(r["PitchNo"]): int(r["pxp_idx"]) for _, r in px_pa.iterrows() if not pd.isna(r.get("PitchNo"))}

            for _, tmrow in tm_pa.iterrows():
                tm_idx = int(tmrow["trackman_idx"])
                if tm_idx in matched_tm:
                    continue
                if not pd.isna(tmrow.get("pitch_no")) and tmrow.get("pitch_no") in px_global_map:
                    px_idx = int(px_global_map[int(tmrow.get("pitch_no"))])
                    if px_idx not in matched_px:
                        mappings.append({
                            "trackman_idx": tm_idx,
                            "pxp_idx": px_idx,
                            "match_type": "exact_pitch_no",
                            "confidence": 0.95,
                            "comment": "matched by global pitch number"
                        })
                        matched_tm.add(tm_idx)
                        matched_px.add(px_idx)

            rem_tm = [int(r["trackman_idx"]) for _, r in tm_pa.iterrows() if int(r["trackman_idx"]) not in matched_tm]
            rem_px = [int(r["pxp_idx"]) for _, r in px_pa.iterrows() if int(r["pxp_idx"]) not in matched_px]

            min_len = min(len(rem_tm), len(rem_px))
            for i in range(min_len):
                mappings.append({
                    "trackman_idx": rem_tm[i],
                    "pxp_idx": rem_px[i],
                    "match_type": "seq_alignment",
                    "confidence": 0.7,
                    "comment": f"seq pos {i+1} within PA {pa}"
                })

            for t in rem_tm[min_len:]:
                mappings.append({
                    "trackman_idx": t,
                    "pxp_idx": np.nan,
                    "match_type": "unmatched_tm_in_pa",
                    "confidence": 0.0,
                    "comment": "no pxp row to match"
                })
            for p in rem_px[min_len:]:
                mappings.append({
                    "trackman_idx": np.nan,
                    "pxp_idx": p,
                    "match_type": "unmatched_px_in_pa",
                    "confidence": 0.0,
                    "comment": "no trackman row to match"
                })
        else:
            if not tm_pa.empty:
                for _, r in tm_pa.iterrows():
                    mappings.append({
                        "trackman_idx": int(r["trackman_idx"]),
                        "pxp_idx": np.nan,
                        "match_type": "unmatched_tm_pa_only",
                        "confidence": 0.0,
                        "comment": "pxp missing for this PA"
                    })
            if not px_pa.empty:
                for _, r in px_pa.iterrows():
                    mappings.append({
                        "trackman_idx": np.nan,
                        "pxp_idx": int(r["pxp_idx"]),
                        "match_type": "unmatched_px_pa_only",
                        "confidence": 0.0,
                        "comment": "trackman missing for this PA"
                    })

    mapping_df = pd.DataFrame(mappings, columns=["trackman_idx", "pxp_idx", "match_type", "confidence", "comment"])
    return mapping_df

# -----------------------
# High-level align function with anomaly check
# -----------------------
def align_game(trackman_df: Optional[pd.DataFrame], pxp_df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    Align Trackman (or None) with PxP. If anomalies are found, raises ValueError with details.
    Otherwise returns merged debug DataFrame (mapping joined with both sides).
    """
    tm = preprocess_trackman(trackman_df, game_id) if (trackman_df is not None and not trackman_df.empty) else pd.DataFrame(columns=[])
    px = preprocess_pxp(pxp_df, game_id)

    # ---------------------------------------------------------
    # CHECK FOR PA_ID MISMATCHES (ignore empty rows)
    # ---------------------------------------------------------

    def nonempty_pa_ids(df: pd.DataFrame) -> set:
        """
        Return a set of PA IDs from rows that are not 'empty'.
        A row is considered empty if all columns *except 'pa_id'* are NaN or empty strings.
        """
        if df is None or df.empty:
            return set()

        # Columns to check for emptiness (all except 'pa_id')
        cols_to_check = [c for c in df.columns if c != "pa_id"]

        # Keep rows where at least one of the other columns has a value
        df2 = df[df[cols_to_check].apply(lambda row: row.notna() & (row.astype(str).str.strip() != "")).any(axis=1)]

        # Return valid PA IDs
        return set(df2["pa_id"].dropna().astype(str))

    tm_ids  = nonempty_pa_ids(tm)
    px_ids = nonempty_pa_ids(px)

    extra_in_px = px_ids - tm_ids
    extra_in_tm  = tm_ids - px_ids

    if extra_in_px or extra_in_tm:
        msgs = []

        if extra_in_px:
            msgs.append("Extra PXP rows with unmatched PA IDs:\n  " +
                        "\n  ".join(sorted(extra_in_px)))
        if extra_in_tm:
            msgs.append("Extra TrackMan rows with unmatched PA IDs:\n  " +
                        "\n  ".join(sorted(extra_in_tm)))

        full_msg = (
            "\nPA mismatch detected before merge.\n\n" +
            "\n\n".join(msgs) +
            "\n\nTo fix: ensure each PA exists in BOTH datasets, "
            "or remove/adjust the extra rows."
        )

        raise ValueError(full_msg)

    # Run anomaly detection BEFORE aligning
    anomalies = find_anomalies(tm, px)
    if anomalies:
        # Format a readable error message
        lines = ["Alignment aborted: anomalies detected in input files.", ""]
        for a in anomalies:
            side = a["side"]
            idx = a["row_index"]
            inning = a["inning"]
            pa = a["pa_of_inning"]
            pitch = a["pitch_of_pa"]
            pno = a["pitch_no"]
            issue = a["issue"]
            suggestion = a["suggestion"]
            lines.append(f"- {side.upper()} row {idx} | inning={inning} pa={pa} pitch_of_pa={pitch} pitch_no={pno} :: {issue}")
            lines.append(f"    Suggestion: {suggestion}")
        # Join and raise
        raise ValueError("\n".join(lines))

    # No anomalies -> proceed
    if tm.empty:
        # If there's no trackman, return mapping marking all px rows as unmatched
        mapping = []
        for _, r in px.iterrows():
            mapping.append({
                "trackman_idx": np.nan,
                "pxp_idx": int(r["pxp_idx"]),
                "match_type": "unmatched_px_pa_only",
                "confidence": 0.0,
                "comment": "no trackman file"
            })
        mapping_df = pd.DataFrame(mapping, columns=["trackman_idx", "pxp_idx", "match_type", "confidence", "comment"])
        merged = mapping_df.merge(px.add_suffix("_px"), left_on="pxp_idx", right_on="pxp_idx_px", how="left")
        return merged

    mapping_df = align_by_pa(tm, px)

    # merge for debugging
    mapped = mapping_df.copy()
    mapped = mapped.merge(tm.add_suffix("_tm"), left_on="trackman_idx", right_on="trackman_idx_tm", how="left")
    mapped = mapped.merge(px.add_suffix("_px"), left_on="pxp_idx", right_on="pxp_idx_px", how="left")
    return mapped
