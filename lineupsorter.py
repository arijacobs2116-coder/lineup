import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Lineup Tracker", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def clean_name(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # normalize double spaces
    s = " ".join(s.split())
    return s.upper()

def lineup_key_from_row(row: pd.Series) -> str:
    players = [clean_name(row.get(f"P{i}", "")) for i in range(1, 6)]
    players = [p for p in players if p != ""]
    players_sorted = sorted(players)
    return " | ".join(players_sorted)

def add_lineup_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LINEUP_KEY"] = df.apply(lineup_key_from_row, axis=1)
    return df

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    return df

def summarize_side(df: pd.DataFrame, side_value: str, stat_cols: list[str], poss_col: str) -> pd.DataFrame:
    # Filter side
    d = df[df["Side"].astype(str).str.strip().str.upper() == side_value].copy()

    # possessions: use OffPoss/DefPoss if present; otherwise count rows
    if poss_col in d.columns:
        d[poss_col] = pd.to_numeric(d[poss_col], errors="coerce").fillna(0)
        d["POSS"] = d[poss_col]
    else:
        d["POSS"] = 1

    d = safe_numeric(d, stat_cols + ["POSS"])

    # Group
    g = (
        d.groupby("LINEUP_KEY", as_index=False)
        .agg({**{c: "sum" for c in stat_cols}, "POSS": "sum"})
    )

    # Per-100 possessions
    for c in stat_cols:
        g[f"{c}_per100"] = np.where(g["POSS"] > 0, (g[c] / g["POSS"]) * 100, 0)

    # Split out player display columns for readability (optional)
    # We'll keep the key and also provide 5 player columns:
    players = g["LINEUP_KEY"].str.split(r"\s*\|\s*", expand=True)
    for i in range(5):
        col = players[i] if i in players.columns else ""
        g[f"P{i+1}"] = col

    # Put columns in a nice order
    front = ["P1", "P2", "P3", "P4", "P5", "POSS"]
    keep = front + stat_cols + [f"{c}_per100" for c in stat_cols]
    keep = [c for c in keep if c in g.columns]
    return g[keep].sort_values("POSS", ascending=False)

# -----------------------------
# UI
# -----------------------------
st.title("Lineup Tracker (Upload → Group 5-Man Lineups → Rank)")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if not uploaded:
    st.info("Upload an .xlsx file to begin.")
    st.stop()

# Read the Possessions sheet
try:
    raw = pd.read_excel(uploaded, sheet_name="Possessions")
except Exception as e:
    st.error(f"Could not read sheet 'Possessions'. Error: {e}")
    st.stop()

needed_cols = ["Side", "P1", "P2", "P3", "P4", "P5"]
missing = [c for c in needed_cols if c not in raw.columns]
if missing:
    st.error(f"Missing required columns in 'Possessions' sheet: {missing}")
    st.stop()

raw = add_lineup_key(raw)

# Minimum possessions threshold control
st.subheader("Filters")
min_poss = st.number_input("Minimum possessions threshold", min_value=0, value=20, step=1)

tab_off, tab_def = st.tabs(["Offense", "Defense"])

# -----------------------------
# OFFENSE TAB
# -----------------------------
with tab_off:
    st.header("Offense")

    off_stat_cols = [
        "PtsFor",        # pts for
        "TOV",           # tov
        "OReb",          # offensive rebounds
        "OppDefReb",     # oppdefrebounds
        "FTA For",       # fta for
        "RimAtt",        # rim attempts
        "ThreePA",       # 3p attempts
        "NonPaint3",     # non-paint 3
        "Action3",       # action 3
        "Transition3",   # transition 3
        "Paint3",        # paint 3
    ]

    off_summary = summarize_side(raw, side_value="OFF", stat_cols=off_stat_cols, poss_col="OffPoss")
    off_summary = off_summary[off_summary["POSS"] >= min_poss].copy()

    st.write(f"Lineups meeting threshold: **{len(off_summary)}**")

    # Ranking controls
    metric_options = {c: f"{c}_per100" for c in off_stat_cols}
    chosen = st.selectbox("Rank by (per 100 possessions)", list(metric_options.keys()), index=0)

    sort_col = metric_options[chosen]
    off_summary = off_summary.sort_values(sort_col, ascending=False)

    # Show
    st.dataframe(off_summary, use_container_width=True)

# -----------------------------
# DEFENSE TAB
# -----------------------------
with tab_def:
    st.header("Defense")

    def_stat_cols = [
        "PtsAg",         # pts against
        "Non rim ag",    # non-rim against
        "TOV Forced",    # tov forced
        "OReb Ag",       # Oreb against
        "Def Reb",       # def reb
        "FTA Ag",        # FTA against
        "RimAtt Ag",     # rim attempt against
        "ThreePA Ag",    # three point attempt against
    ]

    def_summary = summarize_side(raw, side_value="DEF", stat_cols=def_stat_cols, poss_col="DefPoss")
    def_summary = def_summary[def_summary["POSS"] >= min_poss].copy()

    st.write(f"Lineups meeting threshold: **{len(def_summary)}**")

    metric_options = {c: f"{c}_per100" for c in def_stat_cols}
    chosen = st.selectbox("Rank by (per 100 possessions)", list(metric_options.keys()), index=0)

    sort_col = metric_options[chosen]

    # Note: For defense, "better" might mean LOWER PtsAg_per100 etc.
    # We'll add a toggle so you can flip direction easily.
    ascending = st.toggle("Sort ascending (useful for 'against' stats like PtsAg)", value=True)

    def_summary = def_summary.sort_values(sort_col, ascending=ascending)

    st.dataframe(def_summary, use_container_width=True)
