import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lineup Tracker", layout="wide")

PLAYER_COLS = ["P1", "P2", "P3", "P4", "P5"]

# -----------------------------
# CLEAN + LOAD
# -----------------------------
def clean_df(df):
    df = df.dropna(how="all")

    for c in PLAYER_COLS:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.upper()
        )

    numeric_cols = [
        "PtsFor","PtsAg","TOV","RimAtt","ThreePA",
        "OffPoss","DefPoss"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# -----------------------------
# BUILD LINEUPS (1–5 MAN)
# -----------------------------
@st.cache_data(show_spinner=False)
def build_lineups(df, k):
    agg = {}

    for _, r in df.iterrows():
        players = sorted([r[p] for p in PLAYER_COLS])

        for combo in itertools.combinations(players, k):
            if combo not in agg:
                agg[combo] = {
                    "PtsFor": 0.0,
                    "PtsAg": 0.0,
                    "TOV": 0.0,
                    "RimAtt": 0.0,
                    "ThreePA": 0.0,
                    "OffPoss": 0.0,
                    "DefPoss": 0.0,
                }

            agg[combo]["PtsFor"] += r["PtsFor"]
            agg[combo]["PtsAg"] += r["PtsAg"]
            agg[combo]["TOV"] += r["TOV"]
            agg[combo]["RimAtt"] += r["RimAtt"]
            agg[combo]["ThreePA"] += r["ThreePA"]
            agg[combo]["OffPoss"] += r["OffPoss"]
            agg[combo]["DefPoss"] += r["DefPoss"]

    rows = []
    for combo, vals in agg.items():
        row = {"Lineup": " / ".join(combo)}
        row.update(vals)
        row["TotalPoss"] = vals["OffPoss"] + vals["DefPoss"]
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# DASHBOARD METRICS
# -----------------------------
def dashboard_metrics(df, per100):
    mult = 100 if per100 else 1
    df = df.copy()

    df["ORTG"] = np.where(df["OffPoss"] > 0, mult * df["PtsFor"] / df["OffPoss"], np.nan)
    df["DRTG"] = np.where(df["DefPoss"] > 0, mult * df["PtsAg"] / df["DefPoss"], np.nan)
    df["NRTG"] = df["ORTG"] - df["DRTG"]
    df["TovR"] = np.where(df["OffPoss"] > 0, mult * df["TOV"] / df["OffPoss"], np.nan)
    df["RimR"] = np.where(df["OffPoss"] > 0, mult * df["RimAtt"] / df["OffPoss"], np.nan)
    df["3PR"]  = np.where(df["OffPoss"] > 0, mult * df["ThreePA"] / df["OffPoss"], np.nan)

    return df.round(1)


# -----------------------------
# UI
# -----------------------------
st.title("Lineup Dashboard (1–5 Man)")

uploaded = st.file_uploader("Upload Lineup Tracker Excel", type=["xlsx"])
if not uploaded:
    st.stop()

df = pd.read_excel(uploaded, sheet_name="Possessions", engine="openpyxl")
df = clean_df(df)

with st.sidebar:
    lineup_k = st.selectbox("Lineup size", options=[1, 2, 3, 4, 5], index=4)
    per100 = st.toggle("Per 100 possessions", value=True)
    min_poss = st.number_input("Minimum Total Possessions", 0, value=25, step=1)

base = build_lineups(df, lineup_k)
base = base[base["TotalPoss"] >= min_poss].copy()

dash = dashboard_metrics(base, per100)

tabs = st.tabs(["Dashboard", "Offense", "Defense"])

# -----------------------------
# DASHBOARD TAB
# -----------------------------
with tabs[0]:
    st.subheader(f"{lineup_k}-Man Lineups — Dashboard")

    sort_by = st.selectbox(
        "Sort by",
        ["NRTG", "ORTG", "DRTG", "TovR", "RimR", "3PR", "TotalPoss"],
    )

    view = dash[
        ["Lineup","TotalPoss","OffPoss","DefPoss","ORTG","DRTG","NRTG","TovR","RimR","3PR"]
    ].sort_values(sort_by, ascending=False)

    st.dataframe(view, use_container_width=True, hide_index=True)

# -----------------------------
# OFFENSE TAB
# -----------------------------
with tabs[1]:
    st.subheader(f"{lineup_k}-Man Lineups — Offense")

    off = dash[
        ["Lineup","OffPoss","PtsFor","TOV","RimAtt","ThreePA"]
    ].round(1)

    st.dataframe(off, use_container_width=True, hide_index=True)

# -----------------------------
# DEFENSE TAB
# -----------------------------
with tabs[2]:
    st.subheader(f"{lineup_k}-Man Lineups — Defense")

    deff = dash[
        ["Lineup","DefPoss","PtsAg"]
    ].round(1)

    st.dataframe(deff, use_container_width=True, hide_index=True)
