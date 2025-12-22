import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lineup Tracker", layout="wide")

PLAYER_COLS = ["P1", "P2", "P3", "P4", "P5"]

# Offensive columns (from your instructions + needed for Dashboard)
OFF_COLS = {
    "PtsFor": "PtsFor",
    "TOV": "TOV",
    "OReb": "OReb",
    "OppDefReb": "OppDefReb",
    "FTA For": "FTA For",
    "RimAtt": "RimAtt",
    "ThreePA": "ThreePA",
    "NonPaint3": "NonPaint3",
    "Action3": "Action3",
    "Transition3": "Transition3",
    "Paint3": "Paint3",
    "OffPoss": "OffPoss",
}

# Defensive columns (from your instructions + needed for Dashboard)
DEF_COLS = {
    "PtsAg": "PtsAg",
    "Non rim ag": "Non rim ag",  # "non-rim against"
    "TOV Forced": "TOV Forced",
    "OReb Ag": "OReb Ag",
    "Def Reb": "Def Reb",
    "FTA Ag": "FTA Ag",
    "RimAtt Ag": "RimAtt Ag",
    "ThreePA Ag": "ThreePA Ag",
    "DefPoss": "DefPoss",
}

# Dashboard metrics requested:
# ORTG=100*PtsFor/OffPoss, DRTG=100*PtsAg/DefPoss, NRTG=ORTG-DRTG,
# TovR=TOVfor/OffPoss, RimR=RimAttFor/OffPoss, 3PR=ThreePA_For/OffPoss.


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    # Make sure required columns exist
    needed = ["Side"] + PLAYER_COLS + list(OFF_COLS.values()) + list(DEF_COLS.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize player names (string, trimmed, uppercase)
    for c in PLAYER_COLS:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.upper()
        )

    # Numeric coercion for stat columns
    stat_cols = sorted(set(list(OFF_COLS.values()) + list(DEF_COLS.values())))
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


@st.cache_data(show_spinner=False)
def build_lineup_table(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Creates aggregated table for k-man lineups (k in {2,3,5})
    using all combinations within each 5-player row.
    """
    rows = df[PLAYER_COLS + list(OFF_COLS.values()) + list(DEF_COLS.values())].copy()

    agg = {}  # key: tuple(players), value: dict sums

    off_keys = list(OFF_COLS.values())
    def_keys = list(DEF_COLS.values())
    all_sum_keys = sorted(set(off_keys + def_keys))

    for _, r in rows.iterrows():
        players = [r[c] for c in PLAYER_COLS]
        players = sorted(players)  # order doesn't matter

        for combo in itertools.combinations(players, k):
            if combo not in agg:
                agg[combo] = {col: 0.0 for col in all_sum_keys}
            for col in all_sum_keys:
                agg[combo][col] += float(r[col])

    out = []
    for combo, sums in agg.items():
        rec = {"Lineup": " / ".join(combo)}
        rec.update(sums)
        out.append(rec)

    out_df = pd.DataFrame(out)

    # Helpful totals
    out_df["TotalPoss"] = out_df["OffPoss"] + out_df["DefPoss"]

    return out_df


def add_dashboard_metrics(df: pd.DataFrame, per100: bool) -> pd.DataFrame:
    mult = 100.0 if per100 else 1.0

    df = df.copy()

    # Safe division
    df["ORTG"] = np.where(df["OffPoss"] > 0, mult * (df["PtsFor"] / df["OffPoss"]), np.nan)
    df["DRTG"] = np.where(df["DefPoss"] > 0, mult * (df["PtsAg"] / df["DefPoss"]), np.nan)
    df["NRTG"] = df["ORTG"] - df["DRTG"]

    df["TovR"] = np.where(df["OffPoss"] > 0, mult * (df["TOV"] / df["OffPoss"]), np.nan)
    df["RimR"] = np.where(df["OffPoss"] > 0, mult * (df["RimAtt"] / df["OffPoss"]), np.nan)
    df["3PR"]  = np.where(df["OffPoss"] > 0, mult * (df["ThreePA"] / df["OffPoss"]), np.nan)

    return df


def add_offense_rates(df: pd.DataFrame, per100: bool) -> pd.DataFrame:
    mult = 100.0 if per100 else 1.0
    df = df.copy()

    # Rates per OffPoss for requested offensive stats
    rate_map = {
        "PtsFor": "PtsFor_R",
        "TOV": "TOV_R",
        "OReb": "OReb_R",
        "OppDefReb": "OppDefReb_R",
        "FTA For": "FTAFor_R",
        "RimAtt": "RimAtt_R",
        "ThreePA": "ThreePA_R",
        "NonPaint3": "NonPaint3_R",
        "Action3": "Action3_R",
        "Transition3": "Transition3_R",
        "Paint3": "Paint3_R",
    }

    for raw, new in rate_map.items():
        df[new] = np.where(df["OffPoss"] > 0, mult * (df[raw] / df["OffPoss"]), np.nan)

    return df


def add_defense_rates(df: pd.DataFrame, per100: bool) -> pd.DataFrame:
    mult = 100.0 if per100 else 1.0
    df = df.copy()

    # Rates per DefPoss for requested defensive stats
    rate_map = {
        "PtsAg": "PtsAg_R",
        "Non rim ag": "NonRimAg_R",
        "TOV Forced": "TOVForced_R",
        "OReb Ag": "ORebAg_R",
        "Def Reb": "DefReb_R",
        "FTA Ag": "FTAAg_R",
        "RimAtt Ag": "RimAttAg_R",
        "ThreePA Ag": "ThreePAAg_R",
    }

    for raw, new in rate_map.items():
        df[new] = np.where(df["DefPoss"] > 0, mult * (df[raw] / df["DefPoss"]), np.nan)

    return df


def one_decimal_display(df: pd.DataFrame) -> pd.DataFrame:
    # Round ALL numeric columns to 1 decimal for display
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(1)
    return df


st.title("Lineup Tracker (2-Man / 3-Man / 5-Man)")

uploaded = st.file_uploader("Upload your Lineup Tracker Excel (.xlsx)", type=["xlsx"])

with st.sidebar:
    st.header("Controls")
    per100 = st.toggle("Show as per 100 possessions", value=True)
    min_poss = st.number_input("Minimum Total Possessions threshold", min_value=0, value=25, step=1)
    lineup_k = st.selectbox("Lineup size", options=[2, 3, 5], index=2)
    st.caption("TotalPoss = OffPoss + DefPoss")

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    poss_df = pd.read_excel(uploaded, sheet_name="Possessions", engine="openpyxl")
    poss_df = _clean_df(poss_df)
except Exception as e:
    st.error(f"Could not read sheet 'Possessions'. Error: {e}")
    st.stop()

base = build_lineup_table(poss_df, k=lineup_k)
base = base[base["TotalPoss"] >= float(min_poss)].copy()

# Build each view dataset
dash = add_dashboard_metrics(base, per100=per100)
offv = add_offense_rates(base, per100=per100)
defv = add_defense_rates(base, per100=per100)

tabs = st.tabs(["Dashboard", "Offense", "Defense"])

# ------------------- Dashboard -------------------
with tabs[0]:
    st.subheader(f"Dashboard — {lineup_k}-Man Lineups")

    sort_metric = st.selectbox(
        "Sort by",
        options=["NRTG", "ORTG", "DRTG", "TovR", "RimR", "3PR", "TotalPoss", "OffPoss", "DefPoss"],
        index=0,
    )
    ascending = st.checkbox("Ascending", value=False)

    show_cols = ["Lineup", "TotalPoss", "OffPoss", "DefPoss", "ORTG", "DRTG", "NRTG", "TovR", "RimR", "3PR"]
    view = dash[show_cols].sort_values(by=sort_metric, ascending=ascending, na_position="last")
    view = one_decimal_display(view)

    st.dataframe(view, use_container_width=True, hide_index=True)

# ------------------- Offense -------------------
with tabs[1]:
    st.subheader(f"Offense — {lineup_k}-Man Lineups")

    sort_metric = st.selectbox(
        "Sort offense table by",
        options=[
            "PtsFor_R", "TOV_R", "OReb_R", "OppDefReb_R", "FTAFor_R",
            "RimAtt_R", "ThreePA_R", "NonPaint3_R", "Action3_R",
            "Transition3_R", "Paint3_R", "OffPoss", "TotalPoss"
        ],
        index=0,
        key="off_sort",
    )
    ascending = st.checkbox("Ascending", value=False, key="off_asc")

    show_cols = [
        "Lineup", "TotalPoss", "OffPoss",
        "PtsFor_R", "TOV_R", "OReb_R", "OppDefReb_R", "FTAFor_R",
        "RimAtt_R", "ThreePA_R", "NonPaint3_R", "Action3_R",
        "Transition3_R", "Paint3_R"
    ]

    view = offv[show_cols].sort_values(by=sort_metric, ascending=ascending, na_position="last")
    view = one_decimal_display(view)

    st.dataframe(view, use_container_width=True, hide_index=True)

# ------------------- Defense -------------------
with tabs[2]:
    st.subheader(f"Defense — {lineup_k}-Man Lineups")

    sort_metric = st.selectbox(
        "Sort defense table by",
        options=[
            "PtsAg_R", "NonRimAg_R", "TOVForced_R", "ORebAg_R", "DefReb_R",
            "FTAAg_R", "RimAttAg_R", "ThreePAAg_R", "DefPoss", "TotalPoss"
        ],
        index=0,
        key="def_sort",
    )
    ascending = st.checkbox("Ascending", value=False, key="def_asc")

    show_cols = [
        "Lineup", "TotalPoss", "DefPoss",
        "PtsAg_R", "NonRimAg_R", "TOVForced_R", "ORebAg_R", "DefReb_R",
        "FTAAg_R", "RimAttAg_R", "ThreePAAg_R"
    ]

    view = defv[show_cols].sort_values(by=sort_metric, ascending=ascending, na_position="last")
    view = one_decimal_display(view)

    st.dataframe(view, use_container_width=True, hide_index=True)
