import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Lineup Tracker", layout="wide")

# ----------------------------
# Config
# ----------------------------
SHEET_NAME = "Possessions"  # your workbook uses this sheet

OFFENSE_STATS = {
    "Pts For": "PtsFor",
    "TOV": "TOV",
    "Off Reb": "OReb",
    "Opp Def Reb": "OppDefReb",
    "FTA For": "FTA For",
    "Rim Attempts": "RimAtt",
    "3P Attempts": "ThreePA",
    "Non-Paint 3": "NonPaint3",
    "Action 3": "Action3",
    "Transition 3": "Transition3",
    "Paint 3": "Paint3",
}

DEFENSE_STATS = {
    "Pts Against": "PtsAg",
    "Non-Rim Against": "Non rim ag",
    "TOV Forced": "TOV Forced",
    "OReb Against": "OReb Ag",
    "Def Reb": "Def Reb",
    "FTA Against": "FTA Ag",
    "Rim Attempts Against": "RimAtt Ag",
    "3P Attempts Against": "ThreePA Ag",
}

PLAYER_COLS = ["P1", "P2", "P3", "P4", "P5"]
SIDE_COL = "Side"
OFF_POSS_COL = "OffPoss"
DEF_POSS_COL = "DefPoss"


# ----------------------------
# Helpers
# ----------------------------
def normalize_player(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Keep original casing if you want; I normalize to upper for stable grouping
    return " ".join(s.split()).upper()


def make_lineup_key(row) -> str:
    players = [normalize_player(row.get(c)) for c in PLAYER_COLS]
    players = [p for p in players if p is not None]
    if len(players) != 5:
        return None
    players_sorted = sorted(players)
    return " | ".join(players_sorted)


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_possessions_sheet(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=SHEET_NAME, engine="openpyxl")

    # Drop completely empty rows
    df = df.dropna(how="all")

    # Normalize Side
    if SIDE_COL in df.columns:
        df[SIDE_COL] = df[SIDE_COL].astype(str).str.strip().str.title()

    # Build lineup key
    df["Lineup"] = df.apply(make_lineup_key, axis=1)
    df = df[df["Lineup"].notna()].copy()

    # Ensure numeric for all potential stat columns + poss columns
    numeric_cols = (
        list(OFFENSE_STATS.values())
        + list(DEFENSE_STATS.values())
        + [OFF_POSS_COL, DEF_POSS_COL]
    )
    df = safe_numeric(df, numeric_cols)

    return df


def aggregate_lineups(df: pd.DataFrame, side: str, stats_map: dict, poss_col: str) -> pd.DataFrame:
    """
    Aggregates totals by lineup for the given side.
    Returns a dataframe with totals + possessions + per100 columns.
    """
    d = df[df[SIDE_COL].str.upper() == side.upper()].copy()

    # Keep only needed columns that exist
    stat_cols = [c for c in stats_map.values() if c in d.columns]
    keep_cols = ["Lineup", poss_col] + stat_cols
    d = d[keep_cols].copy()

    # Fill NaNs with 0 for aggregation (common for side-specific stats)
    for c in stat_cols + [poss_col]:
        if c in d.columns:
            d[c] = d[c].fillna(0)

    grouped = d.groupby("Lineup", as_index=False).sum(numeric_only=True)

    # Build per100 columns
    poss = grouped[poss_col].replace(0, np.nan)
    for label, col in stats_map.items():
        if col not in grouped.columns:
            continue
        grouped[f"{label} (per100)"] = (grouped[col] / poss) * 100

    # Friendly display columns (raw)
    rename_raw = {v: k for k, v in stats_map.items() if v in grouped.columns}
    grouped = grouped.rename(columns=rename_raw)

    # Move possessions up front
    grouped = grouped.rename(columns={poss_col: "Possessions"})
    cols_front = ["Lineup", "Possessions"]
    other_cols = [c for c in grouped.columns if c not in cols_front]
    grouped = grouped[cols_front + other_cols]

    # Sort stable default: possessions desc
    grouped = grouped.sort_values(["Possessions"], ascending=False).reset_index(drop=True)
    return grouped


def build_display_table(agg: pd.DataFrame, per100: bool, stats_map: dict) -> pd.DataFrame:
    """
    Returns the table users see (either raw or per100),
    with numeric rounding for readability.
    """
    if agg.empty:
        return agg

    base_cols = ["Lineup", "Possessions"]

    if per100:
        per100_cols = [f"{k} (per100)" for k in stats_map.keys() if f"{k} (per100)" in agg.columns]
        out = agg[base_cols + per100_cols].copy()
        # Round per100 stats
        for c in per100_cols:
            out[c] = out[c].astype(float).round(2)
        return out
    else:
        raw_cols = [k for k in stats_map.keys() if k in agg.columns]
        out = agg[base_cols + raw_cols].copy()
        # Round raw (mostly ints anyway)
        for c in raw_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if c.lower().startswith("pts") or "Attempts" in c or "Reb" in c or "TOV" in c or "FTA" in c or "3" in c:
                out[c] = out[c].fillna(0).round(0).astype(int)
        return out


# ----------------------------
# UI
# ----------------------------
st.title("5-Man Lineup Rankings")

uploaded = st.file_uploader("Upload Lineup Tracker Excel (.xlsx)", type=["xlsx"])

if not uploaded:
    st.info("Upload your Excel file to begin.")
    st.stop()

try:
    df = load_possessions_sheet(uploaded)
except Exception as e:
    st.error(f"Could not read sheet '{SHEET_NAME}'. {e}")
    st.stop()

# Global switch: per100 or not
per100_toggle = st.toggle("Show stats per 100 possessions", value=True)

tab_off, tab_def = st.tabs(["Offense", "Defense"])

with tab_off:
    st.subheader("Offense")

    min_poss_off = st.number_input(
        "Minimum Off Possessions",
        min_value=0,
        value=20,
        step=5,
        help="Lineups with fewer OffPoss than this will be hidden.",
    )

    agg_off = aggregate_lineups(df, side="Off", stats_map=OFFENSE_STATS, poss_col=OFF_POSS_COL)
    agg_off = agg_off[agg_off["Possessions"] >= min_poss_off].copy()

    display_off = build_display_table(agg_off, per100=per100_toggle, stats_map=OFFENSE_STATS)

    if display_off.empty:
        st.warning("No offensive lineups match your possessions threshold.")
    else:
        sort_choices = [c for c in display_off.columns if c not in ["Lineup"]]
        default_sort = "Pts For (per100)" if per100_toggle and "Pts For (per100)" in sort_choices else (
            "Pts For" if "Pts For" in sort_choices else sort_choices[0]
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            sort_col = st.selectbox("Sort / Rank by", sort_choices, index=sort_choices.index(default_sort))
        with c2:
            descending = st.toggle("Descending", value=True)

        display_off = display_off.sort_values(sort_col, ascending=not descending).reset_index(drop=True)
        display_off.insert(0, "Rank", np.arange(1, len(display_off) + 1))

        st.dataframe(display_off, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Offense Table (CSV)",
            data=display_off.to_csv(index=False).encode("utf-8"),
            file_name="lineups_offense.csv",
            mime="text/csv",
        )

with tab_def:
    st.subheader("Defense")

    min_poss_def = st.number_input(
        "Minimum Def Possessions",
        min_value=0,
        value=20,
        step=5,
        help="Lineups with fewer DefPoss than this will be hidden.",
    )

    agg_def = aggregate_lineups(df, side="Def", stats_map=DEFENSE_STATS, poss_col=DEF_POSS_COL)
    agg_def = agg_def[agg_def["Possessions"] >= min_poss_def].copy()

    display_def = build_display_table(agg_def, per100=per100_toggle, stats_map=DEFENSE_STATS)

    if display_def.empty:
        st.warning("No defensive lineups match your possessions threshold.")
    else:
        sort_choices = [c for c in display_def.columns if c not in ["Lineup"]]
        default_sort = "Pts Against (per100)" if per100_toggle and "Pts Against (per100)" in sort_choices else (
            "Pts Against" if "Pts Against" in sort_choices else sort_choices[0]
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            sort_col = st.selectbox("Sort / Rank by", sort_choices, index=sort_choices.index(default_sort))
        with c2:
            descending = st.toggle("Descending", value=False, help="For defense, you usually want ascending for points against.")

        display_def = display_def.sort_values(sort_col, ascending=not descending).reset_index(drop=True)
        display_def.insert(0, "Rank", np.arange(1, len(display_def) + 1))

        st.dataframe(display_def, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Defense Table (CSV)",
            data=display_def.to_csv(index=False).encode("utf-8"),
            file_name="lineups_defense.csv",
            mime="text/csv",
        )
