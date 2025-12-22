import itertools
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import bcrypt

###############################################################################
# Config / storage
###############################################################################
st.set_page_config(page_title="Lineup Tracker", layout="wide")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "users.sqlite3"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Auth helpers (SQLite + bcrypt)
###############################################################################
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            pw_hash  BLOB NOT NULL
        );
        """
    )
    return conn

def _valid_username(u: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,24}", u or ""))

def _hash_pw(password: str) -> bytes:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt)

def _check_pw(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def create_user(username: str, password: str) -> Tuple[bool, str]:
    if not _valid_username(username):
        return False, "Username must be 3–24 characters: letters, numbers, underscore."
    if password is None or len(password) < 8:
        return False, "Password must be at least 8 characters."

    conn = _get_conn()
    try:
        cur = conn.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cur.fetchone() is not None:
            return False, "That username already exists."

        conn.execute(
            "INSERT INTO users(username, pw_hash) VALUES(?, ?)",
            (username, _hash_pw(password)),
        )
        conn.commit()
        return True, "Account created. You can log in now."
    finally:
        conn.close()

def authenticate(username: str, password: str) -> Tuple[bool, str]:
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT pw_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            return False, "Invalid username or password."
        pw_hash = row[0]
        if isinstance(pw_hash, memoryview):  # sqlite can return memoryview
            pw_hash = pw_hash.tobytes()
        ok = _check_pw(password, pw_hash)
        return (ok, "Logged in." if ok else "Invalid username or password.")
    finally:
        conn.close()

def user_upload_path(username: str) -> Path:
    safe = username
    return UPLOAD_DIR / f"{safe}.xlsx"

###############################################################################
# Data logic
###############################################################################
REQUIRED_COLUMNS = [
    "Side", "P1", "P2", "P3", "P4", "P5",
    "PtsFor", "PtsAg",
    "TOV", "TOV Forced",
    "OReb", "OReb Ag",
    "Def Reb", "OppDefReb",
    "FTA For", "FTA Ag",
    "RimAtt", "RimAtt Ag",
    "ThreePA", "ThreePA Ag",
    "Non rim", "Non rim ag",
    "NonPaint3", "Action3", "Transition3", "Paint3",
    "OffPoss", "DefPoss",
]

OFFENSE_COLS = {
    "Pts For": "PtsFor",
    "TOV": "TOV",
    "OReb": "OReb",
    "OppDefReb": "OppDefReb",
    "FTA For": "FTA For",
    "Rim Att": "RimAtt",
    "3PA": "ThreePA",
    "Non-Paint 3": "NonPaint3",
    "Action 3": "Action3",
    "Transition 3": "Transition3",
    "Paint 3": "Paint3",
}

DEFENSE_COLS = {
    "Pts Against": "PtsAg",
    "Non-Rim Against": "Non rim ag",
    "TOV Forced": "TOV Forced",
    "OReb Against": "OReb Ag",
    "Def Reb": "Def Reb",
    "FTA Against": "FTA Ag",
    "Rim Att Against": "RimAtt Ag",
    "3PA Against": "ThreePA Ag",
}

@st.cache_data(show_spinner=False)
def load_possessions_from_xlsx(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Possessions")
    # Drop any completely unnamed trailing columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    # Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    # Basic validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Coerce numeric columns
    numeric_cols = [
        "PtsFor","PtsAg","TOV","TOV Forced","OReb","OReb Ag","Def Reb","OppDefReb",
        "FTA For","FTA Ag","RimAtt","RimAtt Ag","ThreePA","ThreePA Ag","Non rim","Non rim ag",
        "NonPaint3","Action3","Transition3","Paint3","OffPoss","DefPoss",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Clean players
    for p in ["P1","P2","P3","P4","P5"]:
        df[p] = df[p].astype(str).str.strip()
        df.loc[df[p].isin(["", "nan", "None", "NaN"]), p] = ""

    df["Side"] = df["Side"].astype(str).str.strip().str.title()  # Off / Def
    return df

def lineup_key(players: List[str], k: int) -> Tuple[str, ...]:
    clean = [p for p in players if p]
    clean = sorted(clean)
    if len(clean) < k:
        return tuple()
    return tuple(clean[:k])  # only used when k == len(clean) in 1-man? (we will use combos below)

def explode_to_combos(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    For each row, generate all combinations of size k from P1..P5, order-insensitive.
    """
    records = []
    player_cols = ["P1","P2","P3","P4","P5"]
    for idx, row in df.iterrows():
        players = [row[c] for c in player_cols if row[c]]
        if len(players) < k:
            continue
        for combo in itertools.combinations(sorted(players), k):
            rec = row.to_dict()
            rec["Lineup"] = " | ".join(combo)
            rec["LineupSize"] = k
            records.append(rec)
    if not records:
        return pd.DataFrame(columns=list(df.columns) + ["Lineup","LineupSize"])
    return pd.DataFrame.from_records(records)

def aggregate_lineups(df_poss: pd.DataFrame, k: int) -> pd.DataFrame:
    dfc = explode_to_combos(df_poss, k)
    if dfc.empty:
        return dfc

    # Aggregate sums
    sum_cols = [
        "PtsFor","PtsAg","TOV","TOV Forced","OReb","OReb Ag","Def Reb","OppDefReb",
        "FTA For","FTA Ag","RimAtt","RimAtt Ag","ThreePA","ThreePA Ag","Non rim","Non rim ag",
        "NonPaint3","Action3","Transition3","Paint3","OffPoss","DefPoss",
    ]
    g = dfc.groupby("Lineup", as_index=False)[sum_cols].sum()

    # Derived dashboard metrics
    # Protect divide-by-zero
    g["ORTG"] = (100.0 * g["PtsFor"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["DRTG"] = (100.0 * g["PtsAg"] / g["DefPoss"].replace(0, pd.NA)).fillna(0)
    g["NRTG"] = g["ORTG"] - g["DRTG"]
    g["TovR"] = (g["TOV"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["RimR"] = (g["RimAtt"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["3PR"]  = (g["ThreePA"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)

    # Convenience possessions
    g["TotalPoss"] = g["OffPoss"] + g["DefPoss"]
    return g

def apply_per100(df: pd.DataFrame, per100: bool) -> pd.DataFrame:
    out = df.copy()
    rate_cols = ["TovR","RimR","3PR"]
    if per100:
        out[rate_cols] = out[rate_cols] * 100.0
    return out

def round_1_decimal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(1)
    return out

###############################################################################
# UI helpers
###############################################################################
def stat_rank_table(df_lineups: pd.DataFrame, title: str, metric_map: Dict[str, str], poss_col: str):
    st.subheader(title)

    cols = st.columns([2, 1, 3, 3])
    with cols[0]:
        metric_label = st.selectbox("Rank by", list(metric_map.keys()), key=f"{title}_metric")
    with cols[1]:
        ascending = st.toggle("Low is better", value=False, key=f"{title}_asc")
    with cols[2]:
        min_poss = st.number_input(f"Minimum {poss_col}", min_value=0.0, value=25.0, step=1.0, key=f"{title}_minposs")
    with cols[3]:
        topn = st.number_input("Show top N", min_value=10, value=50, step=10, key=f"{title}_topn")

    col = metric_map[metric_label]

    view = df_lineups.copy()
    if poss_col in view.columns:
        view = view[view[poss_col] >= float(min_poss)]

    view = view.sort_values(col, ascending=bool(ascending)).head(int(topn))

    # Put lineup + possessions + chosen metric first
    front = ["Lineup", poss_col, col]
    other = [c for c in view.columns if c not in front]
    view = view[front + other]

    st.dataframe(round_1_decimal(view), use_container_width=True, hide_index=True)

###############################################################################
# App state: login + file persistence
###############################################################################
def logout():
    st.session_state.pop("username", None)
    st.session_state.pop("data_path", None)

def login_block():
    st.title("Lineup Tracker")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Log in")
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Log in", use_container_width=True):
            ok, msg = authenticate(u, p)
            if ok:
                st.session_state["username"] = u
                st.success(msg)
                # If they already uploaded a file before, load it automatically
                up = user_upload_path(u)
                if up.exists():
                    st.session_state["data_path"] = str(up)
            else:
                st.error(msg)

    with c2:
        st.subheader("Create account")
        nu = st.text_input("New username (3–24 chars; letters/numbers/_)", key="new_u")
        npw = st.text_input("New password (8+ chars)", type="password", key="new_p")
        if st.button("Create account", use_container_width=True):
            ok, msg = create_user(nu, npw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

###############################################################################
# Main
###############################################################################
if "username" not in st.session_state:
    login_block()
    st.stop()

username = st.session_state["username"]
st.sidebar.success(f"Logged in as: {username}")
if st.sidebar.button("Log out"):
    logout()
    st.rerun()

st.title("Lineup Tracker")

# Upload once, persist per user
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Lineup_Tracker Excel (.xlsx)", type=["xlsx"])
if uploaded is not None:
    save_path = user_upload_path(username)
    save_path.write_bytes(uploaded.getbuffer())
    st.session_state["data_path"] = str(save_path)
    st.sidebar.success("Saved. This file will load automatically next time you log in.")

data_path = st.session_state.get("data_path")
if not data_path or not Path(data_path).exists():
    st.info("Upload your Excel file in the sidebar to get started.")
    st.stop()

# Controls
st.sidebar.header("Settings")
lineup_size = st.sidebar.slider("Lineup size", min_value=1, max_value=5, value=5, step=1)
min_total_poss = st.sidebar.number_input("Minimum total possessions (OffPoss+DefPoss)", min_value=0.0, value=50.0, step=1.0)
per100_toggle = st.sidebar.toggle("Show rates per 100 possessions", value=True)
show_raw_columns = st.sidebar.toggle("Show raw sum columns", value=False)

# Load + aggregate
try:
    df_poss = load_possessions_from_xlsx(data_path)
except Exception as e:
    st.error(f"Could not read sheet 'Possessions'. {e}")
    st.stop()

agg = aggregate_lineups(df_poss, lineup_size)
if agg.empty:
    st.warning("No lineups found for that lineup size.")
    st.stop()

agg = agg[agg["TotalPoss"] >= float(min_total_poss)].copy()
agg = apply_per100(agg, per100_toggle)
agg = round_1_decimal(agg)

# Tabs
tabs = st.tabs(["Dashboard", "Offense", "Defense"])

with tabs[0]:
    st.subheader(f"Dashboard — {lineup_size}-Man Lineups")
    st.caption("Metrics: ORTG=100*PtsFor/OffPoss, DRTG=100*PtsAg/DefPoss, NRTG=ORTG-DRTG, "
               "TovR=TOVfor/OffPoss, RimR=RimAttFor/OffPoss, 3PR=ThreePA_For/OffPoss.")

    # Sort controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sort_metric = st.selectbox("Sort by", ["NRTG", "ORTG", "DRTG", "TovR", "RimR", "3PR", "TotalPoss"])
    with c2:
        asc = st.toggle("Ascending", value=False)
    with c3:
        topn = st.number_input("Show top N", min_value=10, value=50, step=10)

    base_cols = ["Lineup", "TotalPoss", "OffPoss", "DefPoss", "ORTG", "DRTG", "NRTG", "TovR", "RimR", "3PR"]
    view = agg.sort_values(sort_metric, ascending=bool(asc)).head(int(topn))

    if not show_raw_columns:
        view = view[base_cols]
    st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Offense")
    # For offense ranking, use OffPoss threshold since metrics are offensive events
    metric_map = {
        "Pts For": "PtsFor",
        "TOV": "TOV",
        "Offensive Rebounds": "OReb",
        "Opp Def Rebounds": "OppDefReb",
        "FTA For": "FTA For",
        "Rim Attempts": "RimAtt",
        "3P Attempts": "ThreePA",
        "Non-Paint 3": "NonPaint3",
        "Action 3": "Action3",
        "Transition 3": "Transition3",
        "Paint 3": "Paint3",
    }
    stat_rank_table(agg, "Rank offensive lineups", metric_map, poss_col="OffPoss")

with tabs[2]:
    st.subheader("Defense")
    metric_map = {
        "Pts Against": "PtsAg",
        "Non-Rim Against": "Non rim ag",
        "TOV Forced": "TOV Forced",
        "OReb Against": "OReb Ag",
        "Def Reb": "Def Reb",
        "FTA Against": "FTA Ag",
        "Rim Attempts Against": "RimAtt Ag",
        "3P Attempts Against": "ThreePA Ag",
    }
    stat_rank_table(agg, "Rank defensive lineups", metric_map, poss_col="DefPoss")
