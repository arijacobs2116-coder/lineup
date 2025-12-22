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
# Auth helpers (SQLite + bcrypt) — Email-based accounts
###############################################################################
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email   TEXT PRIMARY KEY,
            pw_hash BLOB NOT NULL
        );
        """
    )
    return conn

def _valid_email(email: str) -> bool:
    return bool(EMAIL_RE.fullmatch((email or "").strip().lower()))

def _hash_pw(password: str) -> bytes:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt)

def _check_pw(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def create_user(email: str, password: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    if not _valid_email(email):
        return False, "Enter a valid email address."
    if password is None or len(password) < 8:
        return False, "Password must be at least 8 characters."

    conn = _get_conn()
    try:
        cur = conn.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cur.fetchone() is not None:
            return False, "That email already exists."

        conn.execute(
            "INSERT INTO users(email, pw_hash) VALUES(?, ?)",
            (email, _hash_pw(password)),
        )
        conn.commit()
        return True, "Account created."
    finally:
        conn.close()

def authenticate(email: str, password: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT pw_hash FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if row is None:
            return False, "Invalid email or password."
        pw_hash = row[0]
        if isinstance(pw_hash, memoryview):
            pw_hash = pw_hash.tobytes()
        ok = _check_pw(password, pw_hash)
        return (ok, "Logged in." if ok else "Invalid email or password.")
    finally:
        conn.close()

def user_upload_path(email: str) -> Path:
    # simple filesystem-safe name
    safe = re.sub(r"[^a-z0-9_.-]+", "_", (email or "").strip().lower())
    return UPLOAD_DIR / f"{safe}.xlsx"

###############################################################################
# Data logic
###############################################################################
REQUIRED_COLUMNS = [
    "P1", "P2", "P3", "P4", "P5",
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

@st.cache_data(show_spinner=False)
def load_possessions_from_xlsx(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Possessions")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "PtsFor","PtsAg","TOV","TOV Forced","OReb","OReb Ag","Def Reb","OppDefReb",
        "FTA For","FTA Ag","RimAtt","RimAtt Ag","ThreePA","ThreePA Ag","Non rim","Non rim ag",
        "NonPaint3","Action3","Transition3","Paint3","OffPoss","DefPoss",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for p in ["P1","P2","P3","P4","P5"]:
        df[p] = df[p].astype(str).str.strip()
        df.loc[df[p].isin(["", "nan", "None", "NaN"]), p] = ""

    return df

def explode_to_combos(df: pd.DataFrame, k: int) -> pd.DataFrame:
    records = []
    player_cols = ["P1","P2","P3","P4","P5"]
    for _, row in df.iterrows():
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

    sum_cols = [
        "PtsFor","PtsAg","TOV","TOV Forced","OReb","OReb Ag","Def Reb","OppDefReb",
        "FTA For","FTA Ag","RimAtt","RimAtt Ag","ThreePA","ThreePA Ag","Non rim","Non rim ag",
        "NonPaint3","Action3","Transition3","Paint3","OffPoss","DefPoss",
    ]
    g = dfc.groupby("Lineup", as_index=False)[sum_cols].sum()

    # Derived metrics (protect divide-by-zero)
    g["ORTG"] = (100.0 * g["PtsFor"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["DRTG"] = (100.0 * g["PtsAg"] / g["DefPoss"].replace(0, pd.NA)).fillna(0)
    g["NRTG"] = g["ORTG"] - g["DRTG"]
    g["TovR"] = (g["TOV"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["RimR"] = (g["RimAtt"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["3PR"]  = (g["ThreePA"] / g["OffPoss"].replace(0, pd.NA)).fillna(0)
    g["TotalPoss"] = g["OffPoss"] + g["DefPoss"]
    return g

def apply_per100(df: pd.DataFrame, per100: bool) -> pd.DataFrame:
    out = df.copy()
    if per100:
        for c in ["TovR","RimR","3PR"]:
            out[c] = out[c] * 100.0
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

    c1, c2, c3, c4 = st.columns([2, 1, 3, 3])
    with c1:
        metric_label = st.selectbox("Rank by", list(metric_map.keys()), key=f"{title}_metric")
    with c2:
        ascending = st.toggle("Low is better", value=False, key=f"{title}_asc")
    with c3:
        min_poss = st.number_input(f"Minimum {poss_col}", min_value=0.0, value=25.0, step=1.0, key=f"{title}_minposs")
    with c4:
        topn = st.number_input("Show top N", min_value=10, value=50, step=10, key=f"{title}_topn")

    col = metric_map[metric_label]

    view = df_lineups.copy()
    if poss_col in view.columns:
        view = view[view[poss_col] >= float(min_poss)]
    view = view.sort_values(col, ascending=bool(ascending)).head(int(topn))

    front = ["Lineup", poss_col, col]
    other = [c for c in view.columns if c not in front]
    view = view[front + other]

    st.dataframe(round_1_decimal(view), use_container_width=True, hide_index=True)

###############################################################################
# Auth UI
###############################################################################
def logout():
    st.session_state.pop("email", None)
    st.session_state.pop("data_path", None)

def login_or_signup():
    st.title("Lineup Tracker")

    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "login"

    left, center, right = st.columns([1, 1.2, 1])
    with center:
        if st.session_state["auth_view"] == "login":
            st.markdown("### Log in")
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_pw")
                submitted = st.form_submit_button("Log in", use_container_width=True)

            if submitted:
                ok, msg = authenticate(email, password)
                if ok:
                    email = (email or "").strip().lower()
                    st.session_state["email"] = email
                    up = user_upload_path(email)
                    if up.exists():
                        st.session_state["data_path"] = str(up)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            # Clickable word vibe: secondary button that just flips the view
            cols = st.columns([1, 1, 1])
            with cols[1]:
                if st.button("Create account", type="secondary", use_container_width=True):
                    st.session_state["auth_view"] = "signup"
                    st.rerun()

        else:
            st.markdown("### Create account")
            with st.form("create_form", clear_on_submit=False):
                email = st.text_input("Email", key="new_email")
                password = st.text_input("Password (8+ chars)", type="password", key="new_pw")
                created = st.form_submit_button("Create account", use_container_width=True)

            if created:
                ok, msg = create_user(email, password)
                if ok:
                    st.success(msg)
                    st.info("Now log in with your email and password.")
                    st.session_state["auth_view"] = "login"
                    st.rerun()
                else:
                    st.error(msg)

            if st.button("Back to log in", use_container_width=True):
                st.session_state["auth_view"] = "login"
                st.rerun()

    st.caption("Tip: Upload your Excel once; it auto-loads whenever you log back in.")

###############################################################################
# Main app
###############################################################################
if "email" not in st.session_state:
    login_or_signup()
    st.stop()

email = st.session_state["email"]
st.sidebar.success(f"Logged in: {email}")
if st.sidebar.button("Log out"):
    logout()
    st.rerun()

st.title("Lineup Tracker")

# Upload once, persist per user
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Lineup_Tracker Excel (.xlsx)", type=["xlsx"])
if uploaded is not None:
    save_path = user_upload_path(email)
    save_path.write_bytes(uploaded.getbuffer())
    st.session_state["data_path"] = str(save_path)
    st.sidebar.success("Saved. This file auto-loads next time you log in.")

data_path = st.session_state.get("data_path")
if not data_path or not Path(data_path).exists():
    st.info("Upload your Excel file in the sidebar to get started.")
    st.stop()

# Controls
st.sidebar.header("Settings")
lineup_size = st.sidebar.slider("Lineup size", 1, 5, 5, 1)
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
    st.caption(
        "Metrics: ORTG=100*PtsFor/OffPoss, DRTG=100*PtsAg/DefPoss, NRTG=ORTG-DRTG, "
        "TovR=TOVfor/OffPoss, RimR=RimAttFor/OffPoss, 3PR=ThreePA_For/OffPoss."
    )

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
