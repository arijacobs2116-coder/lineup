import itertools
import re
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, List

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
# Auth helpers (SQLite + bcrypt) — Username/password
###############################################################################
USERNAME_RE = re.compile(r"[A-Za-z0-9_]{3,24}")

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
    return bool(USERNAME_RE.fullmatch((u or "").strip()))

def _hash_pw(password: str) -> bytes:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt)

def _check_pw(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def create_user(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
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
        return True, "Account created."
    finally:
        conn.close()

def authenticate(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT pw_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            return False, "Invalid username or password."
        pw_hash = row[0]
        if isinstance(pw_hash, memoryview):
            pw_hash = pw_hash.tobytes()
        ok = _check_pw(password, pw_hash)
        return (ok, "Logged in." if ok else "Invalid username or password.")
    finally:
        conn.close()

def user_upload_path(username: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", (username or "").strip())
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

def with_player_cols(df: pd.DataFrame, max_players: int = 5) -> pd.DataFrame:
    out = df.copy()
    players = out["Lineup"].astype(str).str.split(r"\s*\|\s*", regex=True)
    for i in range(1, max_players + 1):
        out[f"Player{i}"] = players.str.get(i - 1).fillna("")
    return out

@st.cache_data(show_spinner=False)
def precompute_aggs(df_poss_in: pd.DataFrame):
    return {k: aggregate_lineups(df_poss_in, k) for k in range(1, 6)}

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

    player_cols = [f"Player{i}" for i in range(1, 6) if f"Player{i}" in view.columns]
    front = player_cols + [poss_col, col]
    other = [c for c in view.columns if c not in front]
    view = view[front + other]

    st.dataframe(round_1_decimal(view), use_container_width=True, hide_index=True)

###############################################################################
# Auth UI (centered) — clickable Create account link via ?auth=signup
###############################################################################
def logout():
    st.session_state.pop("username", None)
    st.session_state.pop("data_path", None)

def auth_page():
    st.title("Lineup Tracker")

    qp = st.query_params
    view = qp.get("auth", "login")
    if isinstance(view, list):
        view = view[0] if view else "login"

    left, center, right = st.columns([1, 1.2, 1])
    with center:
        if view == "signup":
            st.markdown("### Create account")
            with st.form("create_form", clear_on_submit=False):
                username = st.text_input("Username", help="3–24 chars: letters/numbers/_")
                password = st.text_input("Password (8+ chars)", type="password")
                created = st.form_submit_button("Create account", use_container_width=True)

            if created:
                ok, msg = create_user(username, password)
                if ok:
                    st.success(msg)
                    st.info("Now log in.")
                    st.query_params["auth"] = "login"
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.markdown("Already have an account?")
            if st.button("Back to log in", use_container_width=True):
                st.query_params["auth"] = "login"
                st.rerun()

        else:
            st.markdown("### Log in")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", use_container_width=True)

            if submitted:
                ok, msg = authenticate(username, password)
                if ok:
                    username = (username or "").strip()
                    st.session_state["username"] = username
                    up = user_upload_path(username)
                    if up.exists():
                        st.session_state["data_path"] = str(up)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.markdown(
                "Don't have an account? "
                "<a href='?auth=signup' style='text-decoration:none; font-weight:600;'>Create account</a>",
                unsafe_allow_html=True,
            )

    st.caption("Tip: Upload your Excel once; it auto-loads whenever you log back in.")

###############################################################################
# Main app
###############################################################################
if "username" not in st.session_state:
    auth_page()
    st.stop()

username = st.session_state["username"]
st.sidebar.success(f"Logged in: {username}")
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
    st.sidebar.success("Saved. This file auto-loads next time you log in.")

data_path = st.session_state.get("data_path")
if not data_path or not Path(data_path).exists():
    st.info("Upload your Excel file in the sidebar to get started.")
    st.stop()

# Load base data
try:
    df_poss = load_possessions_from_xlsx(data_path)
except Exception as e:
    st.error(f"Could not read sheet 'Possessions'. {e}")
    st.stop()

all_players = sorted(set([p for c in ["P1","P2","P3","P4","P5"] for p in df_poss[c].unique() if isinstance(p, str) and p.strip()]))

agg_by_k = precompute_aggs(df_poss)

# Controls
st.sidebar.header("Settings")
lineup_size = st.sidebar.slider("Lineup size", 1, 5, 5, 1)
min_total_poss = st.sidebar.number_input("Minimum total possessions (OffPoss+DefPoss)", min_value=0.0, value=50.0, step=1.0)
per100_toggle = st.sidebar.toggle("Show rates per 100 possessions", value=True)
show_raw_columns = st.sidebar.toggle("Show raw sum columns", value=False)

# Aggregate for selected lineup size (for main tables)
agg = agg_by_k.get(int(lineup_size), pd.DataFrame()).copy()
if agg.empty:
    st.warning("No lineups found for that lineup size.")
    st.stop()

agg = agg[agg["TotalPoss"] >= float(min_total_poss)].copy()
agg = apply_per100(agg, per100_toggle)
agg = round_1_decimal(agg)
agg_view = with_player_cols(agg)

###############################################################################
# Tabs
###############################################################################
tabs = st.tabs(["Dashboard", "Comparison", "Test Lineup", "Offense", "Defense"])

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

    base_cols = ["Player1","Player2","Player3","Player4","Player5","TotalPoss","OffPoss","DefPoss","ORTG","DRTG","NRTG","TovR","RimR","3PR"]
    view = agg_view.sort_values(sort_metric, ascending=bool(asc)).head(int(topn))
    if not show_raw_columns:
        view = view[base_cols]
    st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Comparison")
    st.caption(
        "Pick metrics below. Each lineup gets a rank for every selected metric, then we average those ranks. "
        "**Lower average rank = better overall**."
    )

    metric_defs = {
        "ORTG (higher better)": ("ORTG", True),
        "DRTG (lower better)": ("DRTG", False),
        "NRTG (higher better)": ("NRTG", True),
        "TovR (lower better)": ("TovR", False),
        "RimR (higher better)": ("RimR", True),
        "3PR (higher better)": ("3PR", True),
    }

    st.markdown("#### Choose metrics")
    cols = st.columns(len(metric_defs))
    selected_labels: List[str] = []
    for i, label in enumerate(metric_defs.keys()):
        with cols[i]:
            if st.checkbox(label, value=False, key=f"cmp_cb_{i}"):
                selected_labels.append(label)

    if len(selected_labels) == 0:
        st.info("Select at least 1 metric to generate comparison rankings.")
    else:
        tmp = agg_view.copy()
        rank_cols = []
        for label in selected_labels:
            col, higher_better = metric_defs[label]
            tmp[f"Rank_{col}"] = tmp[col].rank(ascending=not higher_better, method="average")
            rank_cols.append(f"Rank_{col}")

        tmp["AvgRank"] = tmp[rank_cols].mean(axis=1)

        c1, c2 = st.columns([2, 1])
        with c1:
            topn = st.number_input("Show top N", min_value=10, value=50, step=10, key="cmp_topn")
        with c2:
            show_rank_cols = st.toggle("Show component ranks", value=True, key="cmp_show_ranks")

        show_cols = ["Player1","Player2","Player3","Player4","Player5","TotalPoss","OffPoss","DefPoss"]
        for label in selected_labels:
            col, _ = metric_defs[label]
            show_cols.append(col)
        if show_rank_cols:
            show_cols += rank_cols
        show_cols += ["AvgRank"]

        out = tmp.sort_values("AvgRank", ascending=True).head(int(topn))
        st.dataframe(round_1_decimal(out[show_cols]), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Test Lineup")
    st.caption(
        "Pick up to 5 players. We’ll show the exact lineup’s aggregated stats (order doesn’t matter). "
        "Leave boxes blank to test 1–4 player groups."
    )

    if st.button("Clear lineup", type="secondary"):
        for k in ["tl_p1", "tl_p2", "tl_p3", "tl_p4", "tl_p5"]:
            st.session_state[k] = ""
        st.rerun()

    def _opts(exclude):
        return [""] + [p for p in all_players if p not in exclude]

    c1, c2, c3, c4, c5 = st.columns(5)

    picked = set()

    with c1:
        p1 = st.selectbox("Player 1", _opts(picked), key="tl_p1")
        if p1: picked.add(p1)
    with c2:
        p2 = st.selectbox("Player 2", _opts(picked), key="tl_p2")
        if p2: picked.add(p2)
    with c3:
        p3 = st.selectbox("Player 3", _opts(picked), key="tl_p3")
        if p3: picked.add(p3)
    with c4:
        p4 = st.selectbox("Player 4", _opts(picked), key="tl_p4")
        if p4: picked.add(p4)
    with c5:
        p5 = st.selectbox("Player 5", _opts(picked), key="tl_p5")
        if p5: picked.add(p5)

    selected = [p for p in [p1, p2, p3, p4, p5] if p]
    if len(selected) == 0:
        st.info("Select at least one player.")
    else:
        k = len(selected)
        lineup_str = " | ".join(sorted(selected))

        dfk = agg_by_k.get(k, pd.DataFrame()).copy()
        if dfk.empty:
            st.warning("No data available for that lineup size.")
        else:
            dfk = dfk.copy()
            dfk["TotalPoss"] = dfk["OffPoss"] + dfk["DefPoss"]
            dfk = dfk[dfk["TotalPoss"] >= float(min_total_poss)].copy()
            dfk = apply_per100(dfk, per100_toggle)
            dfk = round_1_decimal(dfk)
            dfk_view = with_player_cols(dfk)

            row = dfk_view[dfk_view["Lineup"] == lineup_str]
            if row.empty:
                st.warning("That exact lineup isn’t in the data (after the current possession threshold).")
            else:
                st.markdown(f"#### {k}-Man: {lineup_str}")
                player_cols = [f"Player{i}" for i in range(1, k + 1)]
                base_cols = player_cols + ["TotalPoss","OffPoss","DefPoss","ORTG","DRTG","NRTG","TovR","RimR","3PR"]
                st.dataframe(row[base_cols], use_container_width=True, hide_index=True)

                st.markdown("##### Offense/Defense detail")
                detail_cols = [
                    "PtsFor","PtsAg","TOV","TOV Forced","OReb","OReb Ag","Def Reb","OppDefReb",
                    "FTA For","FTA Ag","RimAtt","RimAtt Ag","ThreePA","ThreePA Ag",
                    "Non rim","Non rim ag","NonPaint3","Action3","Transition3","Paint3"
                ]
                detail_cols = [c for c in detail_cols if c in row.columns]
                st.dataframe(row[player_cols + detail_cols], use_container_width=True, hide_index=True)

with tabs[3]:
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
    stat_rank_table(agg_view, "Rank offensive lineups", metric_map, poss_col="OffPoss")

with tabs[4]:
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
    stat_rank_table(agg_view, "Rank defensive lineups", metric_map, poss_col="DefPoss")
