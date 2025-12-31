import itertools
import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

import pandas as pd
import streamlit as st
import bcrypt

# ---------------------------------------------------------------------------
# PDF parsing helper import
# Streamlit Cloud images sometimes ship `pypdf` (new) but not `PyPDF2` (old).
# We'll try both and gracefully degrade if neither exists.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        PdfReader = None  # type: ignore

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

###########################################################
###############################################################################
# Game simulation (Monte Carlo) utilities
###############################################################################
def _safe_pct(made: float, att: float) -> float:
    return float(made) / float(att) if att and att > 0 else 0.0

def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)

def _read_player_stats_csv(uploaded_file) -> pd.DataFrame:
    """Read a CBBAnalytics-style player CSV (or similar) into a normalized frame."""
    df = pd.read_csv(uploaded_file)
    # Common renames (be defensive; different exports name columns slightly differently)
    rename_map = {
        "fullName": "player",
        "jerseyNum": "jersey",
        "teamMarket": "team",
        "minsPg": "mins_pg",
        "mins": "mins_total",
        "gp": "gp",
        "ptsScored": "pts",
        "fgm": "fgm",
        "fga": "fga",
        "fgm2": "fgm2",
        "fga2": "fga2",
        "fgm3": "fgm3",
        "fga3": "fga3",
        "ftm": "ftm",
        "fta": "fta",
        "ast": "ast",
        "tov": "tov",
        "orb": "orb",
        "drb": "drb",
        "reb": "reb",
        "stl": "stl",
        "blk": "blk",
        "pf": "pf",
        "poss": "poss",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Keep only rows that look like players (some exports include team totals)
    if "player" in df.columns:
        df = df[df["player"].notna()].copy()

    # Ensure required columns exist
    required = ["player", "mins_pg", "mins_total", "gp", "poss", "pts", "fga", "fgm", "fga2", "fgm2", "fga3", "fgm3",
                "fta", "ftm", "ast", "tov", "orb", "drb", "reb", "stl", "blk", "pf"]
    for c in required:
        if c not in df.columns:
            df[c] = 0

    # Coerce numerics
    for c in required:
        if c != "player":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Nice formatting
    if "jersey" in df.columns:
        df["jersey"] = df["jersey"].astype(str).str.replace(".0$", "", regex=True)

    if "team" not in df.columns:
        df["team"] = "Opponent"

    return df


def _merge_kenpom_player_table(raw_df: pd.DataFrame, kp_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join KenPom per-player table onto the raw player CSV frame.

    Priority match:
      1) Jersey number if present in both
      2) Cleaned player name

    Adds KenPom columns with a `kp_` prefix.
    """
    if kp_df is None or kp_df.empty:
        return raw_df

    df = raw_df.copy()

    # Clean keys
    if "jersey" in df.columns:
        df["_kp_jersey"] = df["jersey"].apply(clean_jersey)
    else:
        df["_kp_jersey"] = ""

    df["_kp_name"] = df["player"].astype(str).apply(clean_player_name).str.lower().str.replace(r"[^a-z\s]", "", regex=True).str.strip()

    kp = kp_df.copy()
    kp["_kp_jersey"] = kp["Jersey"].apply(clean_jersey)
    kp["_kp_name"] = kp["Player"].astype(str).apply(clean_player_name).str.lower().str.replace(r"[^a-z\s]", "", regex=True).str.strip()

    # First attempt: jersey match (only for rows where jersey is non-empty)
    out = df.merge(
        kp.drop(columns=["Jersey", "Player"], errors="ignore").add_prefix("kp_").assign(_kp_jersey=kp["_kp_jersey"]),
        on="_kp_jersey",
        how="left",
    )

    # For unmatched, try name match (fill only kp_* that are still NaN)
    kp_payload = kp.drop(columns=["Jersey", "Player"], errors="ignore").add_prefix("kp_").assign(_kp_name=kp["_kp_name"])
    out2 = out.merge(kp_payload, on="_kp_name", how="left", suffixes=("", "_name"))

    # Fill NaNs from name-join columns
    for c in kp_payload.columns:
        if c == "_kp_name":
            continue
        name_c = f"{c}_name"
        if name_c in out2.columns:
            if c in out2.columns:
                out2[c] = out2[c].where(out2[c].notna(), out2[name_c])
            else:
                out2[c] = out2[name_c]
            out2 = out2.drop(columns=[name_c])

    # Clean temp keys
    out2 = out2.drop(columns=[c for c in ["_kp_jersey", "_kp_name"] if c in out2.columns], errors="ignore")
    return out2

def _build_team_profile(df_players: pd.DataFrame, minutes_override: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, Dict]:
    """Compute per-possession / per-shot rates used by the simulator."""
    df = df_players.copy()

    # Minutes per game: override (GW) or use mins_pg (opponent)
    if minutes_override is not None:
        # Caller validates sum==200; we respect the chosen rotation exactly.
        df["mins_sim"] = df["player"].map(minutes_override).fillna(0.0).clip(lower=0.0)
    else:
        df["mins_sim"] = df["mins_pg"].fillna(0.0).clip(lower=0.0)

        # Scale opponent minutes so total player-minutes = 200 (40*5)
        total_mins = df["mins_sim"].sum()
        if total_mins <= 0:
            df["mins_sim"] = 0.0
        else:
            df["mins_sim"] = df["mins_sim"] * (200.0 / total_mins)

    # Possession share: combine minutes + observed poss/min to get a realistic touch/usage proxy
    df["poss_per_min"] = df["poss"] / df["mins_total"].replace(0, np.nan)
    df["poss_per_min"] = df["poss_per_min"].fillna(0.0)
    # Weight by minutes to produce expected poss share for the sim
    df["poss_weight"] = df["mins_sim"] * df["poss_per_min"]
    if df["poss_weight"].sum() > 0:
        df["poss_share"] = df["poss_weight"] / df["poss_weight"].sum()
    else:
        df["poss_share"] = 1.0 / max(len(df), 1)

    # Player-level probabilities/rates
    df["to_rate"] = (df["tov"] / df["poss"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    # If KenPom TORate (% of possessions) exists, blend it in (helps when CSV is noisy)
    if "kp_TORate" in df.columns and df["kp_TORate"].notna().any():
        kp_to = (pd.to_numeric(df["kp_TORate"], errors="coerce") / 100.0).clip(0, 1)
        df["to_rate"] = (0.6 * df["to_rate"] + 0.4 * kp_to).fillna(df["to_rate"]).clip(0, 1)
    df["fga_rate"] = (df["fga"] / df["poss"].replace(0, np.nan)).fillna(0.0).clip(0, 3)  # can exceed 1 w/ ORBs; that's ok
    df["three_share"] = (df["fga3"] / df["fga"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

    df["p2"] = (df["fgm2"] / df["fga2"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["p3"] = (df["fgm3"] / df["fga3"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["pft"] = (df["ftm"] / df["fta"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

    # Approximate FT attempts per FGA (shot-based free throws; caps keep it sane)
    df["fta_per_fga"] = (df["fta"] / df["fga"].replace(0, np.nan)).fillna(0.0).clip(0, 2)
    # If KenPom FTRate exists (KenPom reports it like 44.6 meaning 0.446), prefer it
    if "kp_FTRate" in df.columns and df["kp_FTRate"].notna().any():
        kp_ftr = (pd.to_numeric(df["kp_FTRate"], errors="coerce") / 100.0).clip(0, 2)
        df["fta_per_fga"] = df["fta_per_fga"].where(kp_ftr.isna(), kp_ftr).fillna(df["fta_per_fga"])
    # If KenPom FT% exists (parsed as decimal like .683), use it for pft
    if "kp_FT%" in df.columns and df["kp_FT%"].notna().any():
        df["pft"] = df["pft"].where(df["kp_FT%"].isna(), pd.to_numeric(df["kp_FT%"], errors="coerce")).clip(0, 1).fillna(df["pft"])

    # Shares for distributing secondary stats
    def _share(col: str) -> pd.Series:
        s = df[col].clip(lower=0)
        return (s / s.sum()) if s.sum() > 0 else (pd.Series([1/len(df)]*len(df), index=df.index) if len(df)>0 else pd.Series([], dtype=float))

    df["ast_share"] = _share("ast")
    df["orb_share"] = _share("orb")
    df["drb_share"] = _share("drb")
    df["reb_share"] = _share("reb")
    df["stl_share"] = _share("stl")
    df["blk_share"] = _share("blk")

    # If KenPom Stl% / Blk% exist, use them as distribution weights (more stable than raw counts)
    if "kp_Stl%" in df.columns and df["kp_Stl%"].notna().any():
        w = pd.to_numeric(df["kp_Stl%"], errors="coerce").clip(lower=0)
        df["stl_share"] = (w / w.sum()) if w.sum() > 0 else df["stl_share"]
    if "kp_Blk%" in df.columns and df["kp_Blk%"].notna().any():
        w = pd.to_numeric(df["kp_Blk%"], errors="coerce").clip(lower=0)
        df["blk_share"] = (w / w.sum()) if w.sum() > 0 else df["blk_share"]

    # Team-level aggregates
    team = {
        "name": str(df["team"].iloc[0]) if len(df) else "Team",
        "mins_total_sim": float(df["mins_sim"].sum()),
        "base_ppp": float(df["pts"].sum() / df["poss"].sum()) if df["poss"].sum() > 0 else 1.0,
        "orb_total": float(df["orb"].sum()),
        "drb_total": float(df["drb"].sum()),
        "pf_per_min": float(df["pf"].sum() / df["mins_total"].sum()) if df["mins_total"].sum() > 0 else 0.1,
    }
    return df, team

def _expected_adj_eff(adj_o: float, opp_adj_d: float, nat_avg: float = 100.0) -> float:
    # Simple, stable blend: Offense minus how far opponent defense is above average
    return float(adj_o) - (float(opp_adj_d) - nat_avg)

def _apply_kenpom_scaling(df_team: pd.DataFrame, target_ppp: float) -> pd.DataFrame:
    """Scale shooting/turnover a bit so that simulated PPP tracks the KenPom expectation."""
    df = df_team.copy()
    # Estimate team PPP from player profile
    est_ppp = df["poss_share"].mul(
        (1 - df["to_rate"]).clip(0, 1) * (  # possessions that are not turnovers
            # expected points from FG + FT
            (1 - df["three_share"]) * (2 * df["p2"]) +
            (df["three_share"]) * (3 * df["p3"]) +
            df["fta_per_fga"] * df["pft"]
        )
    ).sum()
    if est_ppp <= 0:
        return df

    scale = (target_ppp / est_ppp)
    # Bound so we don't do anything crazy
    scale = float(np.clip(scale, 0.85, 1.15))

    # Adjust makes slightly (sqrt scale), turnovers slightly (inverse)
    make_scale = math.sqrt(scale)
    tov_scale = 1.0 / scale

    df["p2_sim"] = (df["p2"] * make_scale).clip(0, 1)
    df["p3_sim"] = (df["p3"] * make_scale).clip(0, 1)
    df["pft_sim"] = (df["pft"] * make_scale).clip(0, 1)
    df["to_rate_sim"] = (df["to_rate"] * tov_scale).clip(0, 1)

    return df




# ---------------------------------------------------------------------------
# KenPom player advanced/usage table parsing (paste or CSV)
# ---------------------------------------------------------------------------
def clean_player_name(name: str) -> str:
    """
    Cleans KenPom names by removing junk like 'National Rank' etc.
    """
    if not isinstance(name, str):
        name = str(name)

    s = name.replace("\n", " ")
    # Remove 'National Rank...' (case-insensitive, optional spaces)
    s = re.sub(r"(?i)national\s*rank.*$", "", s)
    # Remove trailing digits like '1' in 'Boozer1'
    s = re.sub(r"\d+$", "", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip()


def clean_jersey(j):
    """
    Convert jersey values like '1', '01', '1.0', '1.00', ' 1.0' → '1'.
    If it's not numeric, return stripped string.
    """
    try:
        return str(int(float(str(j).strip())))
    except Exception:
        return str(j).strip()


def _final_clean_kenpom_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared final cleaning logic for KenPom advanced stats DataFrame.
    Assumes df already has 'Jersey' and 'Player' columns populated.
    """
    # Remove category header rows (blank jerseys)
    df = df[df["Jersey"].notna() & (df["Jersey"].str.len() > 0)].copy()

    # Clean jersey formatting
    df["Jersey"] = (
        df["Jersey"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # Clean player names (remove National Rank junk)
    df["Player"] = df["Player"].apply(clean_player_name)

    allowed_columns = [
        "Jersey",
        "Player",
        "ORtg",
        "%Poss",
        "%Shots",
        "eFG%",
        "TS%",
        "OR%",
        "DR%",
        "ARate",
        "TORate",
        "Blk%",
        "Stl%",
        "FC/40",
        "FD/40",
        "FTRate",
    ]

    df = df[[c for c in df.columns if c in allowed_columns]].copy()

    numeric_cols = [c for c in df.columns if c not in ["Jersey", "Player"]]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Truncate to 1 decimal place (KenPom style)
        df[col] = df[col].apply(
            lambda x: float(int(x * 10)) / 10 if pd.notna(x) else x
        )

    return df


def load_and_clean_kenpom_csv(uploaded_file):
    """
    Fallback: Loads a KenPom-style advanced stats CSV:
      - First two columns are Unnamed: 0 (jersey), Unnamed: 1 (name), etc.
    """
    df = pd.read_csv(uploaded_file)

    # Clean whitespace from headers
    df.columns = [c.strip() for c in df.columns]

    # Jersey + player
    if "Unnamed: 0" in df.columns and "Unnamed: 1" in df.columns:
        df["Jersey"] = df["Unnamed: 0"].astype(str).str.strip()
        df["Player"] = df["Unnamed: 1"].astype(str).str.strip()
        df = df.drop(columns=["Unnamed: 0", "Unnamed: 1"])
    else:
        # Fallback if headers were renamed
        jersey_col = None
        player_col = None
        for c in df.columns:
            lc = c.lower()
            if "jersey" in lc or lc in ("#", "no", "number"):
                jersey_col = c
            if "player" in lc or "name" in lc:
                player_col = c

        if jersey_col is None or player_col is None:
            raise ValueError("Could not find jersey/name columns in advanced stats CSV.")

        df["Jersey"] = df[jersey_col].astype(str).str.strip()
        df["Player"] = df[player_col].astype(str).str.strip()

    df = _final_clean_kenpom_df(df)
    return df


def extract_numbers(line: str):
    """Return list of floatable numbers found in a line."""
    nums = re.findall(r"[-+]?\d*\.?\d+", line)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except ValueError:
            continue
    return out


def parse_kenpom_paste(raw: str) -> pd.DataFrame:
    """
    Parse raw text copied directly from a KenPom player-usage/advanced page.

    Handles:
      - Header row (Ht Wt Yr G S ...)
      - Usage category headers (Go-to guys, etc.)
      - 'National Rank' lines
      - Line breaks and rank numbers between stats

    Returns a DataFrame with one row per player.
    """
    import re

    # ---------- 1) CLEAN LINES ----------
    lines = [ln.rstrip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    cleaned = []
    for ln in lines:
        s = ln.strip()

        # Top header row
        if s.startswith("Ht") and "Wt" in s and "Yr" in s:
            continue

        # Usage category headers
        if "possessions used" in s.lower():
            continue

        cleaned.append(ln)

    lines = cleaned

    # ---------- 2) FIND PLAYER BLOCK STARTS ----------
    # A player block starts with: "2 Jahvin Carter", "55 Sean Smith", etc.
    player_starts = [
        idx for idx, line in enumerate(lines)
        if re.match(r"^\s*\d+\s+[A-Za-z]", line)
    ]

    if not player_starts:
        raise ValueError("Could not find any player lines in the pasted KenPom text.")

    def parse_player_block(block_lines: list[str]) -> dict | None:
        # Flatten block so we don't care about original line breaks
        text = " ".join(block_lines)
        tokens = text.split()
        if not tokens:
            return None

        # ---- Remove the literal "National Rank" phrase + its tokens ----
        filtered = []
        i = 0
        while i < len(tokens):
            if (
                tokens[i].lower() == "national"
                and i + 1 < len(tokens)
                and tokens[i + 1].lower() == "rank"
            ):
                i += 2
                continue
            filtered.append(tokens[i])
            i += 1
        tokens = filtered

        # Jersey
        if not tokens[0].isdigit():
            return None
        jersey = tokens[0]

        # ---- Name runs from after jersey up until the height token (like "6-3") ----
        name_tokens = []
        height_idx = None
        for i in range(1, len(tokens)):
            if re.match(r"^\d+-\d+$", tokens[i]):  # 6-3, 6-11, etc.
                height_idx = i
                break
            name_tokens.append(tokens[i])

        if height_idx is None or not name_tokens:
            return None

        name = " ".join(name_tokens)

        # ---- Basic info: height, weight, year, games, starts (optional) ----
        def safe_get(idx: int, default: str = "") -> str:
            return tokens[idx] if idx < len(tokens) else default

        ht = safe_get(height_idx)
        wt = safe_get(height_idx + 1)
        yr = safe_get(height_idx + 2)

        j = height_idx + 3
        g = safe_get(j)
        j += 1

        # S (starts) is optional – if next token has a '.', it's actually %Min
        s = ""
        if j < len(tokens) and "." not in tokens[j]:
            s = tokens[j]
            j += 1

        # %Min (always decimal)
        pct_min = float(tokens[j])
        j += 1

        # ---- Helper: read next stat, skipping integer rank tokens ----
        def next_stat(idx: int):
            # Only accept tokens with '.' (all KenPom advanced stats use a decimal)
            while idx < len(tokens) and "." not in tokens[idx]:
                idx += 1
            if idx >= len(tokens):
                return None, idx
            v = float(tokens[idx])
            idx += 1
            return v, idx

        # ORtg after %Min (possibly with rank in between)
        ortg, j = next_stat(j)

        # Remaining advanced stats in order:
        stat_keys = [
            "%Poss",
            "%Shots",
            "eFG%",
            "TS%",
            "OR%",
            "DR%",
            "ARate",
            "TORate",
            "Blk%",
            "Stl%",
            "FC/40",
            "FD/40",
            "FTRate",
        ]

        stats: dict[str, float | None] = {}
        for key in stat_keys:
            stats[key], j = next_stat(j)

        # ---- Parse FT / 2P / 3P splits from the END ----
        def parse_splits(tokens: list[str]):
            ftma = ftpct = twoma = twopct = threema = threepct = None
            idx = len(tokens) - 1

            def prev_pct(k: int):
                # Move left until token contains '.' (".750", ".500", ".333", etc.)
                while k >= 0 and "." not in tokens[k]:
                    k -= 1
                if k < 0:
                    return None, k
                try:
                    v = float(tokens[k])
                except ValueError:
                    v = None
                return v, k - 1

            def prev_ma(k: int):
                while k >= 0:
                    if re.match(r"^\d+-\d+$", tokens[k]):  # like 9-12, 4-8
                        return tokens[k], k - 1
                    k -= 1
                return None, k

            threepct, idx = prev_pct(idx)
            threema, idx = prev_ma(idx)
            twopct, idx = prev_pct(idx)
            twoma, idx = prev_ma(idx)
            ftpct, idx = prev_pct(idx)
            ftma, idx = prev_ma(idx)

            return ftma, ftpct, twoma, twopct, threema, threepct

        ftma, ftpct, twoma, twopct, threema, threepct = parse_splits(tokens)

        row = {
            "Jersey": jersey,
            "Player": name,
            "Ht": ht,
            "Wt": wt,
            "Yr": yr,
            "G": g,
            "S": s,
            "%Min": pct_min,
            "ORtg": ortg,
            "%Poss": stats["%Poss"],
            "%Shots": stats["%Shots"],
            "eFG%": stats["eFG%"],
            "TS%": stats["TS%"],
            "OR%": stats["OR%"],
            "DR%": stats["DR%"],
            "ARate": stats["ARate"],
            "TORate": stats["TORate"],
            "Blk%": stats["Blk%"],
            "Stl%": stats["Stl%"],
            "FC/40": stats["FC/40"],
            "FD/40": stats["FD/40"],
            "FTRate": stats["FTRate"],
            "FTM-A": ftma,
            "FT%": ftpct,
            "2PM-A": twoma,
            "2P%": twopct,
            "3PM-A": threema,
            "3P%": threepct,
        }
        return row

    # ---------- 3) BUILD DATAFRAME ----------
    players: list[dict] = []
    for idx, start_idx in enumerate(player_starts):
        end_idx = player_starts[idx + 1] if idx + 1 < len(player_starts) else len(lines)
        block = lines[start_idx:end_idx]
        row = parse_player_block(block)
        if row is not None:
            players.append(row)

    df = pd.DataFrame(players)
    return df


def parse_cbb_team_pdf_zones_both(uploaded_pdf) -> pd.DataFrame:
    """Parse CBB Analytics TEAM player-profiles PDF for FGA% and FG% by zone.

    Expected zones in order:
      At Rim, In Paint, Midrange 2s, Above Break 3s, Corner 3s, At Rim + 3s, Heaves

    Returns a DataFrame with columns:
      Jersey, Player, FGA% <zone>..., FG% <zone>...
    """
    if PdfReader is None:
        raise ImportError(
            "PDF parsing requires the `pypdf` package. Add `pypdf` to requirements.txt on Streamlit Cloud."
        )

    from io import BytesIO

    pdf_bytes = uploaded_pdf.read()
    reader_local = PdfReader(BytesIO(pdf_bytes))

    rows = []
    zones = [
        "At Rim",
        "In Paint",
        "Midrange 2s",
        "Above Break 3s",
        "Corner 3s",
        "At Rim + 3s",
        "Heaves",
    ]

    for page in reader_local.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        if "Shot Zone" not in text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        header_line = lines[0]
        m = re.match(r"^(.*?)\s*\(#(\d+)", header_line)
        if not m:
            continue
        name = m.group(1).strip()
        jersey = m.group(2).strip()

        shot_idx = None
        for i, ln in enumerate(lines):
            if "Shot Zone" in ln and "FGA" in ln:
                shot_idx = i
                break
        if shot_idx is None:
            continue

        fga_vals, fg_vals = [], []
        for ln in lines[shot_idx + 1 :]:
            if (
                ln.startswith("DNQ")
                or "Zone % of Shots" in ln
                or "Zone FG%" in ln
                or "Shot Chart" in ln
            ):
                break

            parts = ln.split()
            if not parts:
                continue

            numeric_tokens = [t for t in parts if re.match(r"^-?\d+(\.\d+)?%?$", t)]
            if len(numeric_tokens) < 2:
                continue

            fga_token = numeric_tokens[-2]
            fg_token = numeric_tokens[-1]
            try:
                fga_vals.append(float(fga_token.replace("%", "")))
            except Exception:
                fga_vals.append(float("nan"))
            try:
                fg_vals.append(float(fg_token.replace("%", "")))
            except Exception:
                fg_vals.append(float("nan"))

            if len(fga_vals) >= len(zones):
                break

        if not fga_vals:
            continue

        while len(fga_vals) < len(zones):
            fga_vals.append(float("nan"))
            fg_vals.append(float("nan"))

        row = {"Jersey": jersey, "Player": name}
        for i, z in enumerate(zones):
            row[f"FGA% {z}"] = fga_vals[i]
            row[f"FG% {z}"] = fg_vals[i]
        rows.append(row)

    if not rows:
        raise ValueError("No Shot Zone tables with both FGA% and FG% found in PDF.")
    return pd.DataFrame(rows)


def _apply_shot_zone_overrides(df_players: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """Optionally override shooting mix/accuracy using CBB Analytics shot-zone tables.

    We derive:
      - three_share: (Above Break 3s + Corner 3s) / 100
      - p3: weighted 3PT FG% from the two 3PT zones
      - p2: weighted 2PT FG% from rim/paint/midrange

    Matching tries Jersey first (if present) then player name (case-insensitive).
    """
    if zones_df is None or zones_df.empty:
        return df_players

    df = df_players.copy()
    z = zones_df.copy()

    # Normalize columns to numeric
    core_zones = ["At Rim", "In Paint", "Midrange 2s", "Above Break 3s", "Corner 3s"]
    for col in [f"FGA% {c}" for c in core_zones] + [f"FG% {c}" for c in core_zones]:
        if col in z.columns:
            z[col] = pd.to_numeric(z[col], errors="coerce")

    # Keys for matching (robust to different column names)
    # Identify likely player/jersey columns
    df_player_col = "player" if "player" in df.columns else ("Player" if "Player" in df.columns else None)
    df_jersey_col = "jersey" if "jersey" in df.columns else ("Jersey" if "Jersey" in df.columns else None)

    if df_jersey_col is not None:
        df[df_jersey_col] = df[df_jersey_col].astype(str).str.strip()
    if "Jersey" in z.columns:
        z["Jersey"] = z["Jersey"].astype(str).str.strip()

    # Internal key for fallback name matching (never returned to caller)
    if df_player_col is not None:
        df["_player_key"] = df[df_player_col].astype(str).str.strip().str.lower()
    else:
        df["_player_key"] = ""

    if "Player" in z.columns:
        z["_player_key"] = z["Player"].astype(str).str.strip().str.lower()
    else:
        z["_player_key"] = ""
    # Merge (prefer jersey if present, otherwise name key)
    merged = None
    if df_jersey_col is not None and df[df_jersey_col].replace("", np.nan).notna().any() and "Jersey" in z.columns:
        merged = df.merge(z, how="left", left_on=df_jersey_col, right_on="Jersey")
    else:
        merged = df.merge(z, how="left", left_on="_player_key", right_on="_player_key")

    # If we merged on jersey, pandas will suffix duplicate _player_key columns.
    # Ensure we have a plain _player_key column so downstream code never errors.
    if "_player_key" not in merged.columns:
        if "_player_key_x" in merged.columns:
            merged["_player_key"] = merged["_player_key_x"]
        elif "_player_key_y" in merged.columns:
            merged["_player_key"] = merged["_player_key_y"]

    # Compute derived overrides
    ab3 = merged.get("FGA% Above Break 3s")
    c3 = merged.get("FGA% Corner 3s")
    if ab3 is not None and c3 is not None:
        three_share = (ab3.fillna(0) + c3.fillna(0)) / 100.0
        merged["three_share"] = three_share.clip(0, 1)

    # 3PT accuracy
    fg_ab3 = merged.get("FG% Above Break 3s")
    fg_c3 = merged.get("FG% Corner 3s")
    if (ab3 is not None) and (c3 is not None) and (fg_ab3 is not None) and (fg_c3 is not None):
        w3 = (ab3.fillna(0) + c3.fillna(0)).replace(0, np.nan)
        p3 = ((ab3.fillna(0) * fg_ab3.fillna(0)) + (c3.fillna(0) * fg_c3.fillna(0))) / w3
        merged["p3"] = (p3 / 100.0).fillna(merged["p3"]).clip(0, 1)

    # 2PT accuracy
    ar = merged.get("FGA% At Rim")
    ip = merged.get("FGA% In Paint")
    mr = merged.get("FGA% Midrange 2s")
    fg_ar = merged.get("FG% At Rim")
    fg_ip = merged.get("FG% In Paint")
    fg_mr = merged.get("FG% Midrange 2s")
    if all(v is not None for v in [ar, ip, mr, fg_ar, fg_ip, fg_mr]):
        w2 = (ar.fillna(0) + ip.fillna(0) + mr.fillna(0)).replace(0, np.nan)
        p2 = ((ar.fillna(0) * fg_ar.fillna(0)) + (ip.fillna(0) * fg_ip.fillna(0)) + (mr.fillna(0) * fg_mr.fillna(0))) / w2
        merged["p2"] = (p2 / 100.0).fillna(merged["p2"]).clip(0, 1)

    # Clean up: do not return helper key
    keep = [c for c in df.columns.tolist() if c != "_player_key"]
    out = merged[keep].copy()
    return out

def _simulate_one_game(
    gw_players: pd.DataFrame,
    opp_players: pd.DataFrame,
    kenpom: Dict,
    rng: np.random.Generator,
) -> Tuple[Dict, Dict]:
    """Return (team_results, player_boxscores) for one simulated game."""
    gw = gw_players
    op = opp_players

    # Possessions (tempo)
    tempo_gw = float(kenpom["gw_tempo"])
    tempo_op = float(kenpom["op_tempo"])
    mu_poss = (tempo_gw + tempo_op) / 2.0
    poss = int(np.clip(rng.normal(mu_poss, 3.0), 55, 80))

    # Team ORB probability proxy
    gw_orb = gw["orb"].sum()
    op_drb = op["drb"].sum()
    op_orb = op["orb"].sum()
    gw_drb = gw["drb"].sum()
    gw_orb_prob = float(gw_orb / (gw_orb + op_drb)) if (gw_orb + op_drb) > 0 else 0.28
    op_orb_prob = float(op_orb / (op_orb + gw_drb)) if (op_orb + gw_drb) > 0 else 0.28
    gw_orb_prob = float(np.clip(gw_orb_prob, 0.18, 0.38))
    op_orb_prob = float(np.clip(op_orb_prob, 0.18, 0.38))

    # Player boxes
    def blank_box(df):
        cols = ["MIN","PTS","FGM","FGA","3PM","3PA","FTM","FTA","REB","AST","TO","STL","BLK","OREB","DREB","PF"]
        box = {p: {c: 0.0 for c in cols} for p in df["player"].tolist()}
        # Minutes are deterministic in this simplified model
        for p, m in zip(df["player"], df["mins_sim"]):
            box[p]["MIN"] = float(m)
        return box

    gw_box = blank_box(gw)
    op_box = blank_box(op)

    def run_team_possessions(team_df, opp_df, team_box, opp_box, orb_prob):
        pts = 0
        # precompute arrays for speed
        players = team_df["player"].to_numpy()
        poss_share = team_df["poss_share"].to_numpy()
        to_rate = team_df["to_rate_sim"].to_numpy()
        three_share = team_df["three_share"].to_numpy()
        p2 = team_df["p2_sim"].to_numpy()
        p3 = team_df["p3_sim"].to_numpy()
        pft = team_df["pft_sim"].to_numpy()
        fta_per_fga = team_df["fta_per_fga"].to_numpy()
        fga_rate = team_df["fga_rate"].to_numpy()
        ast_share = team_df["ast_share"].to_numpy()
        orb_share = team_df["orb_share"].to_numpy()
        drb_share = team_df["drb_share"].to_numpy()
        stl_share = team_df["stl_share"].to_numpy()
        blk_share = team_df["blk_share"].to_numpy()

        # helper to choose rebounder/assister
        def pick(idx_probs, exclude=None):
            if exclude is None:
                return int(rng.choice(len(players), p=idx_probs))
            # crude exclude: zero out and renormalize
            probs = idx_probs.copy()
            probs[exclude] = 0
            s = probs.sum()
            if s <= 0:
                return int(rng.integers(len(players)))
            probs = probs / s
            return int(rng.choice(len(players), p=probs))

        
        # Team-level context for steal/block rates
        opp_stl_per_pos = float(opp_df["stl"].sum() / opp_df["poss"].sum()) if (len(opp_df) and opp_df["poss"].sum() > 0) else 0.08
        opp_blk_per_pos = float(opp_df["blk"].sum() / opp_df["poss"].sum()) if (len(opp_df) and opp_df["poss"].sum() > 0) else 0.05

        # Expected 2PA per possession for this team (for mapping blocks onto 2PA)
        exp_fga_per_pos = float(np.sum(poss_share * fga_rate))
        exp_two_pa_per_pos = float(np.sum(poss_share * fga_rate * (1 - three_share)))
        exp_tov_per_pos = float(np.sum(poss_share * to_rate))

        # Probability a turnover is credited as a steal (bounded)
        p_steal_given_tov = float(np.clip(opp_stl_per_pos / max(exp_tov_per_pos, 1e-6), 0.35, 0.85))

        # Probability a 2PA is blocked (bounded)
        p_block_given_2pa = float(np.clip(opp_blk_per_pos / max(exp_two_pa_per_pos, 1e-6), 0.03, 0.18))

        for _ in range(poss):
            shooter_i = int(rng.choice(len(players), p=poss_share))
            shooter = players[shooter_i]

            had_shot = False  # used to decide whether to award FTs
            main_shot_is_three = False

            # Turnover ends the possession: NO shot, NO FTs.
            if rng.random() < to_rate[shooter_i]:
                team_box[shooter]["TO"] += 1

                # Steal attribution: choose a defender weighted by opponent steal share
                if len(opp_df) > 0 and rng.random() < p_steal_given_tov:
                    stl_probs = opp_df["stl_share"].to_numpy() if "stl_share" in opp_df.columns else None
                    if stl_probs is not None and stl_probs.sum() > 0:
                        stl_i = int(rng.choice(len(opp_df), p=stl_probs))
                    else:
                        stl_i = int(rng.integers(len(opp_df)))
                    stl_p = opp_df["player"].iloc[stl_i]
                    if stl_p in opp_box:
                        opp_box[stl_p]["STL"] += 1
                continue  # possession over

            # Decide shot type: 3PA vs 2PA
            main_shot_is_three = (rng.random() < three_share[shooter_i])

            if main_shot_is_three:
                had_shot = True
                team_box[shooter]["FGA"] += 1
                team_box[shooter]["3PA"] += 1
                made = rng.random() < p3[shooter_i]
                if made:
                    team_box[shooter]["3PM"] += 1
                    team_box[shooter]["FGM"] += 1
                    team_box[shooter]["PTS"] += 3
                    pts += 3
                    # Assist?
                    if rng.random() < 0.58 and len(players) > 1:
                        a_i = pick(ast_share, exclude=shooter_i)
                        team_box[players[a_i]]["AST"] += 1
                else:
                    # Offensive rebound chance
                    if rng.random() < orb_prob:
                        r_i = pick(orb_share)
                        rebounder = players[r_i]
                        team_box[rebounder]["OREB"] += 1
                        team_box[rebounder]["REB"] += 1
                        # Putback 2PA (quick)
                        team_box[rebounder]["FGA"] += 1
                        made2 = rng.random() < min(0.75, p2[r_i] + 0.08)
                        if made2:
                            team_box[rebounder]["FGM"] += 1
                            team_box[rebounder]["PTS"] += 2
                            pts += 2
                            if rng.random() < 0.52 and len(players) > 1:
                                a_i = pick(ast_share, exclude=r_i)
                                team_box[players[a_i]]["AST"] += 1
                # FTs on 3PA are rare; we keep it simple: no FTs on 3s here.

            else:
                had_shot = True
                team_box[shooter]["FGA"] += 1

                # Block check on 2PA (uses opponent block environment)
                was_blocked = (len(opp_df) > 0 and rng.random() < p_block_given_2pa)
                if was_blocked:
                    # Attribute block to an opponent defender weighted by block share
                    blk_probs = opp_df["blk_share"].to_numpy() if "blk_share" in opp_df.columns else None
                    if blk_probs is not None and blk_probs.sum() > 0:
                        blk_i = int(rng.choice(len(opp_df), p=blk_probs))
                    else:
                        blk_i = int(rng.integers(len(opp_df)))
                    blk_p = opp_df["player"].iloc[blk_i]
                    if blk_p in opp_box:
                        opp_box[blk_p]["BLK"] += 1
                    made = False
                else:
                    made = rng.random() < p2[shooter_i]

                if made:
                    team_box[shooter]["FGM"] += 1
                    team_box[shooter]["PTS"] += 2
                    pts += 2
                    if rng.random() < 0.55 and len(players) > 1:
                        a_i = pick(ast_share, exclude=shooter_i)
                        team_box[players[a_i]]["AST"] += 1
                else:
                    # Offensive rebound chance
                    if rng.random() < orb_prob:
                        r_i = pick(orb_share)
                        rebounder = players[r_i]
                        team_box[rebounder]["OREB"] += 1
                        team_box[rebounder]["REB"] += 1
                        # Putback 2PA
                        team_box[rebounder]["FGA"] += 1
                        made2 = rng.random() < min(0.75, p2[r_i] + 0.1)
                        if made2:
                            team_box[rebounder]["FGM"] += 1
                            team_box[rebounder]["PTS"] += 2
                            pts += 2
                            if rng.random() < 0.52 and len(players) > 1:
                                a_i = pick(ast_share, exclude=r_i)
                                team_box[players[a_i]]["AST"] += 1

                # Free throws off the main 2PA (rough, but only if a shot happened)
                lam = fta_per_fga[shooter_i]
                if lam > 0 and rng.random() < 0.55:
                    nfta = int(min(3, rng.poisson(lam)))
                    if nfta > 0:
                        team_box[shooter]["FTA"] += nfta
                        ftm = int(rng.binomial(nfta, pft[shooter_i]))
                        team_box[shooter]["FTM"] += ftm
                        team_box[shooter]["PTS"] += ftm
                        pts += ftm

        return pts


    gw_pts = run_team_possessions(gw, op, gw_box, op_box, gw_orb_prob)
    op_pts = run_team_possessions(op, gw, op_box, gw_box, op_orb_prob)

    # Distribute defensive rebounds and PFs (minute-proportional, low-variance)
    def finalize_boxes(team_df, box, opp_missed_shots):
        # Defensive rebounds: allocate remaining rebounds by drb share
        total_drb = team_df["drb"].sum()
        if total_drb > 0:
            # Approximate team defensive reb total from season rate scaled to game misses
            est_drb = float(team_df["drb"].sum() / max(team_df["fga"].sum() - team_df["fgm"].sum(), 1)) * opp_missed_shots
            est_drb = float(np.clip(est_drb, 18, 35))
        else:
            est_drb = 25.0
        # allocate
        for i, p in enumerate(team_df["player"]):
            share = float(team_df.loc[team_df["player"] == p, "drb_share"].iloc[0])
            drb_i = est_drb * share
            box[p]["DREB"] += drb_i
            box[p]["REB"] += drb_i

        # PF: prefer KenPom FC/40 if present, else fallback to season PF/min
        if "kp_FC/40" in team_df.columns and team_df["kp_FC/40"].notna().any():
            fc40 = pd.to_numeric(team_df["kp_FC/40"], errors="coerce").fillna(0.0).clip(lower=0.0)
            for p, m, r in zip(team_df["player"], team_df["mins_sim"], fc40):
                lam_pf = max(0.0, float(r) / 40.0 * float(m))
                box[p]["PF"] = float(rng.poisson(lam_pf))
        else:
            pf_per_min = float(team_df["pf"].sum() / team_df["mins_total"].sum()) if team_df["mins_total"].sum() > 0 else 0.12
            for p, m in zip(team_df["player"], team_df["mins_sim"]):
                box[p]["PF"] = float(rng.poisson(max(0.0, pf_per_min * m)))

    gw_misses = (sum(v["FGA"] for v in gw_box.values()) - sum(v["FGM"] for v in gw_box.values()))
    op_misses = (sum(v["FGA"] for v in op_box.values()) - sum(v["FGM"] for v in op_box.values()))
    finalize_boxes(gw, gw_box, op_misses)
    finalize_boxes(op, op_box, gw_misses)

    team_res = {"GW": gw_pts, "OPP": op_pts, "poss": poss}
    player_res = {"GW": gw_box, "OPP": op_box}
    return team_res, player_res

def _aggregate_boxscores(boxscores: List[Dict]) -> Dict:
    """Average a list of boxscore dicts (player -> stat -> value)."""
    if not boxscores:
        return {}
    players = boxscores[0].keys()
    out = {p: {} for p in players}
    stats = next(iter(boxscores[0].values())).keys()
    for p in players:
        for s in stats:
            out[p][s] = float(np.mean([b[p][s] for b in boxscores]))
    return out

def _round_boxscore(avg_box: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Turn averaged boxscores into a displayable DataFrame with realistic rounding."""
    rows = []
    for p, stt in avg_box.items():
        fga = int(round(stt["FGA"]))
        fgm = int(round(min(fga, stt["FGM"])))
        tpa = int(round(stt["3PA"]))
        tpm = int(round(min(tpa, stt["3PM"])))
        fta = int(round(stt["FTA"]))
        ftm = int(round(min(fta, stt["FTM"])))
        row = {
            "PLAYER": p,
            "MIN": int(round(stt["MIN"])),
            "PTS": int(round(stt["PTS"])),
            "FG": f"{fgm}-{fga}",
            "3PT": f"{tpm}-{tpa}",
            "FT": f"{ftm}-{fta}",
            "REB": int(round(stt["REB"])),
            "AST": int(round(stt["AST"])),
            "TO": int(round(stt["TO"])),
            "STL": int(round(stt["STL"])),
            "BLK": int(round(stt["BLK"])),
            "OREB": int(round(stt["OREB"])),
            "DREB": int(round(stt["DREB"])),
            "PF": int(round(stt["PF"])),
        }
        rows.append(row)
    dfr = pd.DataFrame(rows)
    # Sort by minutes, then points
    if not dfr.empty:
        dfr = dfr.sort_values(["MIN", "PTS"], ascending=[False, False]).reset_index(drop=True)
    return dfr


####################
# Tabs
###############################################################################
tabs = st.tabs(["Dashboard", "Comparison", "Test Lineup", "Offense", "Defense", "Simulation"])

with tabs[0]:
    st.subheader(f"Dashboard — {lineup_size}-Man Lineups")
    
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
            topn = st.number_input("Show Top", min_value=10, value=50, step=10, key="cmp_topn")
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



with tabs[5]:
    st.subheader("Game Simulation (Monte Carlo)")

    st.markdown(
        """Simulate a matchup using (1) per-player season stats from CSVs and (2) team-level KenPom inputs.
- **GW minutes**: you choose minutes per player (we auto-scale to 200 total).
- **Opponent minutes**: derived from each player's average minutes per game in their CSV.
- Output: **score distribution + spread** and an **average box score** (rounded realistically) across simulations."""
    )

    c_up1, c_up2 = st.columns(2)
    with c_up1:
        gw_csv = st.file_uploader("Upload GW player-stats CSV (.csv)", type=["csv"], key="gw_player_csv")
    with c_up2:
        opp_csv = st.file_uploader("Upload Opponent player-stats CSV (.csv)", type=["csv"], key="opp_player_csv")

    c_pdf1, c_pdf2 = st.columns(2)
    with c_pdf1:
        gw_pdf = st.file_uploader(
            "Optional: Upload GW CBB Analytics player-profiles shot-chart PDF (.pdf)",
            type=["pdf"],
            key="gw_shot_pdf",
        )
    with c_pdf2:
        opp_pdf = st.file_uploader(
            "Optional: Upload Opponent CBB Analytics player-profiles shot-chart PDF (.pdf)",
            type=["pdf"],
            key="opp_shot_pdf",
        )


    # -----------------------------------------------------------------------
    # KenPom per-player advanced/usage tables (paste or upload)
    # -----------------------------------------------------------------------
    gw_kp_df = None
    op_kp_df = None

    with st.expander("KenPom player tables (optional, but recommended)", expanded=True):
        st.caption(
            "Paste the **expanded player page** table from KenPom (like the screenshot) or upload a CSV export. "
            "We'll merge FC/40, FD/40, TORate, Stl%, Blk%, FTRate, etc. into the simulation."
        )

        kp_col1, kp_col2 = st.columns(2)
        with kp_col1:
            gw_kp_raw = st.text_area(
                "GW: Paste raw KenPom advanced/usage table here",
                value="",
                height=180,
                key="gw_kp_player_raw",
            )
            gw_kp_csv = st.file_uploader("Or upload a GW KenPom advanced CSV", type=["csv"], key="gw_kp_player_csv")

            try:
                if gw_kp_csv is not None:
                    gw_kp_df = load_and_clean_kenpom_csv(gw_kp_csv)
                elif gw_kp_raw.strip():
                    gw_kp_df = parse_kenpom_paste(gw_kp_raw)
            except Exception as e:
                st.warning(f"GW KenPom parse issue: {e}")
                gw_kp_df = None

            if gw_kp_df is not None and not gw_kp_df.empty:
                st.success(f"Parsed GW KenPom table: {len(gw_kp_df)} players.")
                st.dataframe(gw_kp_df.head(12), use_container_width=True, hide_index=True)

        with kp_col2:
            op_kp_raw = st.text_area(
                "Opponent: Paste raw KenPom advanced/usage table here",
                value="",
                height=180,
                key="op_kp_player_raw",
            )
            op_kp_csv = st.file_uploader(
                "Or upload an Opponent KenPom advanced CSV",
                type=["csv"],
                key="op_kp_player_csv",
            )

            try:
                if op_kp_csv is not None:
                    op_kp_df = load_and_clean_kenpom_csv(op_kp_csv)
                elif op_kp_raw.strip():
                    op_kp_df = parse_kenpom_paste(op_kp_raw)
            except Exception as e:
                st.warning(f"Opponent KenPom parse issue: {e}")
                op_kp_df = None

            if op_kp_df is not None and not op_kp_df.empty:
                st.success(f"Parsed Opponent KenPom table: {len(op_kp_df)} players.")
                st.dataframe(op_kp_df.head(12), use_container_width=True, hide_index=True)

    if gw_csv is None or opp_csv is None:
        st.info("Upload both CSVs to enable simulation.")
        st.stop()

    try:
        gw_raw = _read_player_stats_csv(gw_csv)
        op_raw = _read_player_stats_csv(opp_csv)
        # Merge KenPom per-player tables (if provided)
        if 'gw_kp_df' in locals() and gw_kp_df is not None:
            gw_raw = _merge_kenpom_player_table(gw_raw, gw_kp_df)
        if 'op_kp_df' in locals() and op_kp_df is not None:
            op_raw = _merge_kenpom_player_table(op_raw, op_kp_df)
    except Exception as e:
        st.error(f"Could not read one of the CSV files: {e}")
        st.stop()

    # Team labels
    gw_team_default = str(gw_raw["team"].iloc[0]) if "team" in gw_raw.columns and len(gw_raw) else "George Washington"
    op_team_default = str(op_raw["team"].iloc[0]) if "team" in op_raw.columns and len(op_raw) else "Opponent"

    
    # Optional: paste KenPom team table text to auto-fill AdjO/AdjD/AdjT
    with st.expander("Paste KenPom team table (optional)", expanded=False):
        st.caption("Paste the KenPom team summary text (e.g., lines containing 'Adj. Efficiency' and 'Adj. Tempo'). We'll try to extract AdjO, AdjD, and AdjT.")
        gw_kp_text = st.text_area("GW KenPom paste", value="", height=120, key="gw_kp_text")
        op_kp_text = st.text_area("Opponent KenPom paste", value="", height=120, key="op_kp_text")

        def _parse_kenpom_team_text(txt: str):
            if not txt:
                return {}
            # normalize
            t = txt.replace("\t", " ").replace("  ", " ")
            # Try to find numbers on lines
            adj_eff = re.search(r"Adj\.?\s*Efficiency\s*([0-9]+\.?[0-9]*)\s*([0-9]+\.?[0-9]*)", t, flags=re.I)
            adj_tempo = re.search(r"Adj\.?\s*Tempo\s*([0-9]+\.?[0-9]*)", t, flags=re.I)
            out = {}
            if adj_eff:
                out["AdjO"] = float(adj_eff.group(1))
                out["AdjD"] = float(adj_eff.group(2))
            if adj_tempo:
                out["AdjT"] = float(adj_tempo.group(1))
            return out

        if st.button("Extract KenPom numbers", use_container_width=True):
            st.session_state.setdefault("kp_fill", {})
            st.session_state["kp_fill"]["gw"] = _parse_kenpom_team_text(gw_kp_text)
            st.session_state["kp_fill"]["op"] = _parse_kenpom_team_text(op_kp_text)
            st.success("Parsed (if possible). Values will populate below when available.")
    st.markdown("### KenPom team inputs")
    kp1, kp2, kp3 = st.columns(3)
    with kp1:
        gw_adj_o = st.number_input("GW AdjO", value=float(st.session_state.get("kp_fill",{}).get("gw",{}).get("AdjO",118.2)), step=0.1, format="%.1f")
        op_adj_o = st.number_input("Opponent AdjO", value=float(st.session_state.get("kp_fill",{}).get("op",{}).get("AdjO",110.0)), step=0.1, format="%.1f")
    with kp2:
        gw_adj_d = st.number_input("GW AdjD", value=float(st.session_state.get("kp_fill",{}).get("gw",{}).get("AdjD",107.7)), step=0.1, format="%.1f")
        op_adj_d = st.number_input("Opponent AdjD", value=float(st.session_state.get("kp_fill",{}).get("op",{}).get("AdjD",107.7)), step=0.1, format="%.1f")
    with kp3:
        gw_tempo = st.number_input("GW AdjTempo", value=float(st.session_state.get("kp_fill",{}).get("gw",{}).get("AdjT",71.6)), step=0.1, format="%.1f")
        op_tempo = st.number_input("Opponent AdjTempo", value=float(st.session_state.get("kp_fill",{}).get("op",{}).get("AdjT",69.2)), step=0.1, format="%.1f")

    st.markdown("### GW minutes (you control these — must total **200**)" )
    st.caption("Minutes are **integers only**. Total player-minutes must equal 200 (40 minutes * 5 players).")

    gw_min_df = gw_raw[["player", "mins_pg"]].copy()
    gw_min_df["mins_pg"] = gw_min_df["mins_pg"].fillna(0.0)
    gw_min_df = gw_min_df.sort_values("mins_pg", ascending=False).reset_index(drop=True)

    # Convenience: seed sliders from mins_pg scaled to 200
    if "gw_minutes_seeded" not in st.session_state:
        st.session_state["gw_minutes_seeded"] = False
    if st.button("Auto-fill GW minutes from CSV (scaled to 200)", type="secondary") or (not st.session_state["gw_minutes_seeded"]):
        base = gw_min_df["mins_pg"].to_numpy(dtype=float)
        if base.sum() > 0:
            scaled = base * (200.0 / base.sum())
        else:
            scaled = np.zeros_like(base)
        # round to ints that sum to 200
        ints = np.floor(scaled).astype(int)
        rem = int(200 - ints.sum())
        if rem > 0:
            frac_order = np.argsort(-(scaled - np.floor(scaled)))
            for i in frac_order[:rem]:
                ints[i] += 1
        for i, p in enumerate(gw_min_df["player"].tolist()):
            st.session_state[f"gw_min_{p}"] = int(ints[i])
        st.session_state["gw_minutes_seeded"] = True

    minutes_override: Dict[str, int] = {}
    cols = st.columns(2)
    for i, p in enumerate(gw_min_df["player"].tolist()):
        with cols[i % 2]:
            minutes_override[p] = int(
                st.slider(
                    p,
                    min_value=0,
                    max_value=40,
                    step=1,
                    value=int(st.session_state.get(f"gw_min_{p}", 0)),
                    key=f"gw_min_{p}",
                )
            )

    total_minutes = int(sum(minutes_override.values()))
    st.write(f"**Total GW minutes:** {total_minutes} / 200")
    if total_minutes != 200:
        st.error("GW minutes must sum to **exactly 200**. Adjust the sliders before running simulations.")
        st.stop()
        st.stop()

    c_sims1, c_sims2 = st.columns([1, 2])
    with c_sims1:
        nsims = st.number_input("Simulations", min_value=100, max_value=50000, value=1000, step=100)
    with c_sims2:
        st.caption("Tip: 5,000–20,000 is usually plenty; 50,000 is a hard cap.")

    run = st.button("Run simulations", type="primary")

    if not run:
        st.stop()

    # Build team profiles
    gw_df, gw_team = _build_team_profile(gw_raw, minutes_override=minutes_override)
    op_df, op_team = _build_team_profile(op_raw, minutes_override=None)

    # Optional: shot-chart PDFs to refine shot mix + zone FG% (CBB Analytics player-profiles PDFs)
    if gw_pdf is not None:
        try:
            gw_zones = parse_cbb_team_pdf_zones_both(gw_pdf)
            gw_df = _apply_shot_zone_overrides(gw_df, gw_zones)
            st.success("GW shot-chart PDF parsed: applied shot-mix + zone FG% overrides.")
        except Exception as e:
            st.warning(f"GW shot-chart PDF uploaded, but couldn't parse shot zones: {e}")

    if opp_pdf is not None:
        try:
            op_zones = parse_cbb_team_pdf_zones_both(opp_pdf)
            op_df = _apply_shot_zone_overrides(op_df, op_zones)
            st.success("Opponent shot-chart PDF parsed: applied shot-mix + zone FG% overrides.")
        except Exception as e:
            st.warning(f"Opponent shot-chart PDF uploaded, but couldn't parse shot zones: {e}")

    # KenPom scaling to set expected PPP
    gw_target_eff = _expected_adj_eff(gw_adj_o, op_adj_d) / 100.0
    op_target_eff = _expected_adj_eff(op_adj_o, gw_adj_d) / 100.0

    gw_df = _apply_kenpom_scaling(gw_df, target_ppp=gw_target_eff)
    op_df = _apply_kenpom_scaling(op_df, target_ppp=op_target_eff)

    kenpom = {
        "gw_tempo": gw_tempo,
        "op_tempo": op_tempo,
    }

    rng = np.random.default_rng(7)  # stable results per run; change if you want different each click

    scores_gw = []
    scores_op = []
    poss_list = []
    gw_boxes = []
    op_boxes = []

    progress = st.progress(0.0)
    # Chunk loop for UI responsiveness
    chunk = 250
    done = 0
    while done < nsims:
        n = min(chunk, nsims - done)
        for _ in range(n):
            team_res, player_res = _simulate_one_game(gw_df, op_df, kenpom, rng)
            scores_gw.append(team_res["GW"])
            scores_op.append(team_res["OPP"])
            poss_list.append(team_res["poss"])
            gw_boxes.append(player_res["GW"])
            op_boxes.append(player_res["OPP"])
        done += n
        progress.progress(done / nsims)

    scores_gw = np.array(scores_gw, dtype=float)
    scores_op = np.array(scores_op, dtype=float)
    spread = scores_gw - scores_op

    st.markdown("### Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg score", f"{gw_team_default} {scores_gw.mean():.1f} — {op_team_default} {scores_op.mean():.1f}")
    r2.metric("Avg spread (GW - Opp)", f"{spread.mean():.1f}")
    r3.metric("GW win %", f"{(spread > 0).mean()*100:.1f}%")
    r4.metric("Avg possessions", f"{np.mean(poss_list):.1f}")

    # Show a quick distribution table
    dist = pd.DataFrame({
        "Metric": ["10th pct", "25th pct", "Median", "75th pct", "90th pct"],
        "GW": np.percentile(scores_gw, [10,25,50,75,90]),
        "Opp": np.percentile(scores_op, [10,25,50,75,90]),
        "Spread": np.percentile(spread, [10,25,50,75,90]),
    })
    st.dataframe(dist, use_container_width=True, hide_index=True)

    st.markdown("### Average box score (rounded)")
    gw_avg = _aggregate_boxscores(gw_boxes)
    op_avg = _aggregate_boxscores(op_boxes)
    gw_box_df = _round_boxscore(gw_avg)
    op_box_df = _round_boxscore(op_avg)

    cbb1, cbb2 = st.columns(2)
    with cbb1:
        st.markdown(f"#### {gw_team_default}")
        st.dataframe(gw_box_df, use_container_width=True, hide_index=True)
    with cbb2:
        st.markdown(f"#### {op_team_default}")
        st.dataframe(op_box_df, use_container_width=True, hide_index=True)

    # Team totals row
    def totals(df_box: pd.DataFrame) -> Dict:
        if df_box.empty:
            return {}
        def parse_made_att(s):
            try:
                m,a = s.split("-")
                return int(m), int(a)
            except Exception:
                return 0,0
        fgm = fga = tpm = tpa = ftm = fta = 0
        for s in df_box["FG"]:
            m,a = parse_made_att(s); fgm += m; fga += a
        for s in df_box["3PT"]:
            m,a = parse_made_att(s); tpm += m; tpa += a
        for s in df_box["FT"]:
            m,a = parse_made_att(s); ftm += m; fta += a
        return {
            "MIN": int(df_box["MIN"].sum()),
            "PTS": int(df_box["PTS"].sum()),
            "FG": f"{fgm}-{fga}",
            "3PT": f"{tpm}-{tpa}",
            "FT": f"{ftm}-{fta}",
            "REB": int(df_box["REB"].sum()),
            "AST": int(df_box["AST"].sum()),
            "TO": int(df_box["TO"].sum()),
            "STL": int(df_box["STL"].sum()),
            "BLK": int(df_box["BLK"].sum()),
            "OREB": int(df_box["OREB"].sum()),
            "DREB": int(df_box["DREB"].sum()),
            "PF": int(df_box["PF"].sum()),
        }

    st.markdown("### Team totals (from the rounded box scores)")
    t1, t2 = st.columns(2)
    with t1:
        st.write(totals(gw_box_df))
    with t2:
        st.write(totals(op_box_df))

    st.markdown("### Notes / assumptions")
    st.caption(
        "This is an intentionally fast, possession-level Monte Carlo model. "
        "It uses player season rates from your CSVs (usage proxy via possessions per minute, turnovers, shot mix, shooting %, FTA/FGA, rebounds) "
        "and then lightly scales shooting/turnovers so team PPP roughly matches the KenPom expectation. "
        "Next step if you want more realism: add shot-profile PDFs (zone rates) and lineup/5-man interaction effects."
    )
