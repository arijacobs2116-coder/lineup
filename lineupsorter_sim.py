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

def _build_team_profile(df_players: pd.DataFrame, minutes_override: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, Dict]:
    """Compute per-possession / per-shot rates used by the simulator."""
    df = df_players.copy()

    # Minutes per game: override (GW) or use mins_pg (opponent)
    if minutes_override:
        df["mins_sim"] = df["player"].map(minutes_override).fillna(0.0)
    else:
        df["mins_sim"] = df["mins_pg"].fillna(0.0)

    # Scale minutes so total player-minutes = 200 (40*5)
    total_mins = df["mins_sim"].sum()
    if total_mins <= 0:
        df["mins_sim"] = 0.0
        total_mins = 0.0
    else:
        scale = 200.0 / total_mins
        df["mins_sim"] = df["mins_sim"] * scale

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
    df["fga_rate"] = (df["fga"] / df["poss"].replace(0, np.nan)).fillna(0.0).clip(0, 3)  # can exceed 1 w/ ORBs; that's ok
    df["three_share"] = (df["fga3"] / df["fga"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

    df["p2"] = (df["fgm2"] / df["fga2"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["p3"] = (df["fgm3"] / df["fga3"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["pft"] = (df["ftm"] / df["fta"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

    # Approximate FT attempts per FGA (shot-based free throws; caps keep it sane)
    df["fta_per_fga"] = (df["fta"] / df["fga"].replace(0, np.nan)).fillna(0.0).clip(0, 2)

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

    def run_team_possessions(team_df, opp_df, team_box, orb_prob):
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

        for _ in range(poss):
            shooter_i = int(rng.choice(len(players), p=poss_share))
            shooter = players[shooter_i]

            # Turnover?
            if rng.random() < to_rate[shooter_i]:
                team_box[shooter]["TO"] += 1
                # Attribute steal to opponent (team total only; optional)
                if rng.random() < 0.55 and len(opp_df) > 0:
                    stl_i = int(rng.choice(len(opp_df), p=opp_df["stl_share"].to_numpy()))
                    stl_p = opp_df["player"].iloc[stl_i]
                    # if this opp player exists in opponent box of the outer scope, it will be updated by caller
                continue

            # Shot type
            is_three = rng.random() < three_share[shooter_i]
            if is_three:
                team_box[shooter]["3PA"] += 1
                team_box[shooter]["FGA"] += 1
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
                    # Rebound
                    if rng.random() < orb_prob:
                        r_i = pick(orb_share)
                        team_box[players[r_i]]["OREB"] += 1
                        team_box[players[r_i]]["REB"] += 1
                        # One quick second-chance 2PT
                        team_box[players[r_i]]["FGA"] += 1
                        made2 = rng.random() < p2[r_i]
                        if made2:
                            team_box[players[r_i]]["FGM"] += 1
                            team_box[players[r_i]]["PTS"] += 2
                            pts += 2
                            if rng.random() < 0.52 and len(players) > 1:
                                a_i = pick(ast_share, exclude=r_i)
                                team_box[players[a_i]]["AST"] += 1
                        else:
                            # Defensive rebound credited later via shares (simple)
                            pass
            else:
                team_box[shooter]["FGA"] += 1
                made = rng.random() < p2[shooter_i]
                if made:
                    team_box[shooter]["FGM"] += 1
                    team_box[shooter]["PTS"] += 2
                    pts += 2
                    if rng.random() < 0.55 and len(players) > 1:
                        a_i = pick(ast_share, exclude=shooter_i)
                        team_box[players[a_i]]["AST"] += 1
                else:
                    # Block sometimes
                    if rng.random() < 0.08 and len(opp_df) > 0:
                        blk_i = int(rng.choice(len(opp_df), p=opp_df["blk_share"].to_numpy()))
                        blk_p = opp_df["player"].iloc[blk_i]
                    # Rebound
                    if rng.random() < orb_prob:
                        r_i = pick(orb_share)
                        team_box[players[r_i]]["OREB"] += 1
                        team_box[players[r_i]]["REB"] += 1
                        # Putback
                        team_box[players[r_i]]["FGA"] += 1
                        made2 = rng.random() < min(0.75, p2[r_i] + 0.1)
                        if made2:
                            team_box[players[r_i]]["FGM"] += 1
                            team_box[players[r_i]]["PTS"] += 2
                            pts += 2

            # Free throws off the main shot (rough)
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

    gw_pts = run_team_possessions(gw, op, gw_box, gw_orb_prob)
    op_pts = run_team_possessions(op, gw, op_box, op_orb_prob)

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

        # PF: use observed pf/min and simulated minutes
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

    if gw_csv is None or opp_csv is None:
        st.info("Upload both CSVs to enable simulation.")
        st.stop()

    try:
        gw_raw = _read_player_stats_csv(gw_csv)
        op_raw = _read_player_stats_csv(opp_csv)
    except Exception as e:
        st.error(f"Could not read one of the CSV files: {e}")
        st.stop()

    # Team labels
    gw_team_default = str(gw_raw["team"].iloc[0]) if "team" in gw_raw.columns and len(gw_raw) else "George Washington"
    op_team_default = str(op_raw["team"].iloc[0]) if "team" in op_raw.columns and len(op_raw) else "Opponent"

    st.markdown("### KenPom team inputs")
    kp1, kp2, kp3 = st.columns(3)
    with kp1:
        gw_adj_o = st.number_input("GW AdjO", value=118.2, step=0.1, format="%.1f")
        op_adj_o = st.number_input("Opponent AdjO", value=110.0, step=0.1, format="%.1f")
    with kp2:
        gw_adj_d = st.number_input("GW AdjD", value=107.7, step=0.1, format="%.1f")
        op_adj_d = st.number_input("Opponent AdjD", value=107.7, step=0.1, format="%.1f")
    with kp3:
        gw_tempo = st.number_input("GW AdjTempo", value=71.6, step=0.1, format="%.1f")
        op_tempo = st.number_input("Opponent AdjTempo", value=69.2, step=0.1, format="%.1f")

    st.markdown("### GW minutes (you control these)")
    gw_min_df = gw_raw[["player", "mins_pg"]].copy()
    gw_min_df["mins_pg"] = gw_min_df["mins_pg"].fillna(0.0)
    gw_min_df = gw_min_df.sort_values("mins_pg", ascending=False).reset_index(drop=True)

    # Editable minutes table
    gw_min_edit = st.data_editor(
        gw_min_df.rename(columns={"mins_pg": "MINUTES"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "player": st.column_config.TextColumn("Player", disabled=True),
            "MINUTES": st.column_config.NumberColumn("Minutes", min_value=0.0, max_value=40.0, step=0.5),
        },
        key="gw_minutes_editor",
    )

    minutes_override = {r["player"]: float(r["MINUTES"]) for _, r in gw_min_edit.iterrows()}

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
