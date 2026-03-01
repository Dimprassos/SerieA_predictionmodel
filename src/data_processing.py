from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# project root = folder one level above /src
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 1) First choice: Pinnacle closing
CLOSING_COLS = ("PSCH", "PSCD", "PSCA")

# 2) Otherwise: average across available bookmakers (common football-data set)
# Add/remove triples εδώ αν βρεις άλλα στα CSV σου
BOOK_TRIPLES = [
    ("B365H", "B365D", "B365A"),
    ("BWH", "BWD", "BWA"),
    ("GBH", "GBD", "GBA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("SJH", "SJD", "SJA"),
    ("VCH", "VCD", "VCA"),
    ("BSH", "BSD", "BSA"),
]

REQUIRED_BASE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def _valid_odds_triplet(oh, od, oa) -> bool:
    """Check odds are finite and > 1 (rough sanity)."""
    if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
        return False
    if oh <= 1.0001 or od <= 1.0001 or oa <= 1.0001:
        return False
    return True


def _odds_to_fair_probs(oh, od, oa) -> np.ndarray:
    """
    Convert bookmaker odds to implied probabilities with overround removed.
    """
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa], dtype=float)
    s = inv.sum()
    if s <= 0 or not np.isfinite(s):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def _fair_probs_to_odds(p: np.ndarray) -> tuple[float, float, float]:
    """
    Convert fair probabilities to odds.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        return (np.nan, np.nan, np.nan)
    p = np.clip(p, 1e-12, 1.0)
    # ensure sums to 1 (safety)
    p = p / p.sum()
    return (float(1.0 / p[0]), float(1.0 / p[1]), float(1.0 / p[2]))


def _pick_best_or_avg_odds_row(row: pd.Series, available_cols: set[str]) -> tuple[float, float, float]:
    """
    Rule:
      1) Use PSCH/PSCD/PSCA if valid
      2) Else average fair probs across available bookmaker triples
      3) Else NaN
    """
    # 1) Closing Pinnacle
    if all(c in available_cols for c in CLOSING_COLS):
        oh, od, oa = row.get(CLOSING_COLS[0]), row.get(CLOSING_COLS[1]), row.get(CLOSING_COLS[2])
        oh, od, oa = pd.to_numeric(oh, errors="coerce"), pd.to_numeric(od, errors="coerce"), pd.to_numeric(oa, errors="coerce")
        if _valid_odds_triplet(oh, od, oa):
            return float(oh), float(od), float(oa)

    # 2) Average across books (in fair probability space)
    probs_list = []
    for h, d, a in BOOK_TRIPLES:
        if h in available_cols and d in available_cols and a in available_cols:
            oh = pd.to_numeric(row.get(h), errors="coerce")
            od = pd.to_numeric(row.get(d), errors="coerce")
            oa = pd.to_numeric(row.get(a), errors="coerce")
            if _valid_odds_triplet(oh, od, oa):
                probs_list.append(_odds_to_fair_probs(float(oh), float(od), float(oa)))

    if len(probs_list) == 0:
        return (np.nan, np.nan, np.nan)

    p_avg = np.nanmean(np.vstack(probs_list), axis=0)
    return _fair_probs_to_odds(p_avg)


def load_and_merge_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "raw"
    files = list(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}\n"
            f"Put your Serie A season CSVs there (e.g., I1_2012.csv, ...)."
        )

    dfs: list[pd.DataFrame] = []
    for file in sorted(files):
        df = pd.read_csv(file)

        missing = [c for c in REQUIRED_BASE if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{file.name}] Missing required columns {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Keep base + any odds columns that exist
        want_cols = set(REQUIRED_BASE)
        for c in CLOSING_COLS:
            if c in df.columns:
                want_cols.add(c)
        for h, d, a in BOOK_TRIPLES:
            if h in df.columns and d in df.columns and a in df.columns:
                want_cols.update([h, d, a])

        df = df[list(want_cols)].copy()

        # rename base
        df = df.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        })

        # parse dates (football-data formats)
        dt1 = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        dt2 = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df["date"] = dt1.fillna(dt2)

        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(df.loc[mask, "date"], dayfirst=True, errors="coerce")

        df = df.dropna(subset=["date"])

        # enforce numeric goals
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df = df.dropna(subset=["home_goals", "away_goals"])
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        # build unified odds columns
        available_cols = set(df.columns)

        odds = df.apply(lambda r: _pick_best_or_avg_odds_row(r, available_cols), axis=1, result_type="expand")
        odds.columns = ["odds_home", "odds_draw", "odds_away"]

        df = pd.concat([df, odds], axis=1)

        # force numeric odds
        df["odds_home"] = pd.to_numeric(df["odds_home"], errors="coerce")
        df["odds_draw"] = pd.to_numeric(df["odds_draw"], errors="coerce")
        df["odds_away"] = pd.to_numeric(df["odds_away"], errors="coerce")

        dfs.append(df[["date", "home_team", "away_team", "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"]])

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out