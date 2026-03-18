import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.artifacts import load_json_if_exists, load_pickle_if_exists
from src.data_processing import load_league_data
from src.poisson_model import (
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs_dc,
    top_k_scorelines_dc,
)
from src.elo import expected_score, match_result, margin_multiplier
from src.calibration import temperature_scale_probs
from src.tuning import blend_probabilities

EXPERIMENT_NAME = "baseline_xgboost_v1"
ARTIFACTS_DIR = Path(r"C:\Users\GamingPC2\SerieA_predictionmodel\artifacts")
PARAMS_FILE = ARTIFACTS_DIR / f"best_params_{EXPERIMENT_NAME}.json"
MODEL_FILE = ARTIFACTS_DIR / f"meta_model_{EXPERIMENT_NAME}.json"
MLP_FILE = ARTIFACTS_DIR / f"mlp_model_{EXPERIMENT_NAME}.pkl"
MLP_META_FILE = ARTIFACTS_DIR / f"best_mlp_{EXPERIMENT_NAME}.json"
BLEND_FILE = ARTIFACTS_DIR / f"best_blend_{EXPERIMENT_NAME}.json"
    

def market_probs_from_odds_row(odds_h, odds_d, odds_a):
    if not (np.isfinite(odds_h) and np.isfinite(odds_d) and np.isfinite(odds_a)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if odds_h <= 1.0001 or odds_d <= 1.0001 or odds_a <= 1.0001:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    inv = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    s = inv.sum()
    if s <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def safe_logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def build_meta_features_single(model_probs, market_probs, elo_h, elo_a, lam_h, lam_a, mom_h, mom_a, mom_diff):
    pm = np.asarray(model_probs, dtype=float)
    pk = np.asarray(market_probs, dtype=float)
    if not np.isfinite(pk).all():
        pk = pm
    feats = [
        safe_logit(pm[0]), safe_logit(pm[1]), safe_logit(pm[2]),
        safe_logit(pk[0]), safe_logit(pk[1]), safe_logit(pk[2]),
        (elo_h - elo_a) / 400.0,
        lam_h + lam_a,
        lam_h - lam_a,
        mom_h,
        mom_a,
        mom_diff
    ]
    return np.array([feats], dtype=float)


def load_artifacts():
    print("Loading models and parameters...")
    params = load_json_if_exists(PARAMS_FILE)
    if params is None:
        sys.exit("Error: Parameters file not found. Run main.py first.")

    if not MODEL_FILE.exists():
        sys.exit("Error: XGBoost model not found. Run main.py first.")
    meta_model = XGBClassifier()
    meta_model.load_model(str(MODEL_FILE))

    mlp_model = load_pickle_if_exists(MLP_FILE)
    mlp_meta = load_json_if_exists(MLP_META_FILE)
    blend_cfg = load_json_if_exists(BLEND_FILE)
    return params, meta_model, mlp_model, mlp_meta, blend_cfg


def get_league_state(league_name, params):
    df = load_league_data(league_name)
    df = df[df["is_played"] == True].sort_values("date").reset_index(drop=True)

    l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
        df, decay=params[league_name]["decay"]
    )

    ratings = {}
    elo_history = {}
    K = params[league_name]["K"]
    ha = params[league_name]["ha"]

    def get_init(r):
        if len(r) >= 5:
            return sum(sorted(r.values())[:3]) / 3.0
        return 1500.0

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        init_r = get_init(ratings)
        rh = ratings.get(h, init_r)
        ra = ratings.get(a, init_r)
        if h not in ratings:
            ratings[h] = rh
        if a not in ratings:
            ratings[a] = ra
        if h not in elo_history:
            elo_history[h] = []
        if a not in elo_history:
            elo_history[a] = []

        exp_h = expected_score(rh + ha, ra)
        sh, sa = match_result(hg, ag)
        mult = margin_multiplier(hg - ag)
        
        new_rh = rh + (K * mult) * (sh - exp_h)
        new_ra = ra + (K * mult) * (sa - (1 - exp_h))
        
        ratings[h] = new_rh
        ratings[a] = new_ra
        
        elo_history[h].append(new_rh)
        elo_history[a].append(new_ra)

    return {
        "ratings": ratings,
        "elo_history": elo_history,
        "att_h": att_h,
        "def_h": def_h,
        "att_a": att_a,
        "def_a": def_a,
        "l_avg_h": l_avg_h,
        "l_avg_a": l_avg_a,
        "params": params[league_name],
    }


def predict_custom_match(home, away, odds_h, odds_d, odds_a, state, meta_model, mlp_model, mlp_meta, blend_cfg):
    p = state["params"]
    elo_h = state["ratings"].get(home, 1500.0)
    elo_a = state["ratings"].get(away, 1500.0)

    def get_momentum(team, current_r, history, window=4):
        if team not in history or len(history[team]) < window:
            return 0.0
        return current_r - history[team][-window]

    mom_h = get_momentum(home, elo_h, state["elo_history"], window=4) / 400.0
    mom_a = get_momentum(away, elo_a, state["elo_history"], window=4) / 400.0
    mom_diff = mom_h - mom_a

    lam_h, lam_a = predict_lambdas_home_away(
        home, away,
        state["l_avg_h"], state["l_avg_a"],
        state["att_h"], state["def_h"],
        state["att_a"], state["def_a"],
    )
    lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=p["beta"])
    pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=p["rho"])
    model_probs_raw = np.array([[pH, pD, pA]])
    model_probs_cal = temperature_scale_probs(model_probs_raw, p["T"])[0]

    if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        mkt_probs = market_probs_from_odds_row(odds_h, odds_d, odds_a)
    else:
        mkt_probs = model_probs_cal

    X = build_meta_features_single(model_probs_cal, mkt_probs, elo_h, elo_a, lam_h, lam_a, mom_h, mom_a, mom_diff)
    meta_probs = meta_model.predict_proba(X)[0]

    if mlp_model is not None:
        mlp_probs_raw = mlp_model.predict_proba(X)
        if mlp_meta is not None and "temperature" in mlp_meta:
            mlp_probs = temperature_scale_probs(mlp_probs_raw, float(mlp_meta["temperature"]))[0]
        else:
            mlp_probs = mlp_probs_raw[0]
    else:
        mlp_probs = meta_probs.copy()

    if blend_cfg is not None:
        ens_probs = blend_probabilities(
            blend_cfg["weights"],
            {
                "base": model_probs_cal.reshape(1, -1),
                "market": mkt_probs.reshape(1, -1),
                "xgb": meta_probs.reshape(1, -1),
                "mlp": mlp_probs.reshape(1, -1),
            },
        )[0]
    else:
        ens_probs = meta_probs.copy()

    top_scores = top_k_scorelines_dc(lam_h, lam_a, p["rho"], k=3)
    return {
        "base": model_probs_cal,
        "market": mkt_probs,
        "meta": meta_probs,
        "mlp": mlp_probs,
        "ensemble": ens_probs,
        "elo": (elo_h, elo_a),
        "xg": (lam_h, lam_a),
        "scores": top_scores,
    }


def pick_team(teams, prompt):
    raw = input(prompt).strip()
    if not raw:
        return None
    matches = [t for t in teams if raw.lower() in t.lower()]
    if not matches:
        print("Team not found.")
        return None
    if len(matches) == 1:
        print(f"Selected: {matches[0]}")
        return matches[0]
    print("Possible matches:")
    for i, t in enumerate(matches[:10], 1):
        print(f"{i}. {t}")
    try:
        idx = int(input("Choose number: ")) - 1
        return matches[idx]
    except Exception:
        return matches[0]


def main():
    print("=== Interactive Match Predictor ===")
    params, meta_model, mlp_model, mlp_meta, blend_cfg = load_artifacts()
    leagues = ["england", "spain", "italy", "germany", "france"]

    while True:
        print("\nAvailable Leagues:")
        for i, l in enumerate(leagues, 1):
            print(f"{i}. {l}")
        try:
            choice = int(input("\nSelect League (number) or 0 to exit: "))
        except ValueError:
            continue
        if choice == 0:
            break
        if choice < 1 or choice > len(leagues):
            continue
        league = leagues[choice - 1]
        print(f"Loading data for {league}...")
        state = get_league_state(league, params)
        teams = sorted(state["ratings"].keys())

        while True:
            print(f"\n--- {league.upper()} Prediction ---")
            home_real = pick_team(teams, "Home Team Name (part or full, enter to go back): ")
            if home_real is None:
                break
            away_real = pick_team(teams, "Away Team Name (part or full): ")
            if away_real is None:
                continue
            if home_real == away_real:
                print("Home and Away teams cannot be the same.")
                continue
            try:
                odds_str = input("Enter Odds (Home Draw Away), e.g. '1.90 3.50 4.00' (enter to skip): ").strip()
                if odds_str:
                    oh, od, oa = map(float, odds_str.split())
                else:
                    oh, od, oa = 0.0, 0.0, 0.0
            except Exception:
                print("Invalid odds format.")
                continue

            res = predict_custom_match(home_real, away_real, oh, od, oa, state, meta_model, mlp_model, mlp_meta, blend_cfg)

            print("\n--- Prediction Summary ---")
            print(f"Match: {home_real} vs {away_real}")
            print(f"Elo: {res['elo'][0]:.1f} vs {res['elo'][1]:.1f}")
            print(f"Expected Goals: {res['xg'][0]:.3f} - {res['xg'][1]:.3f}")
            print(f"Base probs     : H={res['base'][0]:.3f} D={res['base'][1]:.3f} A={res['base'][2]:.3f}")
            print(f"Market probs   : H={res['market'][0]:.3f} D={res['market'][1]:.3f} A={res['market'][2]:.3f}")
            print(f"XGBoost probs  : H={res['meta'][0]:.3f} D={res['meta'][1]:.3f} A={res['meta'][2]:.3f}")
            print(f"MLP probs      : H={res['mlp'][0]:.3f} D={res['mlp'][1]:.3f} A={res['mlp'][2]:.3f}")
            print(f"Ensemble probs : H={res['ensemble'][0]:.3f} D={res['ensemble'][1]:.3f} A={res['ensemble'][2]:.3f}")
            pick = ["H", "D", "A"][int(np.argmax(res['ensemble']))]
            print(f"Final Pick     : {pick}")
            print("Top scorelines:")
            for (hg, ag), psc in res["scores"]:
                print(f"  {hg}-{ag}  ({psc:.3f})")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
