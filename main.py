import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_processing import load_league_data
from src.poisson_model import (
    fit_team_strengths,
    fit_team_strengths_weighted,
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs,
    match_outcome_probs_dc,
    top_k_scorelines_dc,
)
from src.elo import compute_elo_ratings
from src.calibration import fit_temperature, temperature_scale_probs
from src.metrics import multiclass_brier, top_label_ece


# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_NAME = "baseline_xgboost_v1"

USE_CACHED_ARTIFACTS = True
FORCE_RETUNE_LEAGUES = False     # ξαναβρίσκει K, ha, beta, decay, rho, T
FORCE_RETUNE_META = False        # ξαναβρίσκει XGBoost hyperparams
FORCE_REFIT_META_MODEL = False   # ξανακάνει fit το τελικό meta model

TRAIN_CUT = "2024-07-01"
TEST_CUT = "2025-07-01"

LEAGUES = ["england", "spain", "italy", "germany", "france"]

ARTIFACTS_DIR = Path("artifacts")
PARAMS_FILE = ARTIFACTS_DIR / f"best_params_{EXPERIMENT_NAME}.json"
META_FILE = ARTIFACTS_DIR / f"best_meta_{EXPERIMENT_NAME}.json"
MODEL_FILE = ARTIFACTS_DIR / f"meta_model_{EXPERIMENT_NAME}.json"


# ============================================================
# JSON helpers
# ============================================================
def load_json_if_exists(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_cached_params():
    return load_json_if_exists(PARAMS_FILE)


def save_cached_params(params):
    save_json(PARAMS_FILE, params)


def load_cached_meta():
    return load_json_if_exists(META_FILE)


def save_cached_meta(meta_params):
    save_json(META_FILE, meta_params)


# ============================================================
# Market implied probabilities
# ============================================================
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


def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    y = []
    for _, r in df.iterrows():
        if r["home_goals"] > r["away_goals"]:
            y.append(0)
        elif r["home_goals"] == r["away_goals"]:
            y.append(1)
        else:
            y.append(2)
    return np.array(y, dtype=int)


# ============================================================
# Streaming block-walk-forward
# ============================================================
def streaming_block_probs_home_away(
    predict_df,
    full_df,
    beta,
    rho,
    decay,
    K,
    home_adv,
    init_rating=1500.0,
    max_goals=10,
):
    from src.elo import expected_score, match_result, margin_multiplier

    probs_model = []
    probs_mkt = []
    y_true = []
    aux = []

    predict_df = predict_df.sort_values("date")
    full_df = full_df.sort_values("date")

    predict_dates = sorted(predict_df["date"].unique())
    if len(predict_dates) == 0:
        return (
            np.zeros((0, 3)),
            np.zeros((0,), dtype=int),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
        )

    first_predict_date = predict_dates[0]

    ratings = {}
    history_matches = full_df[full_df["date"] < first_predict_date]

    def get_dynamic_init(current_ratings):
        if len(current_ratings) >= 5:
            bottom_elos = sorted(current_ratings.values())[:3]
            return float(sum(bottom_elos) / len(bottom_elos))
        return float(init_rating)

    def update_ratings(matches_batch, current_ratings):
        for _, m in matches_batch.iterrows():
            h, a = m["home_team"], m["away_team"]

            dyn_init = get_dynamic_init(current_ratings)
            r_h = current_ratings.get(h, dyn_init)
            r_a = current_ratings.get(a, dyn_init)

            if h not in current_ratings:
                current_ratings[h] = r_h
            if a not in current_ratings:
                current_ratings[a] = r_a

            exp_h = expected_score(r_h + home_adv, r_a)
            s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
            mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

            current_ratings[h] = r_h + (K * mult) * (s_h - exp_h)
            current_ratings[a] = r_a + (K * mult) * (s_a - (1 - exp_h))
        return current_ratings

    ratings = update_ratings(history_matches, ratings)

    for d in predict_dates:
        day_matches = predict_df[predict_df["date"] == d]
        past_matches = full_df[full_df["date"] < d]

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            past_matches, decay=decay
        )

        for _, row in day_matches.iterrows():
            ht, at = row["home_team"], row["away_team"]

            dyn_init = get_dynamic_init(ratings)
            elo_h = ratings.get(ht, dyn_init)
            elo_a = ratings.get(at, dyn_init)

            lam_h, lam_a = predict_lambdas_home_away(
                ht, at,
                l_avg_h, l_avg_a,
                att_h, def_h,
                att_a, def_a,
            )

            lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=beta)

            pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=max_goals)
            probs_model.append([pH, pD, pA])

            pm = market_probs_from_odds_row(row["odds_home"], row["odds_draw"], row["odds_away"])
            probs_mkt.append(pm.tolist())

            if row["home_goals"] > row["away_goals"]:
                y_true.append(0)
            elif row["home_goals"] == row["away_goals"]:
                y_true.append(1)
            else:
                y_true.append(2)

            elo_diff = (elo_h - elo_a) / 400.0
            total_xg = lam_h + lam_a
            xg_diff = lam_h - lam_a
            aux.append([elo_diff, total_xg, xg_diff])

        ratings = update_ratings(day_matches, ratings)

    return (
        np.array(probs_model, dtype=float),
        np.array(y_true, dtype=int),
        np.array(probs_mkt, dtype=float),
        np.array(aux, dtype=float),
    )


def build_meta_features(p_model, p_mkt, aux):
    p_mkt_fixed = p_mkt.copy()
    for i in range(len(p_mkt_fixed)):
        if not np.isfinite(p_mkt_fixed[i]).all():
            p_mkt_fixed[i] = p_model[i]

    X = []
    for i in range(len(p_model)):
        pm = p_model[i]
        pk = p_mkt_fixed[i]
        feats = [
            safe_logit(pm[0]), safe_logit(pm[1]), safe_logit(pm[2]),
            safe_logit(pk[0]), safe_logit(pk[1]), safe_logit(pk[2]),
            aux[i, 0], aux[i, 1], aux[i, 2],
        ]
        X.append(feats)
    return np.array(X, dtype=float)


def time_split_val(val_df: pd.DataFrame):
    val_sorted = val_df.sort_values("date").reset_index(drop=True)
    mid = len(val_sorted) // 2
    return val_sorted.iloc[:mid].copy(), val_sorted.iloc[mid:].copy()


def simulate_value_betting(probs, raw_odds, y_true, edge_threshold=0.05):
    stats = {
        "Home (1)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
        "Draw (X)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
        "Away (2)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
    }

    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        evs = [
            (p_h * o_h - 1, 0, o_h, "Home (1)"),
            (p_d * o_d - 1, 1, o_d, "Draw (X)"),
            (p_a * o_a - 1, 2, o_a, "Away (2)"),
        ]

        best_ev, choice, odds_taken, label = max(evs)

        if best_ev > edge_threshold:
            stats[label]["count"] += 1
            stats[label]["invested"] += 1.0
            stats[label]["odds_sum"] += odds_taken

            if choice == y_true[i]:
                stats[label]["wins"] += 1
                stats[label]["return"] += odds_taken

    print(f"\n{'Market Segment':<15} | {'Bets':<5} | {'Win%':<7} | {'ROI%':<8}")
    print("-" * 45)

    total_bets = 0
    total_wins = 0
    total_inv = 0
    total_ret = 0
    total_odds_sum = 0

    for label, s in stats.items():
        if s["count"] > 0:
            win_pc = (s["wins"] / s["count"]) * 100
            roi = ((s["return"] - s["invested"]) / s["invested"]) * 100
            print(f"{label:<15} | {s['count']:<5} | {win_pc:>6.1f}% | {roi:>7.2f}%")

            total_bets += s["count"]
            total_wins += s["wins"]
            total_inv += s["invested"]
            total_ret += s["return"]
            total_odds_sum += s["odds_sum"]

    final_profit = total_ret - total_inv
    final_roi = (final_profit / total_inv * 100) if total_inv > 0 else 0
    avg_odds = (total_odds_sum / total_bets) if total_bets > 0 else 0

    print("-" * 45)
    print(f"{'TOTAL':<15} | {int(total_inv):<5} | {'-':>7} | {final_roi:>7.2f}%")

    return total_bets, total_wins, final_profit, final_roi, avg_odds


def get_current_or_next_matchday_fixtures(df_league: pd.DataFrame, max_window_days: int = 4):
    today = pd.Timestamp.now().normalize()

    future_df = df_league[df_league["is_played"] == False].copy()
    future_df = future_df.sort_values("date")

    if future_df.empty:
        return pd.DataFrame(), None

    candidate_df = future_df[future_df["date"] >= today].copy()

    if candidate_df.empty:
        candidate_df = future_df.copy()

    if candidate_df.empty:
        return pd.DataFrame(), None

    matchday_start = candidate_df["date"].min()
    matchday_end = matchday_start + pd.Timedelta(days=max_window_days)

    fixtures = candidate_df[
        (candidate_df["date"] >= matchday_start) &
        (candidate_df["date"] <= matchday_end)
    ].copy()

    return fixtures, matchday_start


# ============================================================
# Per-league tuning
# ============================================================
def tune_league_params(train_fit, val, full_played_df):
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.002, 0.003]

    best = None
    print("\n--- Elo & Beta Tuning ---")

    from src.poisson_model import predict_lambdas

    for K in Ks:
        for ha in home_advs:
            for b in betas:
                elo_pairs = compute_elo_ratings(train_fit, K=K, home_adv=ha, use_margin=True)

                tmp = train_fit.copy()
                tmp["elo_home"], tmp["elo_away"] = zip(*elo_pairs)

                l_avg_h, l_avg_a, att, dfn = fit_team_strengths(tmp)

                full_tmp = pd.concat([train_fit, val], ignore_index=True).sort_values("date")
                elo_full = compute_elo_ratings(full_tmp, K=K, home_adv=ha, use_margin=True)
                full_tmp["elo_home"], full_tmp["elo_away"] = zip(*elo_full)
                val_part = full_tmp.iloc[len(train_fit):]

                probs = []
                y = []

                for _, row in val_part.iterrows():
                    lh, la = predict_lambdas(
                        row["home_team"], row["away_team"],
                        l_avg_h, l_avg_a, att, dfn
                    )
                    lh, la = apply_elo_to_lambdas(
                        lh, la, row["elo_home"], row["elo_away"], beta=b
                    )

                    probs.append(match_outcome_probs(lh, la))

                    if row["home_goals"] > row["away_goals"]:
                        y.append(0)
                    elif row["home_goals"] == row["away_goals"]:
                        y.append(1)
                    else:
                        y.append(2)

                ll = log_loss(np.array(y), np.array(probs))
                if best is None or ll < best[0]:
                    best = (ll, K, ha, b)

    _, best_K, best_ha, best_beta = best
    print(f"Best Config: K={best_K}, ha={best_ha}, beta={best_beta}")

    print("--- Tuning Time Decay ---")
    best_decay = None
    best_decay_ll = float("inf")
    y_val = labels_from_df(val)

    elo_full = compute_elo_ratings(
        full_played_df[full_played_df["date"] < val["date"].max() + pd.Timedelta(days=1)],
        K=best_K, home_adv=best_ha, use_margin=True
    )
    tmp_df = full_played_df[full_played_df["date"] < val["date"].max() + pd.Timedelta(days=1)].copy()
    tmp_df["elo_home"], tmp_df["elo_away"] = zip(*elo_full)
    val_part = tmp_df[tmp_df["date"].isin(val["date"])].copy()

    from src.poisson_model import predict_lambdas

    for d in decays:
        l_avg_h, l_avg_a, att, dfn = fit_team_strengths_weighted(train_fit, decay=d)

        probs = []
        for _, row in val_part.iterrows():
            lh, la = predict_lambdas(
                row["home_team"], row["away_team"],
                l_avg_h, l_avg_a, att, dfn
            )
            lh, la = apply_elo_to_lambdas(
                lh, la, row["elo_home"], row["elo_away"], beta=best_beta
            )
            probs.append(match_outcome_probs(lh, la))

        ll = log_loss(y_val, np.array(probs))
        if ll < best_decay_ll:
            best_decay_ll = ll
            best_decay = d

    print(f"Best Decay: {best_decay}")

    print("--- Joint Tuning rho + Temperature ---")
    rho_grid = np.arange(-0.2, 0.21, 0.02)
    best_joint = None

    for rho in rho_grid:
        val_probs_raw, y_val_stream, _, _ = streaming_block_probs_home_away(
            val, full_played_df,
            beta=best_beta, rho=float(rho), decay=best_decay,
            K=best_K, home_adv=best_ha
        )

        T = fit_temperature(val_probs_raw, y_val_stream)
        val_probs_cal = temperature_scale_probs(val_probs_raw, T)
        ll = log_loss(y_val_stream, val_probs_cal)

        if best_joint is None or ll < best_joint[0]:
            best_joint = (ll, float(rho), float(T))

    _, best_rho, best_T = best_joint
    print(f"Best rho={best_rho}, T={best_T}")

    return {
        "K": int(best_K),
        "ha": int(best_ha),
        "beta": float(best_beta),
        "decay": float(best_decay),
        "rho": float(best_rho),
        "T": float(best_T),
    }


# ============================================================
# Main
# ============================================================
def main():
    cached_params = load_cached_params() if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_LEAGUES else None
    cached_meta = load_cached_meta() if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_META else None

    if cached_params is not None:
        print(f"\nUsing cached league params from: {PARAMS_FILE}")
    else:
        if FORCE_RETUNE_LEAGUES:
            print("\nFORCE_RETUNE_LEAGUES=True -> retuning league params.")
        else:
            print(f"\nNo cached league params found for experiment: {EXPERIMENT_NAME}")

    if cached_meta is not None:
        print(f"Using cached XGBoost hyperparams from: {META_FILE}")
    else:
        if FORCE_RETUNE_META:
            print("FORCE_RETUNE_META=True -> retuning XGBoost hyperparams.")
        else:
            print(f"No cached XGBoost hyperparams found for experiment: {EXPERIMENT_NAME}")

    league_best_params = {} if cached_params is None else cached_params

    all_X_early, all_y_early = [], []
    all_X_late, all_y_late = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []

    all_t_probs_model = []
    all_t_mkt_fixed = []
    all_t_raw_odds = []

    # Per-league test data for individual reporting
    per_league_test = {}  # league -> {"y": [], "p_model": [], "p_mkt": []}

    for league in LEAGUES:
        print("\n" + "=" * 50)
        print(f"=== Processing Data: {league.upper()} ===")
        print("=" * 50)

        df_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        df = df_all[df_all["is_played"] == True].copy().reset_index(drop=True)

        if df.empty:
            print(f"Δεν υπάρχουν played matches για {league}.")
            continue

        train_fit = df[df["date"] < TRAIN_CUT].copy()
        val = df[(df["date"] >= TRAIN_CUT) & (df["date"] < TEST_CUT)].copy()
        test = df[df["date"] >= TEST_CUT].copy()

        print(f"Train_fit: {len(train_fit)}, Validation: {len(val)}, Test: {len(test)}")

        if len(train_fit) == 0 or len(val) == 0 or len(test) == 0:
            print(f"Μη επαρκή splits για {league}.")
            continue

        if league in league_best_params and not FORCE_RETUNE_LEAGUES:
            params = league_best_params[league]
            print("\n--- Using cached params ---")
            print(
                f"K={params['K']}, ha={params['ha']}, beta={params['beta']}, "
                f"decay={params['decay']}, rho={params['rho']}, T={params['T']}"
            )
        else:
            params = tune_league_params(train_fit, val, df)
            league_best_params[league] = params

        best_K = params["K"]
        best_ha = params["ha"]
        best_beta = params["beta"]
        best_decay = params["decay"]
        best_rho = params["rho"]
        best_T = params["T"]

        full_df_for_stream = df.copy()

        val_early, val_late = time_split_val(val)

        ve_probs_raw, ve_y, ve_mkt, ve_aux = streaming_block_probs_home_away(
            val_early, full_df_for_stream,
            beta=best_beta, rho=best_rho, decay=best_decay,
            K=best_K, home_adv=best_ha
        )
        vl_probs_raw, vl_y, vl_mkt, vl_aux = streaming_block_probs_home_away(
            val_late, full_df_for_stream,
            beta=best_beta, rho=best_rho, decay=best_decay,
            K=best_K, home_adv=best_ha
        )

        ve_probs_model = temperature_scale_probs(ve_probs_raw, best_T)
        vl_probs_model = temperature_scale_probs(vl_probs_raw, best_T)

        all_X_early.extend(build_meta_features(ve_probs_model, ve_mkt, ve_aux))
        all_y_early.extend(ve_y)
        all_X_late.extend(build_meta_features(vl_probs_model, vl_mkt, vl_aux))
        all_y_late.extend(vl_y)

        v_probs_raw, v_y, v_mkt, v_aux = streaming_block_probs_home_away(
            val, full_df_for_stream,
            beta=best_beta, rho=best_rho, decay=best_decay,
            K=best_K, home_adv=best_ha
        )
        v_probs_model = temperature_scale_probs(v_probs_raw, best_T)

        all_X_val.extend(build_meta_features(v_probs_model, v_mkt, v_aux))
        all_y_val.extend(v_y)

        t_probs_raw, t_y, t_mkt, t_aux = streaming_block_probs_home_away(
            test, full_df_for_stream,
            beta=best_beta, rho=best_rho, decay=best_decay,
            K=best_K, home_adv=best_ha
        )
        t_probs_model = temperature_scale_probs(t_probs_raw, best_T)

        t_mkt_fixed = t_mkt.copy()
        for i in range(len(t_mkt_fixed)):
            if not np.isfinite(t_mkt_fixed[i]).all():
                t_mkt_fixed[i] = t_probs_model[i]

        all_X_test.extend(build_meta_features(t_probs_model, t_mkt_fixed, t_aux))
        all_y_test.extend(t_y)
        all_t_probs_model.extend(t_probs_model)
        all_t_mkt_fixed.extend(t_mkt_fixed)
        all_t_raw_odds.extend(test[["odds_home", "odds_draw", "odds_away"]].values)

        # Save per-league test data for individual metric reporting
        per_league_test[league] = {
            "y":       np.array(t_y, dtype=int),
            "p_model": np.array(t_probs_model, dtype=float),
            "p_mkt":   np.array(t_mkt_fixed, dtype=float),
        }

    if cached_params is None or FORCE_RETUNE_LEAGUES:
        save_cached_params(league_best_params)
        print(f"\nSaved tuned league params to: {PARAMS_FILE}")

    print("\n" + "=" * 50)
    print("=== META-MODEL Evaluation (XGBoost) ===")
    print("=" * 50)

    X_early_arr = np.array(all_X_early)
    y_early_arr = np.array(all_y_early)
    X_late_arr = np.array(all_X_late)
    y_late_arr = np.array(all_y_late)
    X_val_arr = np.array(all_X_val)
    y_val_arr = np.array(all_y_val)
    X_test_arr = np.array(all_X_test)
    y_test_arr = np.array(all_y_test)
    t_probs_model_arr = np.array(all_t_probs_model)
    t_mkt_fixed_arr = np.array(all_t_mkt_fixed)
    raw_odds_arr = np.array(all_t_raw_odds)

    learning_rates = [0.01, 0.05, 0.1]
    max_depths = [3, 4, 5]
    n_estimators_list = [100, 300, 500]

    if cached_meta is not None and not FORCE_RETUNE_META:
        best_lr = cached_meta["learning_rate"]
        best_md = cached_meta["max_depth"]
        best_ne = cached_meta["n_estimators"]
        best_late_ll = cached_meta.get("late_val_logloss", None)

        print("Using cached XGBoost hyperparameters...")
        print(f"Best XGBoost Config -> LR: {best_lr}, Depth: {best_md}, Trees: {best_ne}")
        if best_late_ll is not None:
            print(f"Cached Late VAL LogLoss: {round(best_late_ll, 4)}")
    else:
        best_meta = None

        print("Tuning XGBoost Hyperparameters on Validation Set. Please wait...")
        for lr in learning_rates:
            for md in max_depths:
                for ne in n_estimators_list:
                    meta = XGBClassifier(
                        n_estimators=ne,
                        learning_rate=lr,
                        max_depth=md,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                    )
                    meta.fit(X_early_arr, y_early_arr)
                    late_probs = meta.predict_proba(X_late_arr)
                    late_ll = log_loss(y_late_arr, late_probs)

                    if best_meta is None or late_ll < best_meta[0]:
                        best_meta = (late_ll, lr, md, ne)

        best_late_ll, best_lr, best_md, best_ne = best_meta

        print(f"Best XGBoost Config -> LR: {best_lr}, Depth: {best_md}, Trees: {best_ne}")
        print(f"Late VAL LogLoss: {round(best_late_ll, 4)}")

        save_cached_meta({
            "learning_rate": float(best_lr),
            "max_depth": int(best_md),
            "n_estimators": int(best_ne),
            "late_val_logloss": float(best_late_ll),
        })
        print(f"Saved XGBoost hyperparams to: {META_FILE}")

    can_load_model = (
        USE_CACHED_ARTIFACTS
        and MODEL_FILE.exists()
        and not FORCE_RETUNE_META
        and not FORCE_REFIT_META_MODEL
    )

    if can_load_model:
        print(f"Loading trained XGBoost meta-model from: {MODEL_FILE}")
        meta_final = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        meta_final.load_model(str(MODEL_FILE))
    else:
        print("Fitting final XGBoost meta-model...")
        meta_final = XGBClassifier(
            n_estimators=best_ne,
            learning_rate=best_lr,
            max_depth=best_md,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        meta_final.fit(X_val_arr, y_val_arr)
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        meta_final.save_model(str(MODEL_FILE))
        print(f"Saved trained XGBoost meta-model to: {MODEL_FILE}")

    # --- Deep Learning Model (MLP) ---
    print("Training Deep Learning Baseline (MLP)...")
    # Τα Νευρωνικά Δίκτυα χρειάζονται scaled features (StandardScaler)
    mlp_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    )
    mlp_model.fit(X_val_arr, y_val_arr)
    
    print("\n--- Final Test Evaluation ---")
    t_probs_meta = meta_final.predict_proba(X_test_arr)
    t_probs_mlp = mlp_model.predict_proba(X_test_arr)

    def report(name, probs):
        print(f"\n{name}:")
        print("LogLoss:", round(log_loss(y_test_arr, probs), 4))
        print("Brier:", round(multiclass_brier(probs, y_test_arr), 4))
        print("ECE:", round(top_label_ece(probs, y_test_arr), 4))

    report("BASE (Model only, calibrated)", t_probs_model_arr)
    report("MARKET (odds implied)", t_mkt_fixed_arr)
    report("META (Market + Model)", t_probs_meta)
    report("DEEP LEARNING (MLP)", t_probs_mlp)

    # ── Per-league breakdown ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("=== PER-LEAGUE TEST METRICS ===")
    print("=" * 65)

    # Build per-league meta probs by re-running predict_proba on league slice
    # We reconstruct the league slices from per_league_test + X_test_arr ordering.
    # Since leagues were appended in LEAGUES order, we can slice X_test_arr.
    league_slice_start = 0
    col_w = 10

    header = (
        f"{'League':<10} | {'N':>4} | "
        f"{'Base LL':>{col_w}} | {'Base Brier':>{col_w}} | "
        f"{'Mkt LL':>{col_w}} | "
        f"{'Meta LL':>{col_w}} | {'Meta Brier':>{col_w}} | {'Meta ECE':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for league in LEAGUES:
        if league not in per_league_test:
            continue

        ld = per_league_test[league]
        y_l   = ld["y"]
        pm_l  = ld["p_model"]
        pmkt_l = ld["p_mkt"]
        n     = len(y_l)

        # Slice the matching rows from X_test_arr
        X_l = X_test_arr[league_slice_start: league_slice_start + n]
        league_slice_start += n

        meta_l = meta_final.predict_proba(X_l)

        base_ll    = round(log_loss(y_l, pm_l),                   4)
        base_brier = round(multiclass_brier(pm_l, y_l),           4)
        mkt_ll     = round(log_loss(y_l, pmkt_l),                 4)
        meta_ll    = round(log_loss(y_l, meta_l),                 4)
        meta_brier = round(multiclass_brier(meta_l, y_l),         4)
        meta_ece   = round(top_label_ece(meta_l, y_l),            4)

        print(
            f"{league.upper():<10} | {n:>4} | "
            f"{base_ll:>{col_w}} | {base_brier:>{col_w}} | "
            f"{mkt_ll:>{col_w}} | "
            f"{meta_ll:>{col_w}} | {meta_brier:>{col_w}} | {meta_ece:>{col_w}}"
        )

    print("\n--- Betting simulation - All Top 5 Leagues ---")
    threshold = 0.05

    bets, wins, profit, roi, avg_odds = simulate_value_betting(
        t_probs_meta, raw_odds_arr, y_test_arr, edge_threshold=threshold
    )

    print(f"Meta-Model Strategy (Edge > {threshold*100}%):")
    print(f"Total Bets Placed: {bets} (out of {len(X_test_arr)} matches)")
    if bets > 0:
        print(f"Won Bets: {wins} ({round(wins / bets * 100, 1)}% Hit Rate)")
        print(f"Average Odds Played: {round(avg_odds, 2)}")
    print(f"Net Profit: {round(profit, 2)} units")
    print(f"ROI: {round(roi, 2)}%")

    print("\n" + "=" * 70)
    print("=== CURRENT / NEXT MATCHDAY PICKS (ALL LEAGUES) ===")
    print("=" * 70)

    upcoming_rows = []

    for league in LEAGUES:
        params = league_best_params.get(league)
        if params is None:
            continue

        df_league_all = load_league_data(league).sort_values("date").reset_index(drop=True)

        fixtures, matchday_start = get_current_or_next_matchday_fixtures(
            df_league_all,
            max_window_days=4
        )

        if fixtures.empty or matchday_start is None:
            print(f"{league.upper()}: No current/upcoming matchday fixtures found.")
            continue

        print(f"\n{league.upper()} MATCHDAY starting: {matchday_start.date()}")

        played_df = df_league_all[
            (df_league_all["is_played"] == True) &
            (df_league_all["date"] < matchday_start)
        ].copy()

        if played_df.empty:
            print(f"{league.upper()}: Not enough played history before selected matchday.")
            continue

        from src.elo import expected_score, match_result, margin_multiplier

        ratings = {}

        def get_dynamic_init_local(current_ratings, default_init=1500.0):
            if len(current_ratings) >= 5:
                bottom_elos = sorted(current_ratings.values())[:3]
                return float(sum(bottom_elos) / len(bottom_elos))
            return float(default_init)

        def update_ratings_local(matches_batch, current_ratings):
            for _, m in matches_batch.iterrows():
                h, a = m["home_team"], m["away_team"]

                dyn_init = get_dynamic_init_local(current_ratings)
                r_h = current_ratings.get(h, dyn_init)
                r_a = current_ratings.get(a, dyn_init)

                if h not in current_ratings:
                    current_ratings[h] = r_h
                if a not in current_ratings:
                    current_ratings[a] = r_a

                exp_h = expected_score(r_h + params["ha"], r_a)
                s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
                mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

                current_ratings[h] = r_h + (params["K"] * mult) * (s_h - exp_h)
                current_ratings[a] = r_a + (params["K"] * mult) * (s_a - (1 - exp_h))
            return current_ratings

        ratings = update_ratings_local(played_df, ratings)

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            played_df, decay=params["decay"]
        )

        fixture_model_probs_raw = []
        fixture_market_probs = []
        fixture_aux = []
        fixture_lambdas = []

        for _, row in fixtures.iterrows():
            ht, at = row["home_team"], row["away_team"]

            dyn_init = get_dynamic_init_local(ratings)
            elo_h = ratings.get(ht, dyn_init)
            elo_a = ratings.get(at, dyn_init)

            lam_h, lam_a = predict_lambdas_home_away(
                ht, at,
                l_avg_h, l_avg_a,
                att_h, def_h,
                att_a, def_a,
            )

            lam_h, lam_a = apply_elo_to_lambdas(
                lam_h, lam_a, elo_h, elo_a, beta=params["beta"]
            )

            pH, pD, pA = match_outcome_probs_dc(
                lam_h, lam_a, rho=params["rho"], max_goals=10
            )

            fixture_model_probs_raw.append([pH, pD, pA])
            fixture_market_probs.append(
                market_probs_from_odds_row(
                    row["odds_home"], row["odds_draw"], row["odds_away"]
                ).tolist()
            )
            fixture_aux.append([
                (elo_h - elo_a) / 400.0,
                lam_h + lam_a,
                lam_h - lam_a,
            ])
            fixture_lambdas.append([lam_h, lam_a])

        fixture_model_probs_raw = np.array(fixture_model_probs_raw, dtype=float)
        fixture_market_probs = np.array(fixture_market_probs, dtype=float)
        fixture_aux = np.array(fixture_aux, dtype=float)
        fixture_lambdas = np.array(fixture_lambdas, dtype=float)

        fixture_model_probs = temperature_scale_probs(fixture_model_probs_raw, params["T"])

        fixture_market_fixed = fixture_market_probs.copy()
        for i in range(len(fixture_market_fixed)):
            if not np.isfinite(fixture_market_fixed[i]).all():
                fixture_market_fixed[i] = fixture_model_probs[i]

        X_future = build_meta_features(fixture_model_probs, fixture_market_fixed, fixture_aux)
        future_meta_probs = meta_final.predict_proba(X_future)

        for i, (_, row) in enumerate(fixtures.iterrows()):
            pH, pD, pA = future_meta_probs[i]
            lam_h, lam_a = fixture_lambdas[i]

            result_pick = ["H", "D", "A"][np.argmax([pH, pD, pA])]

            top_score = top_k_scorelines_dc(
                lam_h,
                lam_a,
                rho=params["rho"],
                k=1,
                max_goals=6,
            )
            (hg, ag), score_prob = top_score[0]

            upcoming_rows.append({
                "League": league.upper(),
                "Date": row["date"].strftime("%Y-%m-%d"),
                "Home": row["home_team"],
                "Away": row["away_team"],
                "P(H)": round(float(pH), 3),
                "P(D)": round(float(pD), 3),
                "P(A)": round(float(pA), 3),
                "Pick": result_pick,
                "Score Pick": f"{hg}-{ag}",
                "Score Prob": round(float(score_prob), 3),
            })

    if len(upcoming_rows) == 0:
        print("No current/upcoming matchday fixtures available.")
    else:
        picks_df = pd.DataFrame(upcoming_rows)
        picks_df = picks_df.sort_values(["Date", "League", "Home"]).reset_index(drop=True)
        print("\n")
        print(picks_df.to_string(index=False))


if __name__ == "__main__":
    main()