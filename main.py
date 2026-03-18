from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.artifacts import load_json_if_exists, save_json, save_pickle, load_pickle_if_exists
from src.evaluation import labels_from_df, simulate_value_betting
from src.fixtures import get_current_or_next_matchday_fixtures
from src.data_processing import load_league_data
from src.poisson_model import (
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs_dc,
    top_k_scorelines_dc,
)
from src.calibration import temperature_scale_probs
from src.meta_features import build_meta_features, time_split_val
from src.streaming import streaming_block_probs_home_away
from src.reporting import print_prob_report, print_per_league_test_metrics
from src.prediction_service import generate_upcoming_matchday_picks
from src.tuning import (
    tune_league_params,
    tune_xgb_hyperparams,
    tune_mlp_hyperparams,
    tune_blend_weights,
    make_mlp_pipeline,
    probs_from_meta_features,
    blend_probabilities,
)

# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_NAME = "baseline_xgboost_v1"

ARTIFACTS_DIR = Path("artifacts")
PARAMS_FILE = ARTIFACTS_DIR / f"best_params_{EXPERIMENT_NAME}.json"
META_FILE = ARTIFACTS_DIR / f"best_meta_{EXPERIMENT_NAME}.json"
MODEL_FILE = ARTIFACTS_DIR / f"meta_model_{EXPERIMENT_NAME}.json"
MLP_META_FILE = ARTIFACTS_DIR / f"best_mlp_{EXPERIMENT_NAME}.json"
MLP_MODEL_FILE = ARTIFACTS_DIR / f"mlp_model_{EXPERIMENT_NAME}.pkl"
BLEND_FILE = ARTIFACTS_DIR / f"best_blend_{EXPERIMENT_NAME}.json"

USE_CACHED_ARTIFACTS = False 
FORCE_RETUNE_LEAGUES = True    # Το αφήνεις False (έχουμε ήδη τέλεια Poisson/Elo params)
FORCE_RETUNE_META = True        # Ενεργοποιεί το Optuna για το XGBoost
FORCE_REFIT_META_MODEL = True   # Αναγκάζει το XGBoost να ξαναγίνει fit με τα νέα params
FORCE_RETUNE_MLP = True         # Ενεργοποιεί το Optuna για το MLP
FORCE_REFIT_MLP_MODEL = True    # Αναγκάζει το MLP να ξαναγίνει fit με τα νέα params
FORCE_RETUNE_BLEND = True       # Απαραίτητο True, γιατί αφού άλλαξαν τα μοντέλα, αλλάζουν και τα ιδανικά βάρη

TRAIN_CUT = "2024-07-01"
TEST_CUT = "2025-07-01"

LEAGUES = ["england", "spain", "italy", "germany", "france"]
# ============================================================
# Main
# ============================================================
def main():
    cached_params = load_json_if_exists(PARAMS_FILE) if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_LEAGUES else None
    cached_meta = load_json_if_exists(META_FILE) if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_META else None
    cached_mlp = load_json_if_exists(MLP_META_FILE) if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_MLP else None
    cached_blend = load_json_if_exists(BLEND_FILE) if USE_CACHED_ARTIFACTS and not FORCE_RETUNE_BLEND else None

    league_best_params = {} if cached_params is None else cached_params

    all_X_early, all_y_early = [], []
    all_X_late, all_y_late = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    all_t_probs_model = []
    all_t_mkt_fixed = []
    all_t_raw_odds = []
    per_league_test = {}

    for league in LEAGUES:
        print("\n" + "=" * 50)
        print(f"=== Processing Data: {league.upper()} ===")
        print("=" * 50)

        df_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        df = df_all[df_all["is_played"] == True].copy().reset_index(drop=True)
        if df.empty:
            print(f"No played matches for {league}.")
            continue

        train_fit = df[df["date"] < TRAIN_CUT].copy()
        val = df[(df["date"] >= TRAIN_CUT) & (df["date"] < TEST_CUT)].copy()
        test = df[df["date"] >= TEST_CUT].copy()
        print(f"Train_fit: {len(train_fit)}, Validation: {len(val)}, Test: {len(test)}")
        if len(train_fit) == 0 or len(val) == 0 or len(test) == 0:
            print(f"Not enough splits for {league}.")
            continue

        if league in league_best_params and not FORCE_RETUNE_LEAGUES:
            params = league_best_params[league]
            print("\n--- Using cached params ---")
            print(f"K={params['K']}, ha={params['ha']}, beta={params['beta']}, decay={params['decay']}, rho={params['rho']}, T={params['T']}")
        else:
            params = tune_league_params(train_fit, val, df, streaming_block_probs_home_away, apply_elo_to_lambdas)
            league_best_params[league] = params

        best_K, best_ha = params["K"], params["ha"]
        best_beta, best_decay = params["beta"], params["decay"]
        best_rho, best_T = params["rho"], params["T"]

        full_df_for_stream = df.copy()
        val_early, val_late = time_split_val(val)

        ve_probs_raw, ve_y, ve_mkt, ve_aux = streaming_block_probs_home_away(val_early, full_df_for_stream, best_beta, best_rho, best_decay, best_K, best_ha)
        vl_probs_raw, vl_y, vl_mkt, vl_aux = streaming_block_probs_home_away(val_late, full_df_for_stream, best_beta, best_rho, best_decay, best_K, best_ha)
        ve_probs_model = temperature_scale_probs(ve_probs_raw, best_T)
        vl_probs_model = temperature_scale_probs(vl_probs_raw, best_T)

        all_X_early.extend(build_meta_features(ve_probs_model, ve_mkt, ve_aux))
        all_y_early.extend(ve_y)
        all_X_late.extend(build_meta_features(vl_probs_model, vl_mkt, vl_aux))
        all_y_late.extend(vl_y)

        v_probs_raw, v_y, v_mkt, v_aux = streaming_block_probs_home_away(val, full_df_for_stream, best_beta, best_rho, best_decay, best_K, best_ha)
        v_probs_model = temperature_scale_probs(v_probs_raw, best_T)
        all_X_val.extend(build_meta_features(v_probs_model, v_mkt, v_aux))
        all_y_val.extend(v_y)

        t_probs_raw, t_y, t_mkt, t_aux = streaming_block_probs_home_away(test, full_df_for_stream, best_beta, best_rho, best_decay, best_K, best_ha)
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
        per_league_test[league] = {"y": np.array(t_y, dtype=int), "p_model": np.array(t_probs_model, dtype=float), "p_mkt": np.array(t_mkt_fixed, dtype=float)}

    if cached_params is None or FORCE_RETUNE_LEAGUES:
        save_json(PARAMS_FILE, league_best_params)
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

    # XGBoost hyperparams
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
        best_meta_cfg = tune_xgb_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        save_json(META_FILE, best_meta_cfg)
        print(f"Saved XGBoost hyperparams to: {META_FILE}")
        best_lr = best_meta_cfg["learning_rate"]
        best_md = best_meta_cfg["max_depth"]
        best_ne = best_meta_cfg["n_estimators"]
        best_late_ll = best_meta_cfg["late_val_logloss"]
        print(f"Best XGBoost Config -> LR: {best_lr}, Depth: {best_md}, Trees: {best_ne}")
        print(f"Late VAL LogLoss: {round(best_late_ll, 4)}")

    # final XGB model
    can_load_meta_model = USE_CACHED_ARTIFACTS and MODEL_FILE.exists() and not FORCE_RETUNE_META and not FORCE_REFIT_META_MODEL
    if can_load_meta_model:
        print(f"Loading trained XGBoost meta-model from: {MODEL_FILE}")
        meta_final = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1)
        meta_final.load_model(str(MODEL_FILE))
    else:
        print("Fitting final XGBoost meta-model...")
        meta_final = XGBClassifier(
            n_estimators=best_ne, learning_rate=best_lr, max_depth=best_md,
            objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        meta_final.fit(X_val_arr, y_val_arr)
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        meta_final.save_model(str(MODEL_FILE))
        print(f"Saved trained XGBoost meta-model to: {MODEL_FILE}")

    # MLP
    if cached_mlp is not None and not FORCE_RETUNE_MLP:
        best_mlp_cfg = cached_mlp
        print("Using cached MLP hyperparameters...")
        print(f"Best MLP -> layers={tuple(best_mlp_cfg['hidden_layer_sizes'])}, alpha={best_mlp_cfg['alpha']}, lr={best_mlp_cfg['learning_rate_init']}, T={best_mlp_cfg['temperature']}")
    else:
        best_mlp_cfg = tune_mlp_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        save_json(MLP_META_FILE, best_mlp_cfg)
        print(f"Saved MLP hyperparams to: {MLP_META_FILE}")

    can_load_mlp_model = USE_CACHED_ARTIFACTS and MLP_MODEL_FILE.exists() and not FORCE_RETUNE_MLP and not FORCE_REFIT_MLP_MODEL
    if can_load_mlp_model:
        print(f"Loading trained MLP model from: {MLP_MODEL_FILE}")
        mlp_model = load_pickle_if_exists(MLP_MODEL_FILE)
    else:
        print("Fitting final MLP model...")
        mlp_model = make_mlp_pipeline(best_mlp_cfg)
        mlp_model.fit(X_val_arr, y_val_arr)
        save_pickle(MLP_MODEL_FILE, mlp_model)
        print(f"Saved trained MLP model to: {MLP_MODEL_FILE}")

    # Blend tuning on late validation
    xgb_late_model = XGBClassifier(
        n_estimators=best_ne, learning_rate=best_lr, max_depth=best_md,
        objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    xgb_late_model.fit(X_early_arr, y_early_arr)
    late_probs_xgb = xgb_late_model.predict_proba(X_late_arr)

    mlp_late_model = make_mlp_pipeline(best_mlp_cfg)
    mlp_late_model.fit(X_early_arr, y_early_arr)
    late_probs_mlp_raw = mlp_late_model.predict_proba(X_late_arr)
    late_probs_mlp = temperature_scale_probs(late_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    late_probs_base = probs_from_meta_features(X_late_arr, 0)
    late_probs_mkt = probs_from_meta_features(X_late_arr, 3)

    if cached_blend is not None and not FORCE_RETUNE_BLEND:
        blend_cfg = cached_blend
        print("Using cached blend weights...")
        print(f"Blend weights: {blend_cfg['weights']}")
        print(f"Cached Late VAL LogLoss: {round(blend_cfg['late_val_logloss'], 4)}")
    else:
        blend_cfg = tune_blend_weights(
            y_late_arr,
            probs_base=late_probs_base,
            probs_market=late_probs_mkt,
            probs_xgb=late_probs_xgb,
            probs_mlp=late_probs_mlp,
            step=0.05,
        )
        save_json(BLEND_FILE, blend_cfg)
        print(f"Saved blend weights to: {BLEND_FILE}")
        print(f"Blend weights: {blend_cfg['weights']}")
        print(f"Late VAL LogLoss: {round(blend_cfg['late_val_logloss'], 4)}")

    # final predictions
    t_probs_meta = meta_final.predict_proba(X_test_arr)
    t_probs_mlp_raw = mlp_model.predict_proba(X_test_arr)
    t_probs_mlp = temperature_scale_probs(t_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    t_probs_blend = blend_probabilities(
        blend_cfg["weights"],
        {
            "base": t_probs_model_arr,
            "market": t_mkt_fixed_arr,
            "xgb": t_probs_meta,
            "mlp": t_probs_mlp,
        },
    )

    print_prob_report("BASE (Model only, calibrated)", t_probs_model_arr, y_test_arr)
    print_prob_report("MARKET (odds implied)", t_mkt_fixed_arr, y_test_arr)
    print_prob_report("META (Market + Model)", t_probs_meta, y_test_arr)
    print_prob_report("DEEP LEARNING (MLP, calibrated)", t_probs_mlp, y_test_arr)
    print_prob_report("ENSEMBLE (XGB + MLP + Market + Base)", t_probs_blend, y_test_arr)

    print_per_league_test_metrics(
        LEAGUES,
        per_league_test,
        t_probs_meta,
        t_probs_mlp,
        t_probs_blend,
    )

    print("\n--- Betting simulation - All Top 5 Leagues ---")
    threshold = 0.05
    bets, wins, profit, roi, avg_odds = simulate_value_betting(t_probs_blend, raw_odds_arr, y_test_arr, edge_threshold=threshold)
    print(f"Ensemble Strategy (Edge > {threshold*100}%):")
    print(f"Total Bets Placed: {bets} (out of {len(X_test_arr)} matches)")
    if bets > 0:
        print(f"Won Bets: {wins} ({round(wins / bets * 100, 1)}% Hit Rate)")
        print(f"Average Odds Played: {round(avg_odds, 2)}")
    print(f"Net Profit: {round(profit, 2)} units")
    print(f"ROI: {round(roi, 2)}%")

    # attach blend config for prediction service
    for league in LEAGUES:
        if league in league_best_params:
            league_best_params[league]["_blend_cfg"] = blend_cfg

    generate_upcoming_matchday_picks(
        LEAGUES,
        league_best_params,
        meta_final,
        mlp_model,
        best_mlp_cfg,
        max_window_days=4,
    )


if __name__ == "__main__":
    main()
