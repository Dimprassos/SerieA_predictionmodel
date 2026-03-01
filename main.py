import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

from src.data_processing import load_and_merge_data
from src.poisson_model import (
    fit_team_strengths,
    fit_team_strengths_weighted,
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs,
    match_outcome_probs_dc,
)
from src.elo import compute_elo_ratings
from src.calibration import fit_temperature, temperature_scale_probs
from src.metrics import multiclass_brier, top_label_ece


# ---------------------------
# Market implied probabilities
# ---------------------------
def market_probs_from_odds_row(odds_h, odds_d, odds_a):
    if not (np.isfinite(odds_h) and np.isfinite(odds_d) and np.isfinite(odds_a)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if odds_h <= 1.0001 or odds_d <= 1.0001 or odds_a <= 1.0001:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    inv = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    s = inv.sum()
    if s <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    # normalize (remove overround)
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


# ------------------------------------------------------------
# Streaming block-walk-forward (per date) with home/away strengths
# Returns: model_probs_raw, y_true, market_probs, aux_features
# ------------------------------------------------------------
def streaming_block_probs_home_away(
    predict_df,
    full_df,
    beta,
    rho,
    decay,
    K,
    home_adv,
    init_rating=1500.0,
    max_goals=10
):
    from src.elo import expected_score, match_result, margin_multiplier

    probs_model = []
    probs_mkt = []
    y_true = []
    aux = []  # [elo_diff_scaled, total_xg, xg_diff]

    predict_df = predict_df.sort_values("date")
    full_df = full_df.sort_values("date")

    predict_dates = sorted(predict_df["date"].unique())
    if len(predict_dates) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0, 3)), np.zeros((0, 3))

    first_predict_date = predict_dates[0]

    # ---- init ratings from history only
    ratings = {}
    history_matches = full_df[full_df["date"] < first_predict_date]

    def update_ratings(matches_batch, current_ratings):
        for _, m in matches_batch.iterrows():
            h, a = m["home_team"], m["away_team"]
            r_h = current_ratings.get(h, init_rating)
            r_a = current_ratings.get(a, init_rating)

            exp_h = expected_score(r_h + home_adv, r_a)
            s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
            mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

            current_ratings[h] = r_h + (K * mult) * (s_h - exp_h)
            current_ratings[a] = r_a + (K * mult) * (s_a - (1 - exp_h))
        return current_ratings

    ratings = update_ratings(history_matches, ratings)

    # ---- per date
    for d in predict_dates:
        day_matches = predict_df[predict_df["date"] == d]
        past_matches = full_df[full_df["date"] < d]

        # strengths once per day (home/away, weighted)
        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            past_matches, decay=decay
        )

        for _, row in day_matches.iterrows():
            ht, at = row["home_team"], row["away_team"]

            elo_h = ratings.get(ht, init_rating)
            elo_a = ratings.get(at, init_rating)

            lam_h, lam_a = predict_lambdas_home_away(
                ht, at,
                l_avg_h, l_avg_a,
                att_h, def_h,
                att_a, def_a
            )

            lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=beta)

            # Dixon–Coles 1X2 probs
            pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=max_goals)
            probs_model.append([pH, pD, pA])

            # market probs
            pm = market_probs_from_odds_row(row["odds_home"], row["odds_draw"], row["odds_away"])
            probs_mkt.append(pm.tolist())

            # label
            if row["home_goals"] > row["away_goals"]:
                y_true.append(0)
            elif row["home_goals"] == row["away_goals"]:
                y_true.append(1)
            else:
                y_true.append(2)

            # aux
            elo_diff = (elo_h - elo_a) / 400.0
            total_xg = lam_h + lam_a
            xg_diff = lam_h - lam_a
            aux.append([elo_diff, total_xg, xg_diff])

        # update Elo after day end
        ratings = update_ratings(day_matches, ratings)

    return (
        np.array(probs_model, dtype=float),
        np.array(y_true, dtype=int),
        np.array(probs_mkt, dtype=float),
        np.array(aux, dtype=float),
    )


# ---------------------------
# Build meta features
# ---------------------------
def build_meta_features(p_model, p_mkt, aux):
    """
    Features:
    - logits of model probs (3)
    - logits of market probs (3)
    - aux (elo_diff, total_xg, xg_diff) (3)
    Total: 9 features
    """
    # handle missing market by fallback to model
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
    """
    Split validation into early/late by date (time-safe)
    """
    val_sorted = val_df.sort_values("date").reset_index(drop=True)
    mid = len(val_sorted) // 2
    return val_sorted.iloc[:mid].copy(), val_sorted.iloc[mid:].copy()

def simulate_value_betting(probs, raw_odds, y_true, edge_threshold=0.05):
    """
    Προσομοίωση Flat Betting (1 unit ανά στοίχημα)
    edge_threshold: το ελάχιστο πλεονέκτημα (EV) που ζητάμε (π.χ. 0.05 = 5% Value)
    """
    bets_placed = 0
    won_bets = 0
    total_invested = 0.0
    total_return = 0.0
    total_odds_taken = 0.0  # ΝΕΟ: Για να βρούμε τη μέση απόδοση

    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        # Αν λείπουν αποδόσεις, το προσπερνάμε
        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        # Υπολογισμός Expected Value (EV) = (Πιθανότητα * Απόδοση) - 1
        ev_h = (p_h * o_h) - 1.0
        ev_d = (p_d * o_d) - 1.0
        ev_a = (p_a * o_a) - 1.0

        # Βρίσκουμε το μεγαλύτερο EV στο ματς
        best_ev = max(ev_h, ev_d, ev_a)

        # Αν βρήκαμε πλεονέκτημα μεγαλύτερο από το κατώφλι που θέσαμε
        if best_ev > edge_threshold:
            bets_placed += 1
            total_invested += 1.0  # Ποντάρουμε 1 μονάδα

            # Βρίσκουμε ποιο σημείο επιλέξαμε
            if best_ev == ev_h:
                choice, odds_taken = 0, o_h
            elif best_ev == ev_d:
                choice, odds_taken = 1, o_d
            else:
                choice, odds_taken = 2, o_a
                
            # Προσθέτουμε την απόδοση στο σύνολο
            total_odds_taken += odds_taken

            # Ελέγχουμε αν το στοίχημα κέρδισε
            if choice == y_true[i]:
                won_bets += 1
                total_return += odds_taken

    profit = total_return - total_invested
    roi = (profit / total_invested * 100) if total_invested > 0 else 0.0
    
    # ΝΕΟ: Υπολογισμός μέσης απόδοσης
    avg_odds = (total_odds_taken / bets_placed) if bets_placed > 0 else 0.0
    
    return bets_placed, won_bets, profit, roi, avg_odds


# ---------------------------
# MAIN
# ---------------------------
def main():
    df = load_and_merge_data()
    df = df.sort_values("date").reset_index(drop=True)

    # SPLITS
    train_fit = df[df["date"] < "2022-07-01"].copy()
    val = df[(df["date"] >= "2022-07-01") & (df["date"] < "2023-07-01")].copy()
    test = df[df["date"] >= "2023-07-01"].copy()

    print(f"Train_fit: {len(train_fit)}, Validation: {len(val)}, Test: {len(test)}")

    # ------------------------------------------------------------
    # 1) Tune Elo & beta on VAL (fast stage)
    # ------------------------------------------------------------
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.0015, 0.002, 0.003]

    best = None
    print("\n--- Elo & Beta Tuning (Validation Set) ---")

    for K in Ks:
        for ha in home_advs:
            for b in betas:
                elo_pairs = compute_elo_ratings(train_fit, K=K, home_adv=ha, use_margin=True)
                tmp = train_fit.copy()
                tmp["elo_home"], tmp["elo_away"] = zip(*elo_pairs)

                l_avg_h, l_avg_a, att, dfn = fit_team_strengths(tmp)

                full_tmp = pd.concat([tmp, val], ignore_index=True).sort_values("date")
                elo_full = compute_elo_ratings(full_tmp, K=K, home_adv=ha, use_margin=True)
                full_tmp["elo_home"], full_tmp["elo_away"] = zip(*elo_full)
                val_part = full_tmp.iloc[len(tmp):]

                probs = []
                y_v = []
                for _, row in val_part.iterrows():
                    # simple poisson probs for fast tuning
                    # (we keep your previous approach here)
                    from src.poisson_model import predict_lambdas
                    lh, la = predict_lambdas(
                        row["home_team"], row["away_team"],
                        l_avg_h, l_avg_a, att, dfn
                    )
                    lh, la = apply_elo_to_lambdas(lh, la, row["elo_home"], row["elo_away"], beta=b)
                    probs.append(match_outcome_probs(lh, la))

                    if row["home_goals"] > row["away_goals"]:
                        y_v.append(0)
                    elif row["home_goals"] == row["away_goals"]:
                        y_v.append(1)
                    else:
                        y_v.append(2)

                ll = log_loss(np.array(y_v), np.array(probs))
                if best is None or ll < best[0]:
                    best = (ll, K, ha, b)

    best_ll, best_K, best_ha, best_beta = best
    print(f"Best Config: K={best_K}, ha={best_ha}, beta={best_beta}, Val LogLoss={round(best_ll, 4)}")

    # ------------------------------------------------------------
    # 2) Build full Elo on all df
    # ------------------------------------------------------------
    elo_all = compute_elo_ratings(df, K=best_K, home_adv=best_ha, use_margin=True)
    df["elo_home"], df["elo_away"] = zip(*elo_all)

    train_fit = df[df["date"] < "2022-07-01"].copy()
    val = df[(df["date"] >= "2022-07-01") & (df["date"] < "2023-07-01")].copy()
    test = df[df["date"] >= "2023-07-01"].copy()

    # ------------------------------------------------------------
    # 3) Tune decay on VAL (fast stage)
    # ------------------------------------------------------------
    print("\n--- Tuning Time Decay ---")
    y_val_static = labels_from_df(val)

    best_decay, best_decay_ll = None, float("inf")
    for d in decays:
        l_avg_h, l_avg_a, att, dfn = fit_team_strengths_weighted(train_fit, decay=d)

        probs_d = []
        from src.poisson_model import predict_lambdas
        for _, row in val.iterrows():
            lh, la = predict_lambdas(row["home_team"], row["away_team"], l_avg_h, l_avg_a, att, dfn)
            lh, la = apply_elo_to_lambdas(lh, la, row["elo_home"], row["elo_away"], beta=best_beta)
            probs_d.append(match_outcome_probs(lh, la))

        ll_d = log_loss(y_val_static, np.array(probs_d))
        if ll_d < best_decay_ll:
            best_decay_ll, best_decay = ll_d, d

    print(f"Best Decay: {best_decay}, Val LogLoss: {round(best_decay_ll, 4)}")

    # ------------------------------------------------------------
    # 4) Joint tune rho + temperature on VAL (streaming, home/away)
    # ------------------------------------------------------------
    print("\n--- Joint Tuning rho + Temperature (VAL, streaming, home/away) ---")
    rho_grid = np.round(np.arange(-0.20, 0.201, 0.01), 2)

    best_joint = None  # (val_cal_ll, rho, T, val_raw_ll)

    full_df_for_stream = df.copy()

    for rho in rho_grid:
        val_probs_raw, y_val, val_mkt, val_aux = streaming_block_probs_home_away(
            val, full_df_for_stream,
            beta=best_beta, rho=float(rho), decay=best_decay,
            K=best_K, home_adv=best_ha
        )
        T = fit_temperature(val_probs_raw, y_val)
        val_probs_cal = temperature_scale_probs(val_probs_raw, T)

        val_raw_ll = log_loss(y_val, val_probs_raw)
        val_cal_ll = log_loss(y_val, val_probs_cal)

        if (best_joint is None) or (val_cal_ll < best_joint[0]):
            best_joint = (val_cal_ll, float(rho), float(T), float(val_raw_ll))

    best_val_cal_ll, best_rho, best_T, best_val_raw_ll = best_joint
    print("\nBest (VAL calibrated LogLoss):")
    print("  rho:", best_rho, "T:", round(best_T, 3))
    print("  VAL raw LogLoss:", round(best_val_raw_ll, 4))
    print("  VAL calibrated LogLoss:", round(best_val_cal_ll, 4))

    # ------------------------------------------------------------
    # 5) META-MODEL: train on VAL (time-split inside VAL for C tuning)
    # ------------------------------------------------------------
    print("\n--- META-MODEL (Market + Model) ---")

    val_early, val_late = time_split_val(val)

    # build streaming preds for early/late with SAME hyperparams
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

    # use calibrated model probs as input to meta (better behaved)
    ve_probs_model = temperature_scale_probs(ve_probs_raw, best_T)
    vl_probs_model = temperature_scale_probs(vl_probs_raw, best_T)

    X_early = build_meta_features(ve_probs_model, ve_mkt, ve_aux)
    X_late = build_meta_features(vl_probs_model, vl_mkt, vl_aux)

    # Tune C time-safe: train on early, pick C by late logloss
    Cs = [0.1, 0.3, 1.0, 3.0, 10.0]
    best_meta = None  # (late_ll, C)

    for C in Cs:
        meta = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        C=C
    )
        meta.fit(X_early, ve_y)
        late_probs = meta.predict_proba(X_late)
        late_ll = log_loss(vl_y, late_probs)

        if best_meta is None or late_ll < best_meta[0]:
            best_meta = (late_ll, C)

    best_late_ll, best_C = best_meta
    print("Best meta C (by late-VAL LogLoss):", best_C, "Late VAL LogLoss:", round(best_late_ll, 4))

    # Refit meta on FULL VAL
    v_probs_raw, v_y, v_mkt, v_aux = streaming_block_probs_home_away(
        val, full_df_for_stream,
        beta=best_beta, rho=best_rho, decay=best_decay,
        K=best_K, home_adv=best_ha
    )
    v_probs_model = temperature_scale_probs(v_probs_raw, best_T)
    X_val = build_meta_features(v_probs_model, v_mkt, v_aux)

    meta_final = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=3000,
        C=best_C
    )
    meta_final.fit(X_val, v_y)

    # ------------------------------------------------------------
    # 6) FINAL TEST: evaluate base vs market vs meta
    # ------------------------------------------------------------
    print("\n--- FINAL TEST EVALUATION ---")

    t_probs_raw, t_y, t_mkt, t_aux = streaming_block_probs_home_away(
        test, full_df_for_stream,
        beta=best_beta, rho=best_rho, decay=best_decay,
        K=best_K, home_adv=best_ha
    )

    # Base calibrated
    t_probs_model = temperature_scale_probs(t_probs_raw, best_T)

    # Market (fallback to model if missing)
    t_mkt_fixed = t_mkt.copy()
    for i in range(len(t_mkt_fixed)):
        if not np.isfinite(t_mkt_fixed[i]).all():
            t_mkt_fixed[i] = t_probs_model[i]

    # Meta
    X_test = build_meta_features(t_probs_model, t_mkt_fixed, t_aux)
    t_probs_meta = meta_final.predict_proba(X_test)

    # Metrics helper
    def report(name, probs):
        print(f"\n{name}:")
        print("LogLoss:", round(log_loss(t_y, probs), 4))
        print("Brier:", round(multiclass_brier(probs, t_y), 4))
        print("ECE:", round(top_label_ece(probs, t_y), 4))

    report("BASE (Model only, calibrated)", t_probs_model)
    report("MARKET (odds implied)", t_mkt_fixed)
    report("META (Market + Model)", t_probs_meta)

    # --- BETTING SIMULATION ---
    print("\n--- BETTING SIMULATION (TEST SET) ---")
    
    # Παίρνουμε τις πραγματικές αποδόσεις από το Test dataframe
    raw_odds = test[["odds_home", "odds_draw", "odds_away"]].values
    
    # Ορίζουμε ότι θέλουμε τουλάχιστον 5% Value για να ποντάρουμε
    threshold = 0.05 
    
    bets, wins, profit, roi, avg_odds = simulate_value_betting(t_probs_meta, raw_odds, t_y, edge_threshold=threshold)
    
    print(f"Meta-Model Strategy (Edge > {threshold*100}%):")
    print(f"Total Bets Placed: {bets} (out of {len(test)} matches)")
    if bets > 0:
        print(f"Won Bets: {wins} ({round(wins/bets*100, 1)}% Hit Rate)")
        print(f"Average Odds Played: {round(avg_odds, 2)}")
    print(f"Net Profit: {round(profit, 2)} units")
    print(f"ROI: {round(roi, 2)}%")

    demo = test.iloc[0]
    #print(f"\nExample match: {demo['home_team']} vs {demo['away_team']}")
    #print(f"Actual Score: {int(demo['home_goals'])}-{int(demo['away_goals'])}")


if __name__ == "__main__":
    main()