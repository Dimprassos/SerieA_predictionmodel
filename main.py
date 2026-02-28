import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.data_processing import load_and_merge_data
from src.elo import compute_elo_ratings
from src.poisson_model import (
    fit_team_strengths,
    fit_team_strengths_weighted,
    predict_lambdas,
    apply_elo_to_lambdas,
    match_outcome_probs,
    match_outcome_probs_dc,
    scoreline_probs_dc,
)
from src.calibration import fit_temperature, temperature_scale_probs
from src.metrics import multiclass_brier, top_label_ece


# ------------------------------------------------------------
# Utility: get predictions (raw DC probabilities)
# ------------------------------------------------------------
def get_probs(df, league_avg_home, league_avg_away, attack, defense, beta, rho):
    y_true = []
    probs = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        lam_h, lam_a = predict_lambdas(
            ht, at,
            league_avg_home, league_avg_away,
            attack, defense
        )

        lam_h, lam_a = apply_elo_to_lambdas(
            lam_h, lam_a,
            row["elo_home"], row["elo_away"],
            beta=beta
        )

        pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=10)
        probs.append([pH, pD, pA])

        if row["home_goals"] > row["away_goals"]:
            y_true.append(0)
        elif row["home_goals"] == row["away_goals"]:
            y_true.append(1)
        else:
            y_true.append(2)

    return np.array(probs), np.array(y_true)


# ------------------------------------------------------------
# Fit rho on train_fit
# ------------------------------------------------------------
def fit_rho(train_df, league_avg_home, league_avg_away, attack, defense,
            K, home_adv, use_margin, beta):

    # ensure Elo exists for the chosen config inside rho-fit
    elo_pairs = compute_elo_ratings(train_df, K=K, home_adv=home_adv, use_margin=use_margin)
    tmp = train_df.copy()
    tmp["elo_home"] = [x[0] for x in elo_pairs]
    tmp["elo_away"] = [x[1] for x in elo_pairs]

    rhos = np.arange(-0.30, 0.301, 0.01)
    best_rho = 0.0
    best_nll = float("inf")

    for rho in rhos:
        nll = 0.0
        ok = True

        for _, row in tmp.iterrows():
            ht, at = row["home_team"], row["away_team"]

            lam_h, lam_a = predict_lambdas(
                ht, at, league_avg_home, league_avg_away, attack, defense
            )

            lam_h, lam_a = apply_elo_to_lambdas(
                lam_h, lam_a,
                row["elo_home"], row["elo_away"],
                beta=beta
            )

            P = scoreline_probs_dc(lam_h, lam_a, rho, max_goals=10)
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])

            p = P[hg][ag] if (hg <= 10 and ag <= 10) else 1e-12
            if p <= 0:
                ok = False
                break

            nll -= np.log(p)

        if ok and nll < best_nll:
            best_nll = nll
            best_rho = float(rho)

    return best_rho


def main():
    df = load_and_merge_data()
    df = df.sort_values("date").reset_index(drop=True)

    # SPLITS (initial, without Elo)
    train_fit = df[df["date"] < "2022-07-01"]
    val = df[(df["date"] >= "2022-07-01") & (df["date"] < "2023-07-01")]
    test = df[df["date"] >= "2023-07-01"]

    print("Train_fit:", len(train_fit))
    print("Validation:", len(val))
    print("Test:", len(test))

    # -------------------------
    # Grid search Elo on validation (Poisson outcome probs)
    # -------------------------
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.0015, 0.002, 0.003]

    best = None  # (logloss, K, ha, beta)

    for K in Ks:
        for ha in home_advs:
            for beta in betas:
                # Elo on train_fit
                elo_pairs = compute_elo_ratings(train_fit, K=K, home_adv=ha, use_margin=True)
                tmp = train_fit.copy()
                tmp["elo_home"] = [x[0] for x in elo_pairs]
                tmp["elo_away"] = [x[1] for x in elo_pairs]

                # strengths on train_fit (unweighted for Elo tuning stage)
                league_avg_home, league_avg_away, attack, defense = fit_team_strengths(tmp)

                # Elo for val chronologically (train_fit + val)
                full_tmp = pd.concat([tmp, val], ignore_index=True).sort_values("date")
                elo_pairs_full = compute_elo_ratings(full_tmp, K=K, home_adv=ha, use_margin=True)
                full_tmp["elo_home"] = [x[0] for x in elo_pairs_full]
                full_tmp["elo_away"] = [x[1] for x in elo_pairs_full]

                # Extract val part safely (last len(val) rows after concat, because val appended)
                val_part = full_tmp.iloc[len(tmp):]

                probs = []
                y_val = []

                for _, row in val_part.iterrows():
                    lam_h, lam_a = predict_lambdas(
                        row["home_team"], row["away_team"],
                        league_avg_home, league_avg_away,
                        attack, defense
                    )

                    lam_h, lam_a = apply_elo_to_lambdas(
                        lam_h, lam_a,
                        row["elo_home"], row["elo_away"],
                        beta=beta
                    )

                    pH, pD, pA = match_outcome_probs(lam_h, lam_a)
                    probs.append([pH, pD, pA])

                    if row["home_goals"] > row["away_goals"]:
                        y_val.append(0)
                    elif row["home_goals"] == row["away_goals"]:
                        y_val.append(1)
                    else:
                        y_val.append(2)

                probs = np.array(probs)
                y_val = np.array(y_val)

                ll = log_loss(y_val, probs)

                if best is None or ll < best[0]:
                    best = (ll, K, ha, beta)

    best_ll, best_K, best_ha, best_beta = best

    print("\nBest config (VAL LogLoss):")
    print("LogLoss:", round(best_ll, 4))
    print("K:", best_K, "home_adv:", best_ha, "beta:", best_beta)

    # -------------------------
    # Build full Elo on ALL df with chosen Elo config
    # -------------------------
    elo_pairs_all = compute_elo_ratings(df, K=best_K, home_adv=best_ha, use_margin=True)
    df["elo_home"] = [x[0] for x in elo_pairs_all]
    df["elo_away"] = [x[1] for x in elo_pairs_all]

    # Re-split AFTER Elo exists
    train_fit = df[df["date"] < "2022-07-01"]
    val = df[(df["date"] >= "2022-07-01") & (df["date"] < "2023-07-01")]
    test = df[df["date"] >= "2023-07-01"]

    # -------------------------
    # Tune decay on VAL (keep best Elo/beta fixed)
    # -------------------------
    best_decay = None
    best_decay_ll = float("inf")

    for decay in decays:
        league_avg_home_d, league_avg_away_d, attack_d, defense_d = \
            fit_team_strengths_weighted(train_fit, decay=decay)

        probs = []
        y_val2 = []

        for _, row in val.iterrows():
            lam_h, lam_a = predict_lambdas(
                row["home_team"], row["away_team"],
                league_avg_home_d, league_avg_away_d,
                attack_d, defense_d
            )

            lam_h, lam_a = apply_elo_to_lambdas(
                lam_h, lam_a,
                row["elo_home"], row["elo_away"],
                beta=best_beta
            )

            pH, pD, pA = match_outcome_probs(lam_h, lam_a)
            probs.append([pH, pD, pA])

            if row["home_goals"] > row["away_goals"]:
                y_val2.append(0)
            elif row["home_goals"] == row["away_goals"]:
                y_val2.append(1)
            else:
                y_val2.append(2)

        probs = np.array(probs)
        y_val2 = np.array(y_val2)
        ll_d = log_loss(y_val2, probs)

        if ll_d < best_decay_ll:
            best_decay_ll = ll_d
            best_decay = decay

    print("\nBest decay (VAL LogLoss):", best_decay, "VAL LogLoss:", round(best_decay_ll, 4))

    # -------------------------
    # Final training with best decay
    # -------------------------
    league_avg_home, league_avg_away, attack, defense = \
        fit_team_strengths_weighted(train_fit, decay=best_decay)

    rho = fit_rho(
        train_fit,
        league_avg_home, league_avg_away,
        attack, defense,
        best_K, best_ha, True, best_beta
    )
    print("Fitted rho:", round(rho, 3))

    # Raw probs
    val_probs_raw, y_val = get_probs(val, league_avg_home, league_avg_away, attack, defense, best_beta, rho)
    test_probs_raw, y_test = get_probs(test, league_avg_home, league_avg_away, attack, defense, best_beta, rho)

    # Temperature scaling
    T = fit_temperature(val_probs_raw, y_val)
    print("Fitted temperature:", round(T, 3))

    test_probs_cal = temperature_scale_probs(test_probs_raw, T)

    # Metrics
    print("\n--- TEST METRICS ---")

    print("\nRaw:")
    print("LogLoss:", round(log_loss(y_test, test_probs_raw), 4))
    print("Brier:", round(multiclass_brier(test_probs_raw, y_test), 4))
    print("ECE:", round(top_label_ece(test_probs_raw, y_test), 4))

    print("\nCalibrated:")
    print("LogLoss:", round(log_loss(y_test, test_probs_cal), 4))
    print("Brier:", round(multiclass_brier(test_probs_cal, y_test), 4))
    print("ECE:", round(top_label_ece(test_probs_cal, y_test), 4))


if __name__ == "__main__":
    main()