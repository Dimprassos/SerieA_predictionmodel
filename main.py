import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from src.data_processing import load_and_merge_data, train_test_split
from src.elo import compute_elo_ratings
from src.poisson_model import (
    fit_team_strengths,
    predict_lambdas,
    apply_elo_to_lambdas,
    match_outcome_probs,        # plain Poisson for grid search
    match_outcome_probs_dc,     # Dixon–Coles for final evaluation
    scoreline_probs_dc,         # needed for fitting rho
)


def evaluate_with_elo_params(
    full_df: pd.DataFrame,
    split_date: str,
    K: float,
    home_adv: float,
    use_margin: bool,
    beta: float,
    window_years: int = 3,
):
    """
    Grid-search objective for Elo params (K, home_adv, use_margin, beta).
    IMPORTANT: uses plain Poisson outcome probs (NOT Dixon–Coles),
    because rho is not fitted yet during this stage.
    """
    elo_pairs = compute_elo_ratings(full_df, K=K, home_adv=home_adv, use_margin=use_margin)
    df2 = full_df.copy()
    df2["elo_home"] = [x[0] for x in elo_pairs]
    df2["elo_away"] = [x[1] for x in elo_pairs]

    train_full = df2[df2["date"] < split_date]
    test_full = df2[df2["date"] >= split_date]

    split_dt = pd.to_datetime(split_date)
    window_start = split_dt - pd.DateOffset(years=window_years)
    train_recent = train_full[train_full["date"] >= window_start]

    league_avg_home, league_avg_away, attack, defense = fit_team_strengths(train_recent)

    y_true = []
    y_pred_probs = []
    y_pred_labels = []

    for _, row in test_full.iterrows():
        ht, at = row["home_team"], row["away_team"]

        lam_h, lam_a = predict_lambdas(ht, at, league_avg_home, league_avg_away, attack, defense)
        lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, row["elo_home"], row["elo_away"], beta=beta)

        # Poisson (no DC here!)
        pH, pD, pA = match_outcome_probs(lam_h, lam_a)

        y_pred_probs.append([pH, pD, pA])
        y_pred_labels.append(int(np.argmax([pH, pD, pA])))

        if row["home_goals"] > row["away_goals"]:
            y_true.append(0)
        elif row["home_goals"] == row["away_goals"]:
            y_true.append(1)
        else:
            y_true.append(2)

    y_pred_probs = np.array(y_pred_probs)
    ll = log_loss(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred_labels)
    return ll, acc


def fit_rho_dc(
    train_df: pd.DataFrame,
    league_avg_home: float,
    league_avg_away: float,
    attack: dict,
    defense: dict,
    K: float,
    home_adv: float,
    use_margin: bool,
    beta: float,
    max_goals: int = 10,
):
    """
    Fit Dixon–Coles rho by grid-search on TRAIN data only.
    Uses the chosen Elo params + beta + Poisson strengths.
    """
    # Elo pre-match ratings within train_df (chronological)
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

            lam_h, lam_a = predict_lambdas(ht, at, league_avg_home, league_avg_away, attack, defense)
            lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, row["elo_home"], row["elo_away"], beta=beta)

            hg = int(row["home_goals"])
            ag = int(row["away_goals"])

            P = scoreline_probs_dc(lam_h, lam_a, rho, max_goals=max_goals)

            if hg <= max_goals and ag <= max_goals:
                p = P[hg][ag]
            else:
                p = 1e-12  # out of truncation grid

            if p <= 0:
                ok = False
                break

            nll -= np.log(p)

        if ok and nll < best_nll:
            best_nll = nll
            best_rho = float(rho)

    return best_rho, best_nll


def main():
    df = load_and_merge_data()
    train, test = train_test_split(df)

    print("Total matches:", len(df))
    print("Train matches:", len(train))
    print("Test matches:", len(test))

    avg_home = train["home_goals"].mean()
    avg_away = train["away_goals"].mean()
    print("Average home goals:", round(avg_home, 3))
    print("Average away goals:", round(avg_away, 3))
    print("Home advantage:", round(avg_home - avg_away, 3))

    full_df = df.copy()
    split_date = "2023-07-01"
    window_years = 3

    # -------------------------
    # Elo grid search (Poisson outcome probs)
    # -------------------------
    print("\n--- Elo Grid Search (recent-window strengths) ---")

    Ks = list(range(35, 71, 5))                 # 35..70 step 5
    home_advs = list(range(50, 111, 10))        # 50..110 step 10
    betas = [round(x, 2) for x in np.arange(0.08, 0.141, 0.01)]  # 0.08..0.14
    use_margins = [True]

    best_ll = None   # (ll, acc, K, ha, um, beta)
    best_acc = None  # (ll, acc, K, ha, um, beta)

    for K in Ks:
        for ha in home_advs:
            for um in use_margins:
                for b in betas:
                    ll, acc = evaluate_with_elo_params(
                        full_df, split_date, K, ha, um, b, window_years=window_years
                    )
                    if (best_ll is None) or (ll < best_ll[0]):
                        best_ll = (ll, acc, K, ha, um, b)
                    if (best_acc is None) or (acc > best_acc[1]):
                        best_acc = (ll, acc, K, ha, um, b)

    # window stats
    train_full_for_print = full_df[full_df["date"] < split_date]
    split_dt = pd.to_datetime(split_date)
    window_start = split_dt - pd.DateOffset(years=window_years)
    train_recent_for_print = train_full_for_print[train_full_for_print["date"] >= window_start]

    print("\nRecent window training:")
    print("Window start:", window_start.date())
    print("Train_full:", len(train_full_for_print), "Train_recent:", len(train_recent_for_print))

    print("\nBest by LogLoss:")
    print("  LogLoss:", round(best_ll[0], 4), "Accuracy:", round(best_ll[1], 4))
    print("  K:", best_ll[2], "home_adv:", best_ll[3], "use_margin:", best_ll[4], "beta:", best_ll[5])

    print("\nBest by Accuracy:")
    print("  LogLoss:", round(best_acc[0], 4), "Accuracy:", round(best_acc[1], 4))
    print("  K:", best_acc[2], "home_adv:", best_acc[3], "use_margin:", best_acc[4], "beta:", best_acc[5])

    # choose final config
    chosen = best_ll  # change to best_acc if you prefer accuracy
    chosen_ll, chosen_acc, best_K, best_ha, best_um, best_beta = chosen

    print("\nChosen config:")
    print("  LogLoss:", round(chosen_ll, 4), "Accuracy:", round(chosen_acc, 4))
    print("  K:", best_K, "home_adv:", best_ha, "use_margin:", best_um, "beta:", best_beta)

    # -------------------------
    # Build df with chosen Elo
    # -------------------------
    elo_pairs = compute_elo_ratings(full_df, K=best_K, home_adv=best_ha, use_margin=best_um)
    full_df["elo_home"] = [x[0] for x in elo_pairs]
    full_df["elo_away"] = [x[1] for x in elo_pairs]

    train_full = full_df[full_df["date"] < split_date]
    test_full = full_df[full_df["date"] >= split_date]

    split_dt = pd.to_datetime(split_date)
    window_start = split_dt - pd.DateOffset(years=window_years)
    train_recent = train_full[train_full["date"] >= window_start]

    # strengths on recent window
    league_avg_home, league_avg_away, attack, defense = fit_team_strengths(train_recent)

    # -------------------------
    # Fit Dixon–Coles rho on TRAIN ONLY
    # -------------------------
    rho, rho_nll = fit_rho_dc(
        train_recent,
        league_avg_home, league_avg_away, attack, defense,
        K=best_K, home_adv=best_ha, use_margin=best_um, beta=best_beta,
        max_goals=10
    )
    print("\nFitted Dixon-Coles rho:", round(rho, 3))

    # -------------------------
    # Final evaluation on test (WITH Dixon–Coles)
    # -------------------------
    y_true = []
    y_pred_probs = []
    y_pred_labels = []

    for _, row in test_full.iterrows():
        ht, at = row["home_team"], row["away_team"]

        lam_h, lam_a = predict_lambdas(ht, at, league_avg_home, league_avg_away, attack, defense)
        lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, row["elo_home"], row["elo_away"], beta=best_beta)

        pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=10)

        y_pred_probs.append([pH, pD, pA])
        y_pred_labels.append(int(np.argmax([pH, pD, pA])))

        if row["home_goals"] > row["away_goals"]:
            y_true.append(0)
        elif row["home_goals"] == row["away_goals"]:
            y_true.append(1)
        else:
            y_true.append(2)

    y_pred_probs = np.array(y_pred_probs)
    ll = log_loss(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred_labels)

    print("\n--- Final Test Evaluation (CHOSEN + Dixon-Coles) ---")
    print("Log Loss:", round(ll, 4))
    print("Accuracy:", round(acc, 4))

    # Demo match
    demo = test_full.iloc[0]
    ht, at = demo["home_team"], demo["away_team"]
    lam_h, lam_a = predict_lambdas(ht, at, league_avg_home, league_avg_away, attack, defense)
    lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, demo["elo_home"], demo["elo_away"], beta=best_beta)
    pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=10)

    print("\nExample match:", ht, "vs", at)
    print("Lambdas:", round(lam_h, 3), round(lam_a, 3))
    print("1X2 probs:", round(pH, 3), round(pD, 3), round(pA, 3))
    print("Actual score:", int(demo["home_goals"]), "-", int(demo["away_goals"]))

    last_date = full_df["date"].max()
    print("\nData last date:", last_date)

if __name__ == "__main__":
    main()