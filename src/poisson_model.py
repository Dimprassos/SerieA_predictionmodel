from __future__ import annotations

import numpy as np
import pandas as pd
import math
from math import exp, factorial
from typing import List, Tuple


def fit_team_strengths(train: pd.DataFrame, eps: float = 1e-6):
    """
    Returns:
      league_avg_home, league_avg_away,
      attack (dict), defense (dict)

    attack: goals scored per match / league avg total goals per match
    defense: goals conceded per match / league avg total goals per match
             (>1 means concedes more than average -> weaker defense)
    """
    league_avg_home = train["home_goals"].mean()
    league_avg_away = train["away_goals"].mean()
    league_avg_total = (league_avg_home + league_avg_away) / 2.0

    # Goals scored by team (home + away)
    home_scored = train.groupby("home_team")["home_goals"].sum()
    away_scored = train.groupby("away_team")["away_goals"].sum()
    scored = home_scored.add(away_scored, fill_value=0)

    # Goals conceded by team (home conceded away_goals, away conceded home_goals)
    home_conceded = train.groupby("home_team")["away_goals"].sum()
    away_conceded = train.groupby("away_team")["home_goals"].sum()
    conceded = home_conceded.add(away_conceded, fill_value=0)

    # Matches played by team (home matches + away matches)
    home_matches = train.groupby("home_team").size()
    away_matches = train.groupby("away_team").size()
    matches = home_matches.add(away_matches, fill_value=0)

    avg_scored = scored / matches
    avg_conceded = conceded / matches

    attack = (avg_scored / (league_avg_total + eps)).to_dict()
    defense = (avg_conceded / (league_avg_total + eps)).to_dict()

    return league_avg_home, league_avg_away, attack, defense

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)



def match_outcome_probs_dc(lam_h: float, lam_a: float, rho: float, max_goals: int = 10) -> Tuple[float, float, float]:
    """
    Computes P(Home win), P(Draw), P(Away win) by summing DC-corrected scoreline probs.
    """
    P = scoreline_probs_dc(lam_h, lam_a, rho, max_goals=max_goals)

    pH = 0.0
    pD = 0.0
    pA = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            if hg > ag:
                pH += P[hg][ag]
            elif hg == ag:
                pD += P[hg][ag]
            else:
                pA += P[hg][ag]
    return pH, pD, pA


def predict_lambdas(home_team: str, away_team: str,
                    league_avg_home: float, league_avg_away: float,
                    attack: dict, defense: dict):
    ah = attack.get(home_team, 1.0)
    dh = defense.get(home_team, 1.0)
    aa = attack.get(away_team, 1.0)
    da = defense.get(away_team, 1.0)

    lam_home = league_avg_home * ah * da
    lam_away = league_avg_away * aa * dh
    return lam_home, lam_away

def apply_elo_to_lambdas(lam_home: float, lam_away: float, elo_home: float, elo_away: float, beta: float = 0.15):
    d = (elo_home - elo_away) / 400.0
    lam_home_adj = lam_home * math.exp(beta * d)
    lam_away_adj = lam_away * math.exp(-beta * d)

    # safety floor
    lam_home_adj = max(lam_home_adj, 0.05)
    lam_away_adj = max(lam_away_adj, 0.05)

    return lam_home_adj, lam_away_adj


def scoreline_probs_dc(lam_h: float, lam_a: float, rho: float, max_goals: int = 10) -> List[List[float]]:
    """
    Returns matrix P[h][a] for h,a=0..max_goals with Dixon–Coles correction,
    normalized to sum to 1 over the truncated grid.
    """
    P = [[0.0 for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
    total = 0.0

    for hg in range(max_goals + 1):
        ph = poisson_pmf(hg, lam_h)
        for ag in range(max_goals + 1):
            pa = poisson_pmf(ag, lam_a)
            tau = dixon_coles_tau(hg, ag, lam_h, lam_a, rho)
            p = ph * pa * tau
            if p < 0:
                p = 0.0  # safety
            P[hg][ag] = p
            total += p

    # normalize (because truncation + tau can change total mass)
    if total > 0:
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                P[hg][ag] /= total

    return P

def top_k_scorelines_dc(lam_h: float, lam_a: float, rho: float, k: int = 5, max_goals: int = 10):
    """
    Top-k most likely scorelines under Dixon–Coles corrected distribution.
    Returns [((hg,ag), prob), ...]
    """
    P = scoreline_probs_dc(lam_h, lam_a, rho, max_goals=max_goals)
    pairs = []
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            pairs.append(((hg, ag), P[hg][ag]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

def fit_team_strengths_home_away(train: pd.DataFrame, eps: float = 1e-6):
    """
    Home/Away team strengths.

    Returns:
      league_avg_home, league_avg_away,
      attack_home, defense_home, attack_away, defense_away (all dicts)

    Note: defense_* here is "weakness": >1 means concedes more than average.
    """
    league_avg_home = train["home_goals"].mean()
    league_avg_away = train["away_goals"].mean()

    # Home-side stats
    home_scored = train.groupby("home_team")["home_goals"].sum()
    home_conceded = train.groupby("home_team")["away_goals"].sum()
    home_matches = train.groupby("home_team").size()

    # Away-side stats
    away_scored = train.groupby("away_team")["away_goals"].sum()
    away_conceded = train.groupby("away_team")["home_goals"].sum()
    away_matches = train.groupby("away_team").size()

    teams = sorted(set(train["home_team"]) | set(train["away_team"]))

    # per-match averages
    home_attack_avg = (home_scored / home_matches).reindex(teams).fillna(league_avg_home)
    home_def_avg = (home_conceded / home_matches).reindex(teams).fillna(league_avg_away)

    away_attack_avg = (away_scored / away_matches).reindex(teams).fillna(league_avg_away)
    away_def_avg = (away_conceded / away_matches).reindex(teams).fillna(league_avg_home)

    # normalize relative to league averages for that side
    attack_home = (home_attack_avg / (league_avg_home + eps)).to_dict()
    defense_home = (home_def_avg / (league_avg_away + eps)).to_dict()

    attack_away = (away_attack_avg / (league_avg_away + eps)).to_dict()
    defense_away = (away_def_avg / (league_avg_home + eps)).to_dict()

    return league_avg_home, league_avg_away, attack_home, defense_home, attack_away, defense_away


def predict_lambdas_home_away(
    home_team: str,
    away_team: str,
    league_avg_home: float,
    league_avg_away: float,
    attack_home: dict,
    defense_home: dict,
    attack_away: dict,
    defense_away: dict,
):
    ah = attack_home.get(home_team, 1.0)
    dh = defense_home.get(home_team, 1.0)
    aa = attack_away.get(away_team, 1.0)
    da = defense_away.get(away_team, 1.0)

    lam_home = league_avg_home * ah * da
    lam_away = league_avg_away * aa * dh
    return lam_home, lam_away



def dixon_coles_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon–Coles correction factor tau for low-score dependence.
    Only affects (0,0), (0,1), (1,0), (1,1).
    """
    if hg == 0 and ag == 0:
        return 1.0 - (lam_h * lam_a * rho)
    if hg == 0 and ag == 1:
        return 1.0 + (lam_h * rho)
    if hg == 1 and ag == 0:
        return 1.0 + (lam_a * rho)
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0

def match_outcome_probs(lam_h: float, lam_a: float, max_goals: int = 10):
    """
    Classic independent Poisson model:
    P(Home win), P(Draw), P(Away win) by summing scorelines up to max_goals.
    """
    pH = pD = pA = 0.0
    for hg in range(max_goals + 1):
        ph = poisson_pmf(hg, lam_h)
        for ag in range(max_goals + 1):
            pa = poisson_pmf(ag, lam_a)
            p = ph * pa
            if hg > ag:
                pH += p
            elif hg == ag:
                pD += p
            else:
                pA += p
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH / s, pD / s, pA / s
    return pH, pD, pA

def fit_team_strengths_weighted(df: pd.DataFrame, decay: float = 0.0015):
    """
    Recency-weighted Poisson team strength estimation.
    decay: exponential time decay parameter (per day)
    """

    df = df.copy()
    df = df.sort_values("date")

    max_date = df["date"].max()

    # compute time difference in days
    df["days_ago"] = (max_date - df["date"]).dt.days

    # exponential weights
    df["weight"] = np.exp(-decay * df["days_ago"])

    teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    attack = {team: 1.0 for team in teams}
    defense = {team: 1.0 for team in teams}

    league_avg_home = np.average(df["home_goals"], weights=df["weight"])
    league_avg_away = np.average(df["away_goals"], weights=df["weight"])

    for _ in range(10):  # iterative updates
        for team in teams:
            home_mask = df["home_team"] == team
            away_mask = df["away_team"] == team

            # weighted attack
            num = np.sum(df.loc[home_mask, "weight"] * df.loc[home_mask, "home_goals"])
            den = np.sum(
                df.loc[home_mask, "weight"]
                * league_avg_home
                * np.array([defense[opp] for opp in df.loc[home_mask, "away_team"]])
            )
            if den > 0:
                attack[team] = num / den

            # weighted defense
            num = np.sum(df.loc[away_mask, "weight"] * df.loc[away_mask, "home_goals"])
            den = np.sum(
                df.loc[away_mask, "weight"]
                * league_avg_home
                * np.array([attack[opp] for opp in df.loc[away_mask, "home_team"]])
            )
            if den > 0:
                defense[team] = num / den

    return league_avg_home, league_avg_away, attack, defense

def fit_team_strengths_home_away_weighted(df: pd.DataFrame, decay: float = 0.001):
    """
    Weighted strengths αλλά με 4 vectors:
    - attack_home[team], defense_home[team]
    - attack_away[team], defense_away[team]
    """
    df = df.copy()
    df = df.sort_values("date")

    last_date = df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.astype(float)
    df["weight"] = np.exp(-decay * df["days_ago"])

    teams = pd.Index(sorted(set(df["home_team"]).union(set(df["away_team"]))))

    # Weighted league averages
    w_sum = df["weight"].sum()
    league_avg_home = (df["weight"] * df["home_goals"]).sum() / w_sum
    league_avg_away = (df["weight"] * df["away_goals"]).sum() / w_sum

    # Weighted aggregates per team
    home_for = (df["weight"] * df["home_goals"]).groupby(df["home_team"]).sum().reindex(teams, fill_value=0.0)
    home_against = (df["weight"] * df["away_goals"]).groupby(df["home_team"]).sum().reindex(teams, fill_value=0.0)
    home_w = df["weight"].groupby(df["home_team"]).sum().reindex(teams, fill_value=0.0)

    away_for = (df["weight"] * df["away_goals"]).groupby(df["away_team"]).sum().reindex(teams, fill_value=0.0)
    away_against = (df["weight"] * df["home_goals"]).groupby(df["away_team"]).sum().reindex(teams, fill_value=0.0)
    away_w = df["weight"].groupby(df["away_team"]).sum().reindex(teams, fill_value=0.0)

    eps = 1e-9

    # Expected goals given league avg & match weights
    exp_home_for = league_avg_home * home_w
    exp_home_against = league_avg_away * home_w

    exp_away_for = league_avg_away * away_w
    exp_away_against = league_avg_home * away_w

    attack_home = (home_for + eps) / (exp_home_for + eps)
    defense_home = (home_against + eps) / (exp_home_against + eps)

    attack_away = (away_for + eps) / (exp_away_for + eps)
    defense_away = (away_against + eps) / (exp_away_against + eps)

    # Optional: clamp to avoid extreme lambdas early-season
    def _clip(s):
        return s.clip(lower=0.3, upper=3.0)

    return (
        float(league_avg_home), float(league_avg_away),
        _clip(attack_home).to_dict(), _clip(defense_home).to_dict(),
        _clip(attack_away).to_dict(), _clip(defense_away).to_dict()
    )


def predict_lambdas_home_away(
    home_team: str, away_team: str,
    league_avg_home: float, league_avg_away: float,
    attack_home: dict, defense_home: dict,
    attack_away: dict, defense_away: dict,
):
    ah = attack_home.get(home_team, 1.0)
    dh = defense_home.get(home_team, 1.0)
    aa = attack_away.get(away_team, 1.0)
    da = defense_away.get(away_team, 1.0)

    lam_h = league_avg_home * ah * da
    lam_a = league_avg_away * aa * dh

    lam_h = max(0.05, float(lam_h))
    lam_a = max(0.05, float(lam_a))
    return lam_h, lam_a