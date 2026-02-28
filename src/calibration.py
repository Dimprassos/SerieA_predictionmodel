import numpy as np

def safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return np.log(p)

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def temperature_scale_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """
    probs: (N,3) raw probabilities
    Returns calibrated probabilities using temperature scaling on logits.
    """
    logits = safe_logit(probs)
    scaled = logits / max(T, 1e-6)
    return softmax(scaled)

def fit_temperature(probs_val: np.ndarray, y_val: np.ndarray, T_grid=None) -> float:
    """
    Finds T that minimizes negative log-likelihood on validation.
    """
    if T_grid is None:
        T_grid = np.arange(0.5, 3.01, 0.05)

    best_T = 1.0
    best_nll = float("inf")

    for T in T_grid:
        p = temperature_scale_probs(probs_val, float(T))
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1.0)))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return best_T