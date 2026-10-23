import numpy as np


def _unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def similarity(a, b):
    """Dynamic Time Warping over cosine distance. Returns ~0..1 (1 = identical)."""
    cost = 1 - _unit(a) @ _unit(b).T
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    S = np.zeros((n + 1, n + 1))
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            d, s = min((D[i - 1, j], S[i - 1, j]), (D[i, j - 1], S[i, j - 1]), (D[i - 1, j - 1], S[i - 1, j - 1]))
            D[i, j] = cost[i - 1, j - 1] + d
            S[i, j] = s + 1
    return float(1 - D[n, n] / S[n, n])


def percent(sim):
    """Hand poses are all fairly alike, so raw cosine is inflated. Stretch 0.6..1.0 to 0..100 (tunable)."""
    return round(float(np.clip((sim - 0.6) / 0.4, 0, 1)) * 100, 1)


def feedback(score):
    if score >= 85: return "Excellent. That is very close to the reference."
    if score >= 65: return "Good. Check your finger shapes and the speed of the movement."
    if score >= 40: return "Getting there. Watch the reference again and match the hand shape first."
    return "Not close yet. Keep your hands fully in frame and try again."
