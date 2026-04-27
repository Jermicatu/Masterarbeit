import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger
from scipy.optimize import least_squares

# ------------------------------------------------------------------
# 1. Forward moments via recurrence
# ------------------------------------------------------------------
def gaussian_moments(mu, v, max_moment):
    """
    Raw non-central moments M_1 .. M_{max_moment} of N(mu, v).
    Uses the recurrence:  M_k = mu * M_{k-1} + (k-1) * v * M_{k-2}
    """
    m = np.zeros(max_moment)
    if max_moment >= 1:
        m[0] = mu                       # M_1
    if max_moment >= 2:
        m[1] = mu**2 + v                # M_2
    for k in range(2, max_moment):      # index k stores M_{k+1}
        m[k] = mu * m[k-1] + k * v * m[k-2]
    return m

# ------------------------------------------------------------------
# 2. Mixture moments from the unconstrained parameter vector
# ------------------------------------------------------------------
def mixture_moments(theta, n):
    """
    theta layout (length 3n-1):
        [z_1 ... z_{n-1} | mu_1 ... mu_n | l_1 ... l_n]
    """
    # unpack
    z  = theta[:n-1]
    mu = theta[n-1:2*n-1]
    l  = theta[2*n-1:3*n-1]

    # --- weights: softmax on (n-1) logits with an implicit 0 for the n-th ---
    z_full = np.concatenate([z, [0.0]])
    z_full = z_full - np.max(z_full)          # numerical stability
    w = np.exp(z_full)
    w = w / np.sum(w)

    # --- variances (strictly positive) ---
    v = np.exp(l)

    # --- mixture moments ---
    max_moment = 3 * n - 1
    M_mix = np.zeros(max_moment)
    for i in range(n):
        M_mix += w[i] * gaussian_moments(mu[i], v[i], max_moment)
    return M_mix


theta0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print(gaussian_moments(1, 1, 5))
print(mixture_moments(theta0, 3))

# To be continued