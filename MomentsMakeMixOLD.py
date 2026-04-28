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

def pack_theta(weights, means, variances):
    """
    Pack physical mixture parameters into the unconstrained theta vector.

    Parameters
    ----------
    weights : array-like, shape (n,)
        Positive weights that sum to 1.
    means : array-like, shape (n,)
        Component means.
    variances : array-like, shape (n,)
        Strictly positive component variances.

    Returns
    -------
    theta : ndarray, shape (3*n - 1,)
        [z_1 ... z_{n-1} | mu_1 ... mu_n | l_1 ... l_n]
        where z are weight logits and l are log-variances.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(means, dtype=float)
    v = np.asarray(variances, dtype=float)

    n = len(w)
    assert len(mu) == n and len(v) == n, "All inputs must have the same length n"
    assert np.isclose(np.sum(w), 1.0), "Weights must sum to 1"
    assert np.all(w > 0), "Weights must be strictly positive"
    assert np.all(v > 0), "Variances must be strictly positive"

    # --- weights to logits ---
    # softmax inverse: z_i = log(w_i) - log(w_n) for i = 1..n-1
    # (the n-th component is the implicit reference with z_n = 0)
    z = np.log(w[:-1]) - np.log(w[-1])

    # --- variances to log-variances ---
    l = np.log(v)

    # --- pack ---
    theta = np.concatenate([z, mu, l])
    return theta

def unpack_theta(theta, n):
    """
    Unpack theta into physical parameters.

    Returns
    -------
    w : ndarray, shape (n,)
    mu : ndarray, shape (n,)
    v : ndarray, shape (n,)
    """
    z = theta[:n-1]
    mu = theta[n-1:2*n-1]
    l = theta[2*n-1:3*n-1]

    # stable softmax
    z_full = np.concatenate([z, [0.0]])
    z_full = z_full - np.max(z_full)
    w = np.exp(z_full)
    w = w / np.sum(w)

    v = np.exp(l)

    return w, mu, v



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


# ------------------------------------------------------------------
# 3. Residuals for the least-squares solver
# ------------------------------------------------------------------
def residuals(theta, M_given, n):
    """
    Returns the vector (M_mix(theta) - M_given).
    M_given must have length 3n-1.
    """
    return mixture_moments(theta, n) - M_given


# ------------------------------------------------------------------
# 4. Example driver (n = 3, so 8 moments)
# ------------------------------------------------------------------
if __name__ == "__main__":
    n = 3
    M_given = np.array([1.0, 2.5, 4.0, 10.0, 25.0, 70.0, 200.0, 600.0])          # length 3n-1 = 8, your target moments

    # Random multi-start is strongly recommended because of permutation
    # symmetry and local minima.
    best_cost = np.inf
    best_sol  = None

    for trial in range(50):
        # heuristic initial spread
        theta0 = np.concatenate([
            np.random.randn(n-1),      # weight logits
            np.random.randn(n),        # means
            np.random.randn(n) - 1.0   # log-variances (variances ~ 0.4)
        ])

        sol = least_squares(residuals, theta0, args=(M_given, n), method='lm')

        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol

    # unpack best solution
    theta_opt = best_sol.x
    z_opt  = theta_opt[:n-1]
    mu_opt = theta_opt[n-1:2*n-1]
    l_opt  = theta_opt[2*n-1:3*n-1]

    z_full = np.concatenate([z_opt, [0.0]])
    w_opt  = np.exp(z_full - np.max(z_full))
    w_opt  = w_opt / np.sum(w_opt)
    v_opt  = np.exp(l_opt)

    print("Weights :", w_opt)
    print("Means   :", mu_opt)
    print("Variances:", v_opt)


"""theta0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print(gaussian_moments(1, 1, 5))
print(mixture_moments(theta0, 3))
"""
# To be continued