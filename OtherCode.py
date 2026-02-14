import numpy as np
from scipy.stats import norm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================
# Gaussian mixture PDF
# ======================================================

def gmm_pdf(x, w, m, s):
    x = x[:, None]
    return np.sum(w * norm.pdf(x, m, s), axis=1)

# ======================================================
# Progressive target density (normalized)
# ======================================================

def progressive_density(x, gamma, w1, m1, s1, w2, m2, s2):
    f1 = gmm_pdf(x, w1, m1, s1)
    f2 = gmm_pdf(x, w2, m2, s2)
    un = f2 * (f1 ** gamma)
    return un / np.trapezoid(un, x)

# ======================================================
# log f1 for RHS
# ======================================================

def log_f1(x, w1, m1, s1):
    return np.log(gmm_pdf(x, w1, m1, s1) + 1e-12)

# ======================================================
# Exact Progressive Bayes ODE
# ======================================================

def rhs_exact(gamma, eta, x, w1, m1, s1, w2, m2, s2, K):
    w = np.abs(eta[:K])
    w /= np.sum(w)
    m = eta[K:2*K]
    s = np.abs(eta[2*K:3*K])

    fhat = gmm_pdf(x, w, m, s)
    fhat /= np.trapezoid(fhat, x)

    f = progressive_density(x, gamma, w1, m1, s1, w2, m2, s2)

    L1 = log_f1(x, w1, m1, s1)
    E_L1 = np.trapezoid(f * L1, x)
    dfdg = f * (L1 - E_L1)

    # basis functions
    Phi = []
    for k in range(K):
        Nk = norm.pdf(x, m[k], s[k])
        Phi.append(Nk)
        Phi.append(w[k] * (x - m[k]) / s[k]**2 * Nk)
        Phi.append(w[k] * ((x - m[k])**2 / s[k]**3 - 1/s[k]) * Nk)

    Phi = np.array(Phi)

    # M matrix
    M = np.zeros((3*K, 3*K))
    for i in range(3*K):
        for j in range(3*K):
            M[i, j] = np.trapezoid(Phi[i] * Phi[j], x)

    # b vector
    b = np.array([np.trapezoid(dfdg * Phi[i], x) for i in range(3*K)])

    # solve linear system
    d_eta = np.linalg.solve(M + 1e-8*np.eye(3*K), b)
    return d_eta

# ======================================================
# Page-3 example
# ======================================================

w1 = np.array([1.0])
m1 = np.array([-1.0])
s1 = np.array([1.0])

w2 = np.array([1.0])
m2 = np.array([1.0])
s2 = np.array([1.0])
x = np.linspace(-5, 5, 1500)

# ======================================================
# Integrate ODE (4 components)
# ======================================================

K = 1
eta0 = np.concatenate([
    np.ones(K) / K,
    np.linspace(-2, 2, K),
    np.ones(K)
])

sol = solve_ivp(
    rhs_exact,
    (0, 1),
    eta0,
    t_eval=[1.0],
    args=(x, w1, m1, s1, w2, m2, s2, K),
    rtol=1e-6,
    atol=1e-8
)

eta = sol.y[:, -1]
w4 = np.abs(eta[:K]); w4 /= np.sum(w4)
m4 = eta[K:2*K]
s4 = np.abs(eta[2*K:3*K])

approx = gmm_pdf(x, w4, m4, s4)
approx /= np.trapezoid(approx, x)

# ======================================================
# Exact 6-component product
# ======================================================

exact = np.zeros_like(x)
for i in range(len(w1)):
    for j in range(len(w2)):
        exact += w1[i] * w2[j] * \
                 norm.pdf(x, m1[i], s1[i]) * \
                 norm.pdf(x, m2[j], s2[j])
exact /= np.trapezoid(exact, x)

# ======================================================
# Plot
# ======================================================

plt.figure(figsize=(10,4))
plt.plot(x, exact, label="Exact product (6 components)", lw=2)
plt.plot(x, approx, "--", label="Exact Progressive Bayes ODE (4 comp.)", lw=2)
plt.legend()
plt.xlabel("x")
plt.ylabel("density")
plt.title("Exact Progressive Gaussian Mixture Reduction")
plt.tight_layout()
plt.show()