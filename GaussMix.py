import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import random

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp

def f1(x):
  return 0.45 * (cos(x+1) + 1)

def f2(x):
  return 0.5 * (sig(x+1) - sig(x-1) + sig(x))

def f3(x):
  return cos(x)

def f4(x):
  return x**3

def id(x):
  return x

def f5(x):
  return torch.sum(x)

def Gauss_approx(f_w, f_mean, f_var, g_w, g_mean, g_var):
  sol = 0

  N = f_mean.shape[0]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = torch.float32

  P = torch.empty(N, N, 3, 3, device=device, dtype=dtype)

  for i in range(0,N):
    for j in range(0,N):
      # TODO richtige formeln
      P[i, j, 0, 0] = 1.0
      P[i, j, 0, 1] = 0.0
      P[i, j, 0, 2] = f_mean[i]

      P[i, j, 1, 0] = g_mean[i]
      P[i, j, 1, 1] = g_mean[j]
      P[i, j, 1, 2] = g_mean[i]

      P[i, j, 2, 0] = f_mean[i]
      P[i, j, 2, 1] = 0.0
      P[i, j, 2, 2] = 1.0

  return P.permute(0, 2, 1, 3).reshape(5*3, 5*3)

def P_1(eta):
    w, mu, sigma = eta[0], eta[1], eta[2]
    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    N = mu.shape[0]
    mu_i = mu[:, None]
    mu_j = mu[None, :]
    sigma_i = sigma[:, None]
    sigma_j = sigma[None, :]
    w_i = w[:, None]
    w_j = w[None, :]

    # remember all sigmas are given in squared for (sigma**2)
    s2 = sigma_i + sigma_j
    mu_diff = mu_i - mu_j

    prefactor = (1.0 / torch.sqrt(2 * math.pi * s2)) * torch.exp(-0.5 * (mu_diff**2) / s2)
    

    s2_2 = s2**2
    s2_3 = s2**3
    s2_4 = s2**4
    md2 = mu_diff**2
    md4 = mu_diff**4

    P11 = torch.ones_like(mu_diff)
    P12 = w_j * mu_diff / s2
    P13 = w_j * torch.sqrt(sigma_j) * (md2 - s2_2) / (s2_2)
    P21 = w_i * (-mu_diff) / s2
    P22 = w_i * w_j * (s2 - md2) / (s2_2)
    P23 = w_i * w_j * torch.sqrt(sigma_j) * (-mu_diff) * (md2 - 3 * s2) / (s2_3)
    P31 = w_i * sigma_i * (md2 - s2_2) / (s2_2)
    P32 = w_i * w_j * torch.sqrt(sigma_i) * (-mu_diff) * (md2 - 3 * s2) / (s2_3)
    frac = md4 + 3.0 * s2 * (s2 - 2.0 * md2) / s2_4
    P33 = w_i * w_j * torch.sqrt(sigma_i) * torch.sqrt(sigma_j) * frac

    print(P22)
    print(prefactor)


    blocks = torch.stack([
        torch.stack([P11, P12, P13], dim=-1),
        torch.stack([P21, P22, P23], dim=-1),
        torch.stack([P31, P32, P33], dim=-1)
    ], dim=-2)

    blocks = blocks * prefactor[..., None, None]
    P_1 = blocks.permute(0, 2, 1, 3).reshape(3*N, 3*N)
    return P_1

def gaussian_mixture(x, eta):
    w, mu, sigma = eta[0], eta[1], eta[2]
    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    x = x[:, None]
    mu = mu[None, :]
    sigma = sigma[None, :]
    w = w[None, :]

    # Gaussian formula
    coef = 1.0 / torch.sqrt(2 * math.pi * sigma)
    exponent = torch.exp(-0.5 * (x - mu) ** 2 / sigma)
    gaussians = coef * exponent  # shape (K, L)

    # Weighted sum over components
    return (w * gaussians).sum(dim=1)

def gaussian_mix_gamma(x, eta, gamma, noise):
    w, mu, var = eta[0], eta[1], eta[2]

    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    factor = ((1 + noise)/(gamma+noise))**2

    x = x[:, None]
    means = mu[None, :]
    variances = var[None, :]
    weights = w[None, :]

    coef = 1.0 / torch.sqrt(2 * math.pi * variances)
    exponent = torch.exp(-0.5 * (x - means) ** 2 / (variances*factor))
    gaussians = coef * exponent  # shape (K, L)
    # Weighted sum over components
    return (weights * gaussians).sum(dim=1)

def gaussian_tilde(x, eta_1, eta_2, gamma, noise):
    gamma_mix_2 = gaussian_mixture(x, eta_2)
    gamma_mix_1 = gaussian_mix_gamma(x, eta_1, gamma, noise)

    return gamma_mix_1 * gamma_mix_2

def big_M(x, eta):
    w, mu, sigma = eta[0], eta[1], eta[2]
    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"
    
  
    K = x.shape[0]
    L = mu.shape[0]

    x_i = x[:, None]
    mu_i = mu[None, :]
    s2 = sigma[None, :]
    w_i = w[None, :]

    s = torch.sqrt(s2)
    s3 = s2*s
    s4 = s2**2
    s5 = s4*s
    s6 = s2**3

    d = x_i - mu_i                   # (K,L)
    d2 = d * d
    d3 = d * d2
    d4 = d2 * d2

    M11 = torch.zeros_like(d)
    M12 = d / (w_i * s2)
    M13 = (d2 - s2) / (w_i * s3)
    M21 = M12
    M22 = (d2 - s2) / s4
    M23 = (d3 - 3.0 * s2 * d) / s5
    M31 = M13
    M32 = M23
    M33 = (d4 - 5.0 * s2 * d2 + 2.0 * s4) / s6
    
    blocks = torch.stack([
      torch.stack([M11, M12, M13], dim=-1),
      torch.stack([M21, M22, M23], dim=-1),
      torch.stack([M31, M32, M33], dim=-1),
    ], dim=-2)

    f = gaussian_mixture(x, w, mu, sigma).reshape(-1, 1, 1, 1)
    blocks = blocks * f

    bigM = torch.zeros((K, L, L, 3, 3), dtype=blocks.dtype, device=blocks.device)
    diag_idx = torch.arange(L, device=blocks.device)
    bigM[:, diag_idx, diag_idx] = blocks
    bigM = bigM.permute(0, 1, 3, 2, 4).reshape(K, 3*L, 3*L)

    return bigM

def delta_P_integrand(x, eta_1, eta_2, eta_3, gamma, noise):
    return (gaussian_mixture(x, eta_1) - gaussian_tilde(x, eta_2, eta_3, gamma, noise)) * big_M(x, eta_1)

def gh_nodes_weights(n_nodes: int):
    # numpy hermgauss (nodes, weights) for weight e^{-x^2}
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_nodes)
    nodes = torch.as_tensor(nodes_np, dtype=torch.float32, device="cpu")   # shape (n,)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device="cpu") # shape (n,)
    return nodes, weights

def delta_P(n_nodes, eta_1, eta_2, eta_3, gamma, noise):
    nodes, weights = gh_nodes_weights(n_nodes)
   
    # n x m x m * n x 1 * n x 1
    return delta_P_integrand(nodes, eta_1, eta_2, eta_3, gamma, noise) * torch.exp(nodes**2).view(-1, 1, 1) * weights.view(-1, 1, 1)

def P(n_nodes, eta_1, eta_2, eta_3, gamma, noise):
   
    return P_1(eta_1) + delta_P(n_nodes, eta_1, eta_2, eta_3, gamma, noise)

def b_integrand(x, eta_1, eta_2, eta_3, gamma, noise):
    w_1, mu_1, sigma_1 = eta_1[0], eta_1[1], eta_1[2]
    w_2, mu_2, sigma_2 = eta_2[0], eta_2[1], eta_2[2]
    x_expanded = x.view(-1, 1)

    f_i = torch.zeros(w_1.shape[0])
    for i in range(0, w_1.shape[0]):
       eta_i = torch.stack([w_1[i], mu_1[i], var_1[i]])
       f_i[i] = gaussian_mixture(x, eta_i)

    inner_derivative = -(((x_expanded - mu_2)**2) / sigma_2) * ((1 + gamma)/((1 + noise)**2))

    factor = inner_derivative* gaussian_mixture(x, eta_2) * gaussian_mixture(x, eta_3) * f_i

    term1 = w_1**(-1).expand(x_expanded.shape[0], -1) * factor
    term2 = ((x_expanded - mu_1) / sigma_1 ) * factor
    term3 = (((x_expanded - mu_1)**2 - (sigma_1)) / (sigma_1 ** (3/2))) * factor

    stacked = torch.cat([term1, term2, term3], dim=1)

    return stacked

def b(n_nodes, eta_1, eta_2, eta_3, gamma, noise):
    nodes, weights = gh_nodes_weights(n_nodes)

    # n x m * n x 1 * n x 1
    return b_integrand(nodes, eta_1, eta_2, eta_3, gamma, noise) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)

def ode(n_nodes, eta_1, eta_2, eta_3, gamma, noise):
    return b(n_nodes, eta_1, eta_2, eta_3, gamma, noise) / P(n_nodes, eta_1, eta_2, eta_3, gamma, noise)

"""
eta_0 = [1.0]           
gamma_span = (0, 1)

f_w = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
f_mean = torch.tensor([1, 2, 3, 4, 5])
f_var = torch.tensor([1, 1, 1, 1, 1])
g_w = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
g_mean = torch.tensor([10, 20, 30, 40, 50])
g_var = torch.tensor([5, 5, 5, 5, 5])

print(Gauss_approx(f_w, f_mean, f_var, g_mean, g_var, g_w))

mu = torch.nn.Parameter(torch.tensor([0.0]))
sigma = torch.nn.Parameter(torch.tensor([2.0]))
w = torch.nn.Parameter(torch.tensor([1.0]))
eta = torch.stack([w, mu, sigma])
P_1 = P_1(eta)

print("big: ", P_1)

w = torch.tensor([0.2, 0.5, 0.3])
mu = torch.tensor([0.0, 2.0, -1.0])
var = torch.tensor([0.5, 0.2, 1.0])
eta = torch.stack([w, mu, var])
x_vals = torch.tensor([1, 0])

print("Gauss mix: ", gaussian_mixture(x_vals, eta))

w_1 = torch.tensor([0.2, 0.5, 0.3])
mu_1 = torch.tensor([0.0, 2.0, -1.0])
var_1 = torch.tensor([0.5, 0.2, 1.0])
eta_1 = torch.stack([w_1, mu_1, var_1])

w_2 = torch.tensor([0.1, 0.6, 0.3])
mu_2 = torch.tensor([0.0, 5.0, -10.0])
var_2 = torch.tensor([0.7, 0.1, 10.0])
eta_2 = torch.stack([w_2, mu_2, var_2])

# Evaluation points
x_vals = torch.tensor([1, 0])
gamma = 0.1
noise = 0.01

print("Gauss tilde: ", gaussian_tilde(x_vals, eta_1, eta_2, gamma, noise))

x_3 = torch.tensor([1])
w_3 = torch.tensor([0.5, 0.5])
mu_3 = torch.tensor([-1.0, 1.0])
var_3 = torch.tensor([0.5, 1.5])
eta_3 = torch.stack([w_3, mu_3, var_3])

print("big_M: ", big_M(x_3, w_3, mu_3, var_3))
"""


w_1 = torch.tensor([0.2, 0.8])
mu_1 = torch.tensor([0.0, 2.0])
var_1 = torch.tensor([0.5, 0.2])
eta_1 = torch.stack([w_1, mu_1, var_1])

w_2 = torch.tensor([0.1, 0.6, 0.3])
mu_2 = torch.tensor([0.0, 5.0, -10.0])
var_2 = torch.tensor([0.7, 0.1, 10.0])
eta_2 = torch.stack([w_2, mu_2, var_2])

w_3 = torch.tensor([0.2, 0.5, 0.3])
mu_3 = torch.tensor([0.0, 2.0, -1.0])
var_3 = torch.tensor([0.5, 0.2, 1.0])
eta_3 = torch.stack([w_3, mu_3, var_3])

print("P_1(eta_1): ", P_1(eta_1))
