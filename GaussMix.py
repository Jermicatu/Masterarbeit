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

def P_1(eta):
    """
    Input:  @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: matrix tensor representing P_1
    """
    w, mu, sigma = eta[0], eta[1], eta[2]
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

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
    P31 = w_i * torch.sqrt(sigma_i) * (md2 - s2_2) / (s2_2)
    P32 = w_i * w_j * torch.sqrt(sigma_i) * (-mu_diff) * (md2 - 3 * s2) / (s2_3)
    frac = md4 + 3.0 * s2 * (s2 - 2.0 * md2) / s2_4
    P33 = w_i * w_j * torch.sqrt(sigma_i) * torch.sqrt(sigma_j) * frac


    blocks = torch.stack([
        torch.stack([P11, P12, P13], dim=-1),
        torch.stack([P21, P22, P23], dim=-1),
        torch.stack([P31, P32, P33], dim=-1)
    ], dim=-2)

    blocks = blocks * prefactor[..., None, None]
    P_1 = blocks.permute(0, 2, 1, 3).reshape(3*N, 3*N)

    return P_1

def gaussian_mixture(x, eta):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian mix f defined via eta
    """
    w, mu, sigma = eta[0], eta[1], eta[2]
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

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
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian mix f defined via eta, gamma and noise
    """
    
    w, mu, var = eta[0], eta[1], eta[2]

    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    factor = ((1 + noise)/(gamma+noise))**2

    x = x[:, None]
    means = mu[None, :]
    variances = var[None, :]
    weights = w[None, :]

    coef = 1.0 # / torch.sqrt(2 * math.pi * variances)
    exponent = torch.exp(-0.5 * (x - means) ** 2 / (variances*factor))
    gaussians = coef * exponent  # shape (K, L)
    # Weighted sum over components
    return (weights * gaussians).sum(dim=1)

def gaussian_mix_gamma_diff(x, eta, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: f'(x) one dimensional vector of same shape as x where f'(x) is a gaussian mix f defined via eta, gamma and noise
    """
    
    w, mu, var = eta[0], eta[1], eta[2]

    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    factor = ((1 + noise)/(gamma+noise))**2

    x = x[:, None]
    means = mu[None, :]
    variances = var[None, :]
    weights = w[None, :]

    coef = (noise - gamma) * (x - means) / (1 + noise)**2 * variances
    exponent = torch.exp(-0.5 * (x - means) ** 2 / (variances*factor))
    gaussians = coef * exponent  # shape (K, L)
    # Weighted sum over components

    return (weights * gaussians).sum(dim=1)

def gaussian_tilde(x, eta_1, eta_2, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian tilde f defined via eta, gamma and noise
    """
    gamma_mix_2 = gaussian_mixture(x, eta_2)
    gamma_mix_1 = gaussian_mix_gamma(x, eta_1, gamma, noise)

    return gamma_mix_1 * gamma_mix_2

def gaussian_individual(x, eta):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: f(x) for each individual Gaussian eta_i
    """

    w, mu, sigma = eta[0], eta[1], eta[2]

    f_i = torch.zeros(x.size(0), w.size(0))
    for i in range(0, w.size(0)):
      eta_i = torch.stack([w[i], mu[i], sigma[i]]).view(-1,1)
      f_i[:, i] = gaussian_mixture(x, eta_i)
   
    return f_i

def big_M(x, eta):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: three dimensional matrix tensor representing matrix P_1(x) for multiple x
    """
    w, mu, sigma = eta[0], eta[1], eta[2]
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"
    
  
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

    f = gaussian_mixture(x, eta).reshape(-1, 1, 1, 1)
    
    blocks = blocks * f
    
    bigM = torch.zeros((K, L, L, 3, 3), dtype=blocks.dtype, device=blocks.device)
    diag_idx = torch.arange(L, device=blocks.device)
    bigM[:, diag_idx, diag_idx] = blocks
    bigM = bigM.permute(0, 1, 3, 2, 4).reshape(K, 3*L, 3*L)

    return bigM

def delta_P_integrand(x, eta_start, eta_1, eta_2, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: the integrand of delta P
    """
    
    return (gaussian_mixture(x, eta_start) - gaussian_tilde(x, eta_1, eta_2, gamma, noise)).view(-1,1,1) * big_M(x, eta_start)

def gh_nodes_weights(n_nodes: int):
    """
    Input:  @param n_nodes: number of nodes the integration uses
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: nodes, weights for the hermgauss for weight e^{-x^2}
    """
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_nodes)
    nodes = torch.as_tensor(nodes_np, dtype=torch.float32, device="cpu")   # shape (n,)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device="cpu") # shape (n,)
    return nodes, weights

def delta_P(n_nodes, eta_start, eta_1, eta_2, gamma, noise):
    """
    Input:  @param n_nodes: number of nodes the integration uses
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: delta P
    """
    nodes, weights = gh_nodes_weights(n_nodes)

    return (delta_P_integrand(nodes, eta_start, eta_1, eta_2, gamma, noise) * torch.exp(nodes**2).view(-1, 1, 1) * weights.view(-1, 1, 1)).sum(dim=0)

def P(n_nodes, eta_start, eta_1, eta_2, gamma, noise):
    """
    Input:  @param n_nodes: number of nodes the integration uses
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: matrix P
    """
    return P_1(eta_start) + delta_P(n_nodes, eta_start, eta_1, eta_2, gamma, noise)

def b_integrand(x, eta_start, eta_1, eta_2, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: the integrand b
    """
    w_start, mu_start, sigma_start = eta_start[0], eta_start[1], eta_start[2]
    x_expanded = x.view(-1, 1)

    f_i = torch.zeros(x.size(0), w_start.size(0))
    for i in range(0, w_start.size(0)):
      eta_i = torch.stack([w_start[i], mu_start[i], sigma_start[i]]).view(-1,1)
      f_i[:, i] = gaussian_mixture(x, eta_i)
    
    factor = (gaussian_mix_gamma_diff(x, eta_1, gamma, noise) * gaussian_mixture(x, eta_2)).view(-1,1) * gaussian_individual(x, eta_start)

    term1 = (w_start**(-1)).expand(x_expanded.shape[0], -1) * factor
    term2 = ((x_expanded - mu_start) / sigma_start ) * factor
    term3 = (((x_expanded - mu_start)**2 - (sigma_start)) / (sigma_start ** (3/2))) * factor

    stacked = torch.cat([term1.unsqueeze(0), term2.unsqueeze(0), term3.unsqueeze(0)], dim=0)
    return stacked.permute(2,0,1).reshape(-1,x.size(0)).T

def b(n_nodes, eta_start, eta_1, eta_2, gamma, noise):
    nodes, weights = gh_nodes_weights(n_nodes)
    
    # n x m * n x 1 * n x 1
    return (b_integrand(nodes, eta_start, eta_1, eta_2, gamma, noise) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)).sum(dim=0)

def solve_Pb(n_nodes, eta_start, eta_1, eta_2, gamma, noise):
    eta_shape = eta_start.size()
    b_vec = b(n_nodes, eta_start, eta_1, eta_2, gamma, noise)
    P_mat = P(n_nodes, eta_start, eta_1, eta_2, gamma, noise)

    return torch.linalg.solve(P_mat, b_vec.unsqueeze(-1)).squeeze(-1).view(eta_shape[::-1]).T

def in_margin(n_nodes, eta_approx, eta_1, eta_2, margin):
    """    
    Input:
        n_nodes: int, number of nodes used in Gauss Hermite integration
        eta_approx: Parameters for the approximated Gauss mix
        eta_1: Parameters for Gauss mix 2
        eta_2: Parameters for Gauss mix 2
        margin: float, states the acceptable error
    Output:
        check: boolean that is true is the error is within acceptable range and false if not
        # TODO test code
    """

    nodes, weights = gh_nodes_weights(n_nodes)
    f_p = gaussian_mixture(nodes, eta_1) * gaussian_mixture(nodes, eta_2)
    f_approx = gaussian_mixture(nodes, eta_approx)

    dif = (f_p - f_approx)**2

    d =          (dif             * torch.exp(nodes**2) * weights).sum(dim=0)
    int_p =      ((f_p ** 2)      * torch.exp(nodes**2) * weights).sum(dim=0)
    int_approx = ((f_approx ** 2) * torch.exp(nodes**2) * weights).sum(dim=0)

    d = d / (int_p + int_approx)

    if d < margin:
       check = True
    else:
       check = False 

    return check

def add_component(n_nodes, eta_approx, eta_1, eta_2):
    """    
    Args:
        n_nodes: int, number of nodes used in Gauss Hermite integration
        eta_approx: Parameters for the approximated Gauss mix
        eta_1: Parameters for Gauss mix 2
        eta_2: Parameters for Gauss mix 2
    Returns:
        new_eta: new eta with an extra component for higher accuracy
        # TODO test code
    """

    nodes, weights = gh_nodes_weights(n_nodes)
    f_p = gaussian_mixture(nodes,eta_1) * gaussian_mixture(nodes,eta_2)
    f_approx = gaussian_mixture(nodes, eta_approx)

    dif = (f_p - f_approx) **2

    distances = (dif * gaussian_individual(nodes, eta_approx) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)).sum(dim=0)

    i = distances.argmax()

    eta_append = eta_approx[:, i:i+1]

    eta_approx[0, i] = eta_approx[0, i]/2
    eta_append[0] = eta_approx[0, i]

    epsilon = 0.4 * eta_approx[2, i]

    eta_append[1] = eta_approx[1, i] + epsilon

    new_eta = torch.cat((eta_approx, eta_append), dim=1)
    
    new_eta[1, i] = eta_approx[1, i] - 2*epsilon

    return new_eta

def rk4_Pb(n_nodes, n_steps, eta_start, eta_1, eta_2, noise):
    """    
    Args:
        n_nodes:
        n_steps:
        eta_start:
        eta_1:
        eta_2:
        noise    
    Returns:
        gammas: torch tensor of shape (n_steps+1,)
        etas:   torch tensor of shape (n_steps+1, n)
    """
    h = 1 / n_steps
    
    gammas = torch.linspace(0, 1, n_steps+1)
    etas = [eta_start]
    eta = eta_start

    x = torch.linspace(-30, 30, 400)
    f = gaussian_mixture(x, eta)
    plt.plot(x, f)

    for i in range(n_steps):
        g = gammas[i]
        k1 = solve_Pb(n_nodes, eta, eta_1, eta_2, g, noise)

        k1[2, :] = torch.clamp(k1[2, :], min=0.01)
        k2 = solve_Pb(n_nodes, eta + h/2 * k1, eta_1, eta_2, g + h/2, noise)
        k2[2, :] = torch.clamp(k2[2, :], min=0.01)
        k3 = solve_Pb(n_nodes, eta + h/2 * k2, eta_1, eta_2, g + h/2, noise)
        k3[2, :] = torch.clamp(k3[2, :], min=0.01)
        k4 = solve_Pb(n_nodes, eta + h * k3,   eta_1, eta_2, g + h,   noise)
        k4[2, :] = torch.clamp(k4[2, :], min=0.01)

        eta = eta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

        #if in_margin(nnodes, etastart, eta1, eta2, margin) == False:
        #    add_component(nnodes, etastart, eta1, eta2)

        f = gaussian_mixture(x, eta)
        plt.plot(x, f)
        etas.append(eta)
    plt.grid(True)
    plt.show()

    # return gammas, torch.stack(etas)
    return eta

def plot_test(n_nodes, n_steps, eta_start, eta_1, eta_2, noise):
    new_eta = rk4_Pb(n_nodes, n_steps, eta_start, eta_1, eta_2, noise)
    # print(new_eta)
    x = torch.linspace(-30, 30, 400)
    f_0 = gaussian_mixture(x, new_eta)
    f_1 = gaussian_mixture(x, eta_1)
    f_2 = gaussian_mixture(x, eta_2)
    f_tild = gaussian_tilde(x, eta_1, eta_2, 1, noise)
    f_p = f_1*f_2


    plt.plot(x, f_0, label="approximation")
    plt.plot(x, f_1, label="f_1")
    plt.plot(x, f_2, label="f_2")
    plt.plot(x, f_tild, label="f_tilde")
    plt.plot(x, f_p, label="f_p")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    # plt.ylim(0, 0.3)
    plt.show()


w0 = torch.tensor([1.0])
mu0 = torch.tensor([0.0])
var0 = torch.tensor([1.0])
eta0 = torch.stack([w0, mu0, var0])

w1 = torch.tensor([0.4, 0.2, 0.4])
mu1 = torch.tensor([-3.0, 0.0, 3.0])
var1 = torch.tensor([1.0, 1.0, 1.0])
eta1 = torch.stack([w1, mu1, var1])

w2 = torch.tensor([0.25, 0.25, 0.25, 0.25])
mu2 = torch.tensor([-2.0, -1.0, 1.0, 2])
var2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
eta2 = torch.stack([w2, mu2, var2])

w3 = torch.tensor([0.15,  0.15, 0.15,  0.15, 0.15, 0.15,  0.15, 0.15])
mu3 = torch.tensor([6.0,  3.0, -3.0, -6.0, -6,  -3.0, 3.0, 6.0])
var3 = torch.tensor([1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0])
eta3 = torch.stack([w3, mu3, var3])

xvals = torch.tensor([1])
nnodes = 20
nsteps = 10
gamma_ = 1.0
noise_ = 0.01
margin_ = 0.1

etastart = eta2


print("===================================================================")
# plot_test(nnodes, nsteps, etastart, eta1, eta2, noise_)
# print("check_error: ", add_component(nnodes, etastart, eta1, eta2))
print("gaussian_individual: ", b_integrand(xvals, etastart, eta1, eta2, gamma_, noise_))


# TODO 
# check warum die weights sich so komisch verhalten
# 1) check ob P und b richtig sind                                gecheckt:           P_1 + delta_P_integrand + delta_P + P
# 2) check ob der solver richtigh läuft
# 3) check ob die Pb lösung richtig umgeformt wird

