import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import Masterarbeit.Archive.GaussMixClass as GaussMixClass
from scipy.integrate import quad, quad_vec
from src.logger_config import logger
import os
os.environ["LOGURU_LEVEL"] = "WARNING"
from torchquad import Simpson

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

def b(n_nodes, mix_start, mix_product, gamma, noise):
    nodes, weights = gh_nodes_weights(n_nodes)
    """w_stack = torch.cat([mix_start.w, mix_product.mix1.w, mix_product.mix2.w])
    x_max = torch.max(w_stack)
    x_min = torch.max(- w_stack)

    s_stack = torch.cat([mix_start.s, mix_product.mix1.s, mix_product.mix2.s])
    var_max = torch.max(s_stack) ** 2

    x = torch.linspace(x_min + var_max, x_max + var_max, steps=100)

    sol = torch.trapezoid(b_integrand(x, mix_start, mix_product, gamma, noise), x, dim=0)

    print(sol - (b_integrand(nodes, mix_start, mix_product, gamma, noise) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)).sum(dim=0))"""
    print(b_integrand(nodes, mix_start, mix_product, gamma, noise))
    # n x m * n x 1 * n x 1
    return (b_integrand(nodes, mix_start, mix_product, gamma, noise) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)).sum(dim=0)

def b_integrand(x, mix_start, mix_product, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: the integrand b
    """
    w_start = mix_start.w
    m_start = mix_start.m
    s_start = mix_start.s ** 2
    x_expanded = x.view(-1, 1)
        
    factor = (mix_product.mix1.eval_gamma_diff(x, gamma, noise) * mix_product.mix2.eval(x)).view(-1,1) * mix_start.eval_individual(x)

    term1 = (w_start**(-1)).expand(x_expanded.shape[0], -1) * factor
    term2 = ((x_expanded - m_start) / s_start ) * factor
    term3 = (((x_expanded - m_start)**2 - (s_start)) / (s_start ** (3/2))) * factor

    stacked = torch.cat([term1.unsqueeze(0), term2.unsqueeze(0), term3.unsqueeze(0)], dim=0)
        
    return stacked.permute(2,0,1).reshape(-1,x.size(0)).T

def b_simple(mix_start, mix_product, gamma, noise):
    res, err = quad_vec(lambda x: b_integrand_simple(x, mix_start, mix_product, gamma, noise), -np.inf, np.inf)
    return res

def b_integrand_simple(x, mix_start, mix_product, gamma, noise):
    """
    Input:  @param x: float
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: the integrand b
    """
    w_start = mix_start.w
    m_start = mix_start.m
    s_start = mix_start.s ** 2
    x_torch = torch.tensor([x])
        
    factor = (mix_product.mix1.eval_gamma_diff(x_torch, gamma, noise) * mix_product.mix2.eval(x_torch)) * mix_start.eval_individual(x_torch).squeeze(0)


    term1 = (w_start**(-1)) * factor
    term2 = ((x_torch - m_start) / s_start ) * factor
    term3 = (((x_torch - m_start)**2 - (s_start)) / (s_start ** (3/2))) * factor

    stacked = torch.stack([term1, term2, term3], dim=1).flatten()

    return stacked

def b_torchquad(mix_start, mix_product, gamma, noise, N=1001):
    """GPU-accelerated adaptive Simpson integration"""
    simpson = Simpson()
    
    # Determine integration domain from Gaussian parameters
    all_means = torch.cat([mix_start.m, mix_product.mix1.m, mix_product.mix2.m])
    all_stds = torch.cat([mix_start.s, mix_product.mix1.s, mix_product.mix2.s])
    
    a = (all_means - 6 * all_stds).min().item()
    b = (all_means + 6 * all_stds).max().item()
    
    # Integration domain
    integration_domain = [[a, b]]
    
    def integrand_fn(x):
        # x shape: [N, 1] for torchquad
        return b_integrand(x.squeeze(), mix_start, mix_product, gamma, noise)
    
    result = simpson.integrate(
        fn=integrand_fn,
        dim=1,
        N=N,
        integration_domain=integration_domain
    )
    
    return result

if __name__ == "__main__":

    w1 = torch.tensor([1.])
    m1 = torch.tensor([1.])
    s1 = torch.tensor([1.])

    w2 = torch.tensor([1.])
    m2 = torch.tensor([0.])
    s2 = torch.tensor([1.])

    nnodes = 20
    nsteps = 10
    gamma = 1.0
    noise = 0.01
    margin = 0.1

    logger.debug(f"Input Gauss mix f_1 is w={w1}, m={m1}, s={s1}.")
    logger.debug(f"Input Gauss mix f_2 is w={w2}, m={m2}, s={s2}.")

    mix_start = GaussMixClass.GaussMix(w2, m2, s2)
    mix_product = GaussMixClass.GaussMixProduct(w1, m1, s1, w2, m2, s2)
    
    x_value = 0
    x = torch.tensor([x_value])

    print(b_integrand(x, mix_start, mix_product, gamma, noise))
    # print(b_integrand_simple(x_value, mix_start, mix_product, gamma, noise))
    print(b_torchquad(mix_start, mix_product, gamma, noise))