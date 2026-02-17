import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger

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


    # TODO DELETE THIS
    #x = torch.linspace(-30, 30, 400)
    #x_expanded = x.view(-1, 1)
    #factor = (mix_product.mix1.eval_gamma_diff(x, gamma, noise) * mix_product.mix2.eval(x)).view(-1,1) * mix_start.eval_individual(x)
#
#    term1 = (w_start**(-1)).expand(x_expanded.shape[0], -1) * factor
#    term2 = ((x_expanded - m_start) / s_start ) * factor
#    term3 = (((x_expanded - m_start)**2 - (s_start)) / (s_start ** (3/2))) * factor
#        
#    plt.plot(x, term1, label="term1(x)")
#    plt.legend()
#    plt.xlabel("x")
#    plt.ylabel("f(x)")
#    plt.grid(True)
#    plt.show()
    # TODO DELETE THIS
    
    # stacked_new = torch.zeros(term1.size(0), term1.size(1) * 3)
    
    #for i in range(0, term1.size(0)):
    #    for j in range(0, term1.size(1)):
    #        stacked_new[i, 3*j    ] = term1[i,j]
    #        stacked_new[i, 3*j + 1] = term2[i,j]
    #        stacked_new[i, 3*j + 2] = term3[i,j]

    stacked = torch.cat([term1.unsqueeze(0), term2.unsqueeze(0), term3.unsqueeze(0)], dim=0)

    #print(stacked_new - stacked.permute(2,0,1).reshape(-1,x.size(0)).T)
        
    return stacked.permute(2,0,1).reshape(-1,x.size(0)).T


if __name__ == "__main__":

    w1 = torch.tensor([1.])
    m1 = torch.tensor([-1.])
    s1 = torch.tensor([1.])

    w2 = torch.tensor([0.1])
    m2 = torch.tensor([1.])
    s2 = torch.tensor([1.])

    nnodes = 2
    nsteps = 10
    gamma = 1.0
    noise = 0.01
    margin = 0.1

    logger.debug(f"Input Gauss mix f_1 is w={w1}, m={m1}, s={s1}.")
    logger.debug(f"Input Gauss mix f_2 is w={w2}, m={m2}, s={s2}.")

    mix_start = GaussMixClass.GaussMix(w2, m2, s2)
    mix_product = GaussMixClass.GaussMixProduct(w1, m1, s1, w2, m2, s2)

    b(nnodes, mix_start, mix_product, gamma, noise)