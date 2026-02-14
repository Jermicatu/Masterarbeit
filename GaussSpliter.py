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

def in_margin(n_nodes, mix_approx, mix_product, margin):
    """    
    Input:
        n_nodes: int, number of nodes used in Gauss Hermite integration
        eta_approx: Parameters for the approximated Gauss mix
        eta_1: Parameters for Gauss mix 1
        eta_2: Parameters for Gauss mix 2
        margin: float, states the acceptable error
    Output:
        check: boolean that is true is the error is within acceptable range and false if not
        # TODO sollte in margin nicht auch gamma enthalten?
    """

    nodes, weights = gh_nodes_weights(n_nodes)
    f_p = mix_product.mix1.eval(nodes) * mix_product.mix2.eval(nodes)
    int_p_1 = (f_p      * torch.exp(nodes**2) * weights).sum(dim=0)
    f_p = f_p / int_p_1
    f_approx = mix_approx.eval(nodes)

    dif = (f_p - f_approx)**2

    d =          (dif           * torch.exp(nodes**2) * weights).sum(dim=0)
    int_p =      ((f_p**2)      * torch.exp(nodes**2) * weights).sum(dim=0)
    int_approx = ((f_approx**2) * torch.exp(nodes**2) * weights).sum(dim=0)

    d = torch.sqrt(d / (int_p + int_approx))
    print(d)
    if d < margin:
       check = True
    else:
       check = False 

    return check

def add_component(n_nodes, mix_approx, mix_product):
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
    f_p = mix_product.mix1.eval(nodes) * mix_product.mix2.eval(nodes)
    f_approx = mix_approx.eval(nodes)

    dif = (f_p - f_approx) **2

    distances = (dif.view(-1, 1) * mix_approx.eval_individual(nodes) * torch.exp(nodes**2).view(-1, 1) * weights.view(-1, 1)).sum(dim=0)

    i = distances.argmax()

    mix_approx.w[i] = mix_approx.w[i]/2
    mix_approx.w = torch.cat((mix_approx.w, torch.tensor([mix_approx.w[i]])), dim=0)

    mix_approx.m[i] = mix_approx.m[i] + 0.4 * mix_approx.s[i]
    mix_approx.m = torch.cat((mix_approx.m, torch.tensor([mix_approx.m[i] - 2 * 0.4 * mix_approx.s[i]])), dim=0)
    
    mix_approx.s = torch.cat((mix_approx.s, torch.tensor([mix_approx.s[i]])), dim=0)

    return mix_approx #TODO check if return is needed