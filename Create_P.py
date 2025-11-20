import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass

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

def big_M(x, mix_start):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: three dimensional matrix tensor representing matrix P_1(x) for multiple x
    """
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"
        
    
    K = x.shape[0]
    L = mix_start.m.shape[0]

    x_i = x[:, None]
    mu_i = mix_start.m[None, :]
    s2 = mix_start.s[None, :]
    w_i = mix_start.w[None, :]

    s = torch.sqrt(s2)
    s3 = s2*s
    s4 = s2**2
    s5 = s4*s
    s6 = s2**3

    d = x_i - mu_i
    d2 = d * d
    d3 = d * d2
    d4 = d2 * d2

    M11 = torch.zeros_like(d)
    M12 = d / (w_i * s2)
    M13 = (d2 - s2) / (w_i * s3)
    M21 = M12.clone().detach()
    M22 = (d2 - s2) / s4
    M23 = (d3 - 3.0 * s2 * d) / s5
    M31 = M13.clone().detach()
    M32 = M23.clone().detach()
    M33 = (d4 - 5.0 * s2 * d2 + 2.0 * s4) / s6

    blocks = torch.stack([
    torch.stack([M11, M12, M13], dim=-1),
    torch.stack([M21, M22, M23], dim=-1),
    torch.stack([M31, M32, M33], dim=-1),
    ], dim=-2)

    #f = mix_start.eval(x).reshape(-1, 1, 1, 1)

    f = mix_start.eval_individual(x)
        
    #blocks = blocks * f
    
    blocks = blocks * f[:, :, None, None]

    M = torch.zeros((K, L, L, 3, 3), dtype=blocks.dtype, device=blocks.device)
    diag_idx = torch.arange(L, device=blocks.device)
    M[:, diag_idx, diag_idx] = blocks
    M = M.permute(0, 1, 3, 2, 4).reshape(K, 3*L, 3*L)
    
    for i in range(0, M.size(0)):
        if not torch.allclose(M[i].transpose(0, 1), M[i]):
            raise ValueError("M[i] Matrix is not symmetric.")

    return M

def delta_P_integrand(x, mix_start, mix_product, gamma, noise):
    """
    Input:  @param x: one dimensional tensor stating the x-values for which we want an output
            @param eta_1: two dimensional tensor of form [w_1, mu_1, sigma_1]
            @param eta_2: two dimensional tensor of form [w_2, mu_2, sigma_2]
            @param eta_3: two dimensional tensor of form [w_3, mu_3, sigma_3]
            @param gamma: float in range [0,1]
            @param noise: float 
    Output: the integrand of delta P
    """

    return (mix_start.eval(x) - mix_product.eval_tilde(x, gamma, noise)).view(-1,1,1) * big_M(x, mix_start)

def delta_P(n_nodes, mix_start, mix_product, gamma, noise):
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

    return (delta_P_integrand(nodes, mix_start, mix_product, gamma, noise) * torch.exp(nodes**2).view(-1, 1, 1) * weights.view(-1, 1, 1)).sum(dim=0)

def P_1(mix):
    """
    Input:  @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: matrix tensor representing P_1
    """
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    L = mix.w.size(0)
    mu_i = mix.m[:, None]
    mu_j = mix.m[None, :]
    sigma_i = mix.s[:, None]
    sigma_j = mix.s[None, :]
    w_i = mix.w[:, None]
    w_j = mix.w[None, :]

    # remember all sigmas are given in squared for (sigma**2)
    s2 = sigma_i + sigma_j
    s2_2 = s2**2
    s2_3 = s2**3
    s2_4 = s2**4
        
    mu_diff = mu_i - mu_j
    md2 = mu_diff**2
    md4 = mu_diff**4

    prefactor = (1.0 / torch.sqrt(2 * math.pi * s2)) * torch.exp(-0.5 * (md2) / s2)

    P11 = torch.ones_like(mu_diff)
    P12 = w_j * mu_diff / s2
    P13 = w_j * torch.sqrt(sigma_j) * (md2 - s2_2) / (s2_2)
    P21 = w_i * (-mu_diff) / s2
    P22 = w_i * w_j * (s2 - md2) / (s2_2)
    P23 = w_i * w_j * torch.sqrt(sigma_j) * (mu_diff) * (md2 - 3 * s2) / (s2_3)
    P31 = w_i * torch.sqrt(sigma_i) * (md2 - s2_2) / (s2_2)
    P32 = w_i * w_j * torch.sqrt(sigma_i) * (-mu_diff) * (md2 - 3 * s2) / (s2_3)
    frac = (md4 + 3.0 * s2 * (s2 - 2.0 * md2)) / s2_4
    P33 = w_i * w_j * torch.sqrt(sigma_i) * torch.sqrt(sigma_j) * frac

    P_1 = torch.zeros(3 * L, 3 * L)

    for i in range(0, L):
        for j in range(0, L):
            P_1[3*i    , 3*j    ] = prefactor[i, j] * P11[i, j]
            P_1[3*i    , 3*j + 1] = prefactor[i, j] * P12[i, j]
            P_1[3*i    , 3*j + 2] = prefactor[i, j] * P13[i, j]

            P_1[3*i + 1, 3*j    ] = prefactor[i, j] * P21[i, j]
            P_1[3*i + 1, 3*j + 1] = prefactor[i, j] * P22[i, j]
            P_1[3*i + 1, 3*j + 2] = prefactor[i, j] * P23[i, j]
            
            P_1[3*i + 2, 3*j    ] = prefactor[i, j] * P31[i, j]
            P_1[3*i + 2, 3*j + 1] = prefactor[i, j] * P32[i, j]
            P_1[3*i + 2, 3*j + 2] = prefactor[i, j] * P33[i, j]


    #print(torch.allclose(prefactor.transpose(0, 1), prefactor))
    #print(torch.allclose(P11.transpose(0, 1), P11))
    #print(torch.allclose(P12.transpose(0, 1), P21))
    #print(torch.allclose(P13.transpose(0, 1), P31))
    #print(torch.allclose(P22.transpose(0, 1), P22))
    #print(torch.allclose(P23.transpose(0, 1), P32))
    #print(torch.allclose(P31.transpose(0, 1), P13))
    #print(torch.allclose(P33.transpose(0, 1), P33))

    if not (P_1.size(0) == 3*L and P_1.size(1) == 3*L):
        raise ValueError("P_1 Matrix is not of size (3*L, 3*L).")
    
    if not torch.allclose(P_1.transpose(0, 1), P_1):
        raise ValueError("P_1 Matrix is not symmetric.")
    
    #eigs = torch.linalg.eigvalsh(P_1)
    #if (eigs < -1e-1).any():
    #    raise ValueError("P_1 is not PSD!")
    
    return P_1

def P_1_old(mix):
    """
    Input:  @param eta: two dimensional tensor of form [w, mu, sigma]
    Output: matrix tensor representing P_1
    """
    # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

    N = mix.m.shape[0]
    mu_i = mix.m[:, None]
    mu_j = mix.m[None, :]
    sigma_i = mix.s[:, None]
    sigma_j = mix.s[None, :]
    w_i = mix.w[:, None]
    w_j = mix.w[None, :]

    # remember all sigmas are given in squared for (sigma**2)
    s2 = sigma_i + sigma_j
    s2_2 = s2**2
    s2_3 = s2**3
    s2_4 = s2**4
    
    mu_diff = mu_i - mu_j
    md2 = mu_diff**2
    md4 = mu_diff**4

    prefactor = (1.0 / torch.sqrt(2 * math.pi * s2)) * torch.exp(-0.5 * (md2) / s2)

    P11 = torch.ones_like(mu_diff)
    P12 = w_j * mu_diff / s2
    P13 = w_j * torch.sqrt(sigma_j) * (md2 - s2_2) / (s2_2)
    P21 = w_i * (-mu_diff) / s2
    P22 = w_i * w_j * (s2 - md2) / (s2_2)
    P23 = w_i * w_j * torch.sqrt(sigma_j) * (mu_diff) * (md2 - 3 * s2) / (s2_3)
    P31 = w_i * torch.sqrt(sigma_i) * (md2 - s2_2) / (s2_2)
    P32 = w_i * w_j * torch.sqrt(sigma_i) * (-mu_diff) * (md2 - 3 * s2) / (s2_3)
    frac = (md4 + 3.0 * s2 * (s2 - 2.0 * md2)) / s2_4
    P33 = w_i * w_j * torch.sqrt(sigma_i) * torch.sqrt(sigma_j) * frac


    blocks = torch.stack([
        torch.stack([P11, P12, P13], dim=-1),
        torch.stack([P21, P22, P23], dim=-1),
        torch.stack([P31, P32, P33], dim=-1)
    ], dim=-2)
    
    blocks = blocks * prefactor[..., None, None]
    
    print(blocks[:,:,1,1])
    P_1 = blocks.permute(0, 2, 1, 3).reshape(3*N, 3*N)
    print(torch.all(P_1.transpose(0, 1) == P_1))
    print(P_1[3,3])

    return P_1

def P(n_nodes, mix_start, mix_product, gamma, noise):
    return P_1(mix_start) + delta_P(n_nodes, mix_start, mix_product, gamma, noise)