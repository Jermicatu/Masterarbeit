import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_hermite
import src.GaussMixClass as GaussMixClass
import MomentsMakeMix
from src.logger_config import logger
import math

# new functions for UncertaintyQuantificationGaussMix

def activation_input_approx_moment_to_moment(z_moments, m, s, mix_w):
    w_moments = MomentsMakeMix.MixToMoments(m, s, mix_w) 
    
    # at this point z_l_moments size should be: z_size, max_moment
    # at this point w_l_moments size should be: a_size, z_size + 1, max_moment
    # the max_moment includes the moment 0 meaning that max_moment = 3 * mix_size

    z_size = z_moments.size(0)
    a_size = w_moments.size(0)
    max_moment = z_moments.size(-1)

    a_moments = torch.zeros(a_size, max_moment)
    
    for j in range(a_size):
        # Start with bias moments (index 0), make a copy
        cumulative = w_moments[j,0,:].clone()
        # Add each z_i * w_{i,j} term (note: W[j][0] corresponds to z[0])
        for i in range(z_size):
            # Element-wise product of moments: m_k(z_i * w_{i+1,j}) = m_k(z_i) * m_k(w_{i,j})
            product_moments = z_moments[i,:] * w_moments[j,i+1,:]
                    
            # Convolve: m_k(S + X) = sum_{split=0}^k C(k,split) * m_split(S) * m_{k-split}(X)
            new_cumulative = torch.zeros(max_moment)
            new_cumulative[0] = 1
            for k in range(1, max_moment):
                for split in range(k + 1):
                    new_cumulative[k] += math.comb(k, split) * cumulative[split] * product_moments[k - split]
                    
            cumulative = new_cumulative.clone()

        a_moments[j,:] = cumulative

    return a_moments

def activation_output_approx_moment_to_moment(a_moments, f):
    m, s, w_mix = MomentsMakeMix.MomentsToMix(a_moments) # TODO check if its differentiable
    z_size = m.size(0)
    max_moments = a_moments.size(0)

    z_moments = torch.ones(z_size, max_moments)

    f_plus = f(m + torch.sqrt(s + 1e-8))
    f_minus = f(m - torch.sqrt(s + 1e-8))

    # This part can be vectorized more but for readability I leave it like that for now
    for j in range(z_size):
        for k in range(1, max_moments):
            z_moments[j, k] = torch.sum(w_mix[j,:] * (f_plus[j,:]**k + f_minus[j,:]**k)) / 2

    return z_moments

# functions needed for the calculations

def ReLu(x):
    return torch.nn.functional.relu(x)

def activation_input_moments(z, W, mix_size: int):
    """Calculate moments of activation input a = w_{0,j} + sum_i z_i * w_{i,j}
    
    Args:
        z: list of GaussMix, previous layer outputs (length z_size)
        W: list of lists of GaussMix, weights W[j][i] for output j, input i
           W[j][0] is bias, W[j][1:] are weights for z_0, z_1, ...
        mix_size: number of components in each mixture
    
    Returns:
        list of moment tensors, one per output unit j
    """
    max_moment = 3 * mix_size - 1
    z_size = len(z)
    a_size = len(W)
    
    # Pre-compute all mixture moments
    z_moments = [z_i.mixture_moments(mix_size) for z_i in z]
    W_moments = [[w_ij.mixture_moments(mix_size) for w_ij in vector] for vector in W]
    
    result = []
    
    for j in range(a_size):
        # Start with bias moments (index -1), make a copy
        cumulative = W_moments[j][-1]
        
        # Add each z_i * w_{i,j} term (note: W[j][0] corresponds to z[0])
        for i in range(z_size):
            # Element-wise product of moments: m_k(z_i * w_{i+1,j}) = m_k(z_i) * m_k(w_{i,j})
            product_moments = z_moments[i] * W_moments[j][i]
            
            # Convolve: m_k(S + X) = sum_{split=0}^k C(k,split) * m_split(S) * m_{k-split}(X)
            new_cumulative = []
            for k in range(max_moment+1):
                total = 0.0
                for split in range(k + 1):
                    total += math.comb(k, split) * cumulative[split] * product_moments[k - split]
                new_cumulative.append(total)
            
            cumulative = new_cumulative
        
        result.append(torch.stack(cumulative))
    
    return result

def activation_output_moments_gh(a, f, mix_size: int, n_points: int = 100):
    """Calculate the moments of the activation output f(a) using Gauss-Hermite quadrature

    Args:
        a: list of GaussMix, layer inputs (length a_size)
        f: activation function (must handle float inputs)
        mix_size: number of components in each mixture
        n_points: number of Gauss-Hermite quadrature points (default 10)

    Returns:
        list of moment tensors, one per output unit j
    """
    max_moment = 3 * mix_size - 1
    a_size = len(a)
    
    # Pre-compute Gauss-Hermite nodes and weights
    # roots_hermite returns nodes x_j and weights w_j for:
    #   ∫_{-∞}^{∞} g(x) exp(-x²) dx ≈ Σ_j w_j g(x_j)
    gh_nodes, gh_weights = roots_hermite(n_points)
    gh_nodes = torch.tensor(gh_nodes)
    gh_weights = torch.tensor(gh_weights)
    
    result = []
    
    for j in range(a_size):
        z_j_moments = torch.ones(max_moment + 1)
        w = a[j].w
        m = a[j].m
        s = a[j].s
        
        for k in range(1, max_moment + 1):
            total = 0.0
            
            for i in range(mix_size):
                mu = m[i].item()
                sigma = math.sqrt(s[i].item())
                weight = w[i].item()
                
                # Transform nodes: x = √2·σ·node + μ
                # This maps the standard GH nodes to N(μ, σ²)
                transformed_nodes = math.sqrt(2) * sigma * gh_nodes + mu
                
                # Evaluate activation at all nodes
                fx = f(transformed_nodes)
                
                # Gauss-Hermite quadrature with probability normalization
                # Divide by √π because the Gaussian density has 1/√(2πσ²) 
                # and the substitution introduces √(2)σ factors that cancel
                moment_i = torch.sum(gh_weights * (fx ** k)) / math.sqrt(math.pi)
                
                total += weight * moment_i
            
            z_j_moments[k] = total
        
        result.append(z_j_moments.clone())
    
    return result

def activation_output_moments(a, f, mix_size: int):
    """Calculate the moments of the activation output f(a)

    Args:
        a (_type_): list of GaussMix, layer inputs (length a_size)
        f (_type_): activation function
        mix_size (int): number of components in each mixture

    Returns:
        list of moment tensors, one per output unit j
    """
    max_moment = 3 * mix_size - 1
    a_size = len(a)

    result = []
    
    # For each z_j we want to get the moments for
    for j in range(a_size):
        z_j_moments = torch.ones(max_moment+1)
        w = a[j].w
        m = a[j].m
        s = a[j].s
        
        f_plus = torch.zeros(mix_size)
        f_minus = torch.zeros(mix_size)
        # create function for each Gaussian
        for i in range(mix_size):
            f_plus[i] = f(m[i] + math.sqrt(s[i]))
            f_minus[i] = f(m[i] - math.sqrt(s[i]))

        # For each moment we want to calculate
        for k in range(1, max_moment+1):
            z_j_moments[k] = torch.sum(w * (f_plus**k + f_minus**k)) / 2
        
        result.append(z_j_moments.clone())

    return result

# we want to export these 2 functions

def activation_input_approx(z, W, mix_size: int):
    a_moments = activation_input_moments(z, W, mix_size)
    a = [MomentsMakeMix.MomentsToMix(a_j_moments) for a_j_moments in a_moments]
    return a

def activation_output_approx(a, f, mix_size: int):
    z_moments = activation_output_moments_gh(a, f, mix_size)
    z = [MomentsMakeMix.MomentsToMix(z_j_moments[1:], mix_size) for z_j_moments in z_moments]
    return z

# end of export rest is for testing

def sample_activation_input_mixture(z, W, n_samples=100000):

    z_size = len(z)
    a_size = len(W)

    samples = []

    for j in range(a_size):
        samples_j = torch.zeros(n_samples)

        for k in range(n_samples):
            for i in range(z_size):
                z_i = z[i]
                w_ij = W[j][i]

                w_z_i = z_i.w
                m_z_i = z_i.m
                s_z_i = z_i.s

                w_w_ij = w_ij.w
                m_w_ij = w_ij.m
                s_w_ij = w_ij.s

                z_i_sample = w_z_i * (torch.randn(w_z_i.size(0)) * torch.sqrt(s_z_i) + m_z_i)
                W_ij_sample = w_w_ij * (torch.randn(w_w_ij.size(0)) * torch.sqrt(s_w_ij) + m_w_ij)

                samples_j[k] += z_i_sample.item() * W_ij_sample.item()
            w_ij = W[-1][j]

            w_w_ij = w_ij.w
            m_w_ij = w_ij.m
            s_w_ij = w_ij.s

            W_ij_sample = w_w_ij* (torch.randn(w_w_ij.size(0)) * torch.sqrt(s_w_ij) + m_w_ij)

            samples_j[k] += W_ij_sample.item()

        samples.append(samples_j.clone())

    return samples

def sample_activation_output_mixture(a, f, n_samples=100000):
    """Sample from ReLU(a) where a is a list of Gaussian mixtures"""
    samples = []
    
    for aj in a:
        w = aj.w
        m = aj.m
        s = aj.s
        
        samples_j = torch.zeros(n_samples)

        for i in range(n_samples):
            samples_j[i] = f(torch.sum(w * (m + torch.randn(w.size(0)) * torch.sqrt(s))))

        samples.append(samples_j.clone())

    return samples

def activation_output_moments_from_samples(a, f, mix_size, n_samples=100000):

    samples = sample_activation_output_mixture(a, f, n_samples=100000)

    moments = []
    max_moment = 3 * mix_size - 1

    for samples_j in samples:
        moments_j = torch.ones(max_moment + 1)
        for k in range(1, max_moment+1):
            moments_j[k] = torch.sum(samples_j**k)/n_samples
        moments.append(moments_j.clone())

    return moments

def activation_input_moments_from_samples(a, W, mix_size, n_samples=100000):

    samples = sample_activation_input_mixture(a, W, n_samples=100000)

    moments = []
    max_moment = 3 * mix_size - 1

    for samples_j in samples:
        moments_j = torch.ones(max_moment + 1)
        for k in range(1, max_moment+1):
            moments_j[k] = torch.sum(samples_j**k)/n_samples
        moments.append(moments_j.clone())

    return moments

def test1():
    # Tests the activation output moments
    mix_size = 1

    a = [GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-1.]), torch.tensor([2.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-2.]), torch.tensor([10.]))]

    my_moments = activation_output_moments_gh(a, ReLu, mix_size)

    empirical_moments = activation_output_moments_from_samples(a, ReLu, mix_size)

    print(my_moments)
    print(empirical_moments)

def test2():
    # Tests the activation output moments
    mix_size = 2

    a = [GaussMixClass.GaussMix(torch.tensor([0.5, 0.5]), torch.tensor([1., 2.]), torch.tensor([1., 10.])),
    GaussMixClass.GaussMix(torch.tensor([0.1, 0.9]), torch.tensor([-1., -2]), torch.tensor([2., 2.])),
    GaussMixClass.GaussMix(torch.tensor([0.3, 0.7]), torch.tensor([-2., -10.]), torch.tensor([10., 0.5]))]

    my_moments = activation_output_moments_gh(a, ReLu, mix_size)

    empirical_moments = activation_output_moments_from_samples(a, ReLu, mix_size)

    print(my_moments)
    print(empirical_moments)

def test3():
    # Tests the activation input moments
    mix_size = 1

    # a size is 3
    a = [GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-1.]), torch.tensor([2.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-2.]), torch.tensor([10.]))]

    # W size is 1x(3+bias)
    W = [[GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-1.]), torch.tensor([2.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-2.]), torch.tensor([10.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-2.]), torch.tensor([10.]))]]

    my_moments = activation_input_moments(a, W, mix_size)

    empirical_moments = activation_input_moments_from_samples(a, W, mix_size)

    print(my_moments)
    print(empirical_moments)

if __name__ == "__main__":    
    test2()

"""
    print("0:")
    print(mixes[0].w)
    print(mixes[0].m)
    print(mixes[0].s)
    print("1:")
    print(mixes[1].w)
    print(mixes[1].m)
    print(mixes[1].s)
    print("2:")
    print(mixes[2].w)
    print(mixes[2].m)
    print(mixes[2].s)
"""

    # TODO
    # 3) integrate the functions into the FP