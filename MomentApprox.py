import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
import MomentsMakeMix
from logger_config import logger

import math

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

def activation_input_approx(z, W, mix_size: int):
    a_moments = activation_input_moments(z, W, mix_size)
    a = [MomentsMakeMix.MomentsToMix(a_j_moments) for a_j_moments in a_moments]
    return a

def activation_output_approx(a, f, mix_size: int):
    z_moments = activation_output_moments(a, f, mix_size)
    z = [MomentsMakeMix.MomentsToMix(z_j_moments[1:], mix_size) for z_j_moments in z_moments]
    return z

if __name__ == "__main__":    
    mix_size = 1

    a = [GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-1.]), torch.tensor([2.])),
    GaussMixClass.GaussMix(torch.tensor([1.]), torch.tensor([-2.]), torch.tensor([10.]))]

    mixes = activation_output_approx(a, ReLu, mix_size)

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


    # TODO
    # 3) integrate the functions into the FP