import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_hermite
from logger_config import logger
import math

# new functions for UncertaintyQuantificationGaussMix

def activation_input_approx(z_moments, m, s, mix_w):
    w_moments = MixToMoments(m, s, mix_w) 
    
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

def activation_output_approx(a_moments, f):
    m, s, w_mix = MomentsToMix(a_moments) # TODO check if its differentiable
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

def MixToMoments(m, s, mix_w):
    # m, s and mix_w are all tensors of the same size: mix_size
    mix_size = m.size(0)
    max_moment = 3 * mix_size

    moments = torch.zeros(max_moment)

    for i in range(mix_size):
        individual_moments = torch.ones(max_moment)
        individual_moments[1] = m[i]
        individual_moments[2] = m[i]**2 + s[i]

        for k in range(3, max_moment):      # k here corresponds to (k+1)-th moment
            individual_moments[k] = m[i] * individual_moments[k-1] + (k-1) * s[i] * individual_moments[k-2]
            
        moments += mix_w[i] * individual_moments

    return moments

def MomentsToMix(a_moments):
    # a_moments should be of shape (a_size, 3 * mix_size = max_moments)
    a_size = a_moments.size(0)
    mix_size = a_moments.size(1) / 3

    # TODO: but a differentiable way to calculate w_opt, m_opt, s_opt here

    # They all should be of shape (a_size, mix_size)
    return w_opt, m_opt, s_opt