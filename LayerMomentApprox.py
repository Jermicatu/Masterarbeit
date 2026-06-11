import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_hermite
from logger_config import logger
import math

def activation_input_approx_vectorized(z_moments, m, s, mix_w):
    w_moments = MixToMoments(m, s, mix_w)
            
    # at this point z_moments size should be: z_size, max_moment
    # at this point w_moments size should be: a_size, z_size + 1, max_moment
    # the max_moment includes the moment 0 meaning that max_moment = 3 * mix_size

    z_size = z_moments.size(0)
    a_size = w_moments.size(0)
    max_moment = z_moments.size(-1)

    product_moments = z_moments * w_moments[:,1:]
    old_cumulative = w_moments[:, 0].clone()
    for i in range(z_size):
        new_cumulative = torch.zeros(a_size, max_moment)
        new_cumulative[:, 0] = 1.0
        for k in range(1, max_moment):
            for split in range(k + 1):
                new_cumulative[:, k] += math.comb(k, split) * old_cumulative[:, split] * product_moments[:, i, k - split]
        old_cumulative = new_cumulative.clone()

    return new_cumulative

# new functions for UncertaintyQuantificationGaussMix

def activation_input_approx(z_moments, m, s, mix_w):
    # should work (aka. tested it)
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
        cumulative = w_moments[j,0].clone()
        # Add each z_i * w_{i,j} term (note: W[j][0] corresponds to z[0])
        for i in range(z_size):
            # Element-wise product of moments: m_k(z_i * w_{i+1,j}) = m_k(z_i) * m_k(w_{i,j})
            product_moments = z_moments[i] * w_moments[j,i+1]
                        
            # Convolve: m_k(S + X) = sum_{split=0}^k C(k,split) * m_split(S) * m_{k-split}(X)
            new_cumulative = torch.zeros(max_moment)
            new_cumulative[0] = 1
            for k in range(1, max_moment):
                for split in range(k + 1):
                    new_cumulative[k] += math.comb(k, split) * cumulative[split] * product_moments[k - split]
                        
            cumulative = new_cumulative.clone()

        a_moments[j,:] = cumulative

    return a_moments

def activation_output_approx_vectorized(a_moments, f):
    m = a_moments[:, 1]
    s = torch.clamp(a_moments[:, 2] - m**2, min=1e-6)
    z_size = a_moments.size(0)
    max_moments = a_moments.size(1)

    z_moments = torch.ones(z_size, max_moments)

    f_plus = f(m + torch.sqrt(s + 1e-8))
    f_minus = f(m - torch.sqrt(s + 1e-8))

    k = torch.arange(1, max_moments, dtype=f_plus.dtype, device=f_plus.device)
    z_moments[:, 1:] = (f_plus.unsqueeze(1) ** k.unsqueeze(0) + f_minus.unsqueeze(1) ** k.unsqueeze(0)) / 2

    return z_moments

def activation_output_approx(a_moments, f):
    m = a_moments[:, 1]
    s = torch.clamp(a_moments[:, 2] - m**2, min=1e-6)
    z_size = a_moments.size(0)
    max_moments = a_moments.size(1)

    z_moments = torch.ones(z_size, max_moments)

    f_plus = f(m + torch.sqrt(s + 1e-8))
    f_minus = f(m - torch.sqrt(s + 1e-8))

    # This part can be vectorized more but for readability I leave it like that for now
    for j in range(z_size):
        for k in range(1, max_moments):
            z_moments[j, k] = (f_plus[j]**k + f_minus[j]**k) / 2

    return z_moments

def activation_output_approx_mix(a_moments, f):
    print("HOLD UP, WAIT A MINUTE, SOMETHING AIN RIGHT")
    m, s, w_mix = MomentsToMix(a_moments) # TODO check if its differentiable
    z_size = m.size(0)
    max_moments = a_moments.size(1)

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
    # should work (aka. tested it)
    # m, s and mix_w are all tensors of the same size: dim1, dim2 , mix_size OR dim1 mix_size
    batch_shape = m.shape[:-1]
    mix_size = m.size(-1)
    max_moment = 3 * mix_size

    moments = torch.ones(batch_shape + (1,))

    m1 = (mix_w * m).sum(dim=-1, keepdim=True)      # [..., 1]
    moments = torch.cat([moments, m1], dim=-1)      # [..., 2]
    
    mu_2 = m**2 + s
    m2 = (mix_w * mu_2).sum(dim=-1, keepdim=True)   # [..., 1]
    moments = torch.cat([moments, m2], dim=-1)      # [..., 3]
    
    # Recurrence state
    mu_km2, mu_km1 = m, mu_2
    
    for k in range(3, max_moment):
        mu_next = m * mu_km1 + (k-1) * s * mu_km2     # [..., 1]
        mk = (mix_w * mu_next).sum(dim=-1, keepdim=True)  # [..., 1]
        moments = torch.cat([moments, mk], dim=-1)  # [..., k+1]
        mu_km2, mu_km1 = mu_km1, mu_next
        
    return moments

def unpack_theta_torch(theta, n):
    """
    Unpack theta into physical parameters.

    Returns
    -------
    mu, v, w : each torch.Tensor, shape (dim1, n)
    """
    z = theta[:,:n-1]
    mu = theta[:,n-1:2*n-1]
    l = theta[:,2*n-1:3*n-1]

    # stable softmax
    z_full = torch.cat([z, torch.zeros(z.size(0), 1, device=z.device)], dim=1)
    z_full = z_full - z_full.max(dim=1, keepdim=True)[0]
    w = torch.exp(z_full)
    w = w / w.sum(dim=-1, keepdim=True)

    v = torch.exp(l)

    return mu, v, w

def MomentsToMix(a_moments):
    # a_moments should be of shape (a_size, 3 * mix_size = max_moments)
    print("HOLD UP, WAIT A MINUTE, SOMETHING AIN RIGHT")
    a_moments = a_moments.detach()
    a_size = a_moments.size(0)
    mix_size = int(a_moments.size(1) / 3)

    theta = torch.nn.Parameter(torch.randn(a_size, 3*mix_size - 1) * 2)  # wider spread
    optimizer = torch.optim.Adam([theta], lr=0.05)

    for step in range(2000):
        optimizer.zero_grad()
        m, s, w = unpack_theta_torch(theta, mix_size)
        a_pred = MixToMoments(m, s, w)
        loss = torch.sum((a_pred - a_moments)**2)
        loss.backward()
        optimizer.step()

    # Unpack best solution
    m_opt, s_opt, w_opt = unpack_theta_torch(theta, mix_size)

    # They all should be of shape (a_size, mix_size)
    return m_opt, s_opt, w_opt

def test0():
    # test MomentsToMix

    a_moments = torch.tensor([
        [  1.0000,   0.8000,   4.5000,   4.6000,  20.0000,  33.4000],
        [  1.0000,   1.4000,   6.3800,  17.1000,  51.7200, 169.8560]
    ])

    m_opt, s_opt, w_opt = MomentsToMix(a_moments)

    print(m_opt)
    print(s_opt)
    print(w_opt)

def test1():
    # test activation_input_approx

    # These are random moments. I dont know what the Gaussian mix would look like
    # z has 2 elements that are Gaussian mixtures of size 2 aka. 6 moments with the first beeing the 0-th moment
    z_moments = torch.tensor([
        [  1.0000,   0.8000,   4.5000,   4.6000,  20.0000,  33.4000],
        [  1.0000,   1.4000,   6.3800,  17.1000,  51.7200, 169.8560]
    ])
    
    # Parameters that describe the Gaussian mixture of the 3 weights (1st is bias) that we multiply with z (except bias of course)
    m = torch.tensor([[[-1, 1], 
                       [-1, 1],
                       [-3, 3]]])
    s = torch.ones(1,3,2)
    mix_w = torch.ones(1,3,2) - 0.5

    # test is here
    a_moments = activation_input_approx(z_moments, m, s, mix_w)
    print(a_moments)

def test2():
    #test activation_output_approx
    a_moments = torch.tensor([
        [  1.0000,   0.8000,   4.5000,   4.6000,  20.0000,  33.4000],
        [  1.0000,   1.4000,   6.3800,  17.1000,  51.7200, 169.8560]
    ])

    def relu(x):
        return torch.relu(x)

    z_moments = activation_output_approx(a_moments, relu)
    print("MOMENTS ARE:")
    print(z_moments)

if __name__ == "__main__":
    print("INITIATE TESTS:")
    # test2()

    a = torch.rand(2,3,4)
    b = torch.rand(3,4)
    print(a)
    print(a[:,1:])