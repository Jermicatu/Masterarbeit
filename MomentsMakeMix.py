import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger

def gaussian_moments(mu, v, max_moment):
    """
    Raw non-central moments M_1 .. M_{max_moment} of N(mu, v).
    Uses the recurrence:  M_k = mu * M_{k-1} + (k-1) * v * M_{k-2}
    """
    m_list = []
    if max_moment >= 1:
        m_list.append(mu)
    if max_moment >= 2:
        m_list.append(mu**2 + v)
    for k in range(2, max_moment):      # k here corresponds to (k+1)-th moment
        m_list.append(mu * m_list[k-1] + k * v * m_list[k-2])
    
    return torch.stack(m_list)

def mixture_moments_torch(w, mu, v, n):
    # --- mixture moments ---
    max_moment = 3 * n - 1

    parts = [w[i] * gaussian_moments(mu[i], v[i], max_moment) for i in range(n)]

    return torch.stack(parts).sum(dim=0)

def pack_theta_torch(weights, means, variances):
    """
    Pack physical mixture parameters into the unconstrained theta vector.
    
    Parameters
    ----------
    weights : torch.Tensor, shape (n,)
    means : torch.Tensor, shape (n,)
    variances : torch.Tensor, shape (n,)

    Returns
    -------
    theta : torch.Tensor, shape (3*n - 1,)
    """
    w = weights.float()
    mu = means.float()
    v = variances.float()

    n = w.numel()
    assert mu.numel() == n and v.numel() == n
    assert torch.isclose(w.sum(), torch.tensor(1.0)), "Weights must sum to 1"
    assert (w > 0).all(), "Weights must be strictly positive"
    assert (v > 0).all(), "Variances must be strictly positive"

    # weights to logits: z_i = log(w_i) - log(w_n)
    z = torch.log(w[:-1]) - torch.log(w[-1])

    # variances to log-variances
    l = torch.log(v)

    # pack
    theta = torch.cat([z, mu, l])
    return theta

def unpack_theta_torch(theta, n):
    """
    Unpack theta into physical parameters.

    Returns
    -------
    w, mu, v : each torch.Tensor, shape (n,)
    """
    z = theta[:n-1]
    mu = theta[n-1:2*n-1]
    l = theta[2*n-1:3*n-1]

    # stable softmax
    z_full = torch.cat([z, torch.zeros(1, device=z.device)])
    z_full = z_full - z_full.max()
    w = torch.exp(z_full)
    w = w / w.sum()

    v = torch.exp(l)

    return w, mu, v

def MomentsToMix(M_given: torch.tensor, mixture_size: int) -> GaussMixClass.GaussMix:
    n = mixture_size
    best_loss = float('inf')
    best_theta = None

    # Multi-start: try many random initializations
    for trial in range(5):
        theta = torch.nn.Parameter(torch.randn(3*n - 1) * 2)  # wider spread
        optimizer = torch.optim.Adam([theta], lr=0.05)

        for step in range(2000):
            optimizer.zero_grad()
            w, m, s = unpack_theta_torch(theta, n)
            M_pred = mixture_moments_torch(w, m, s, n)
            loss = torch.sum((M_pred - M_given)**2)
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_theta = theta.detach().clone()

    # Unpack best solution
    w_opt, m_opt, s_opt = unpack_theta_torch(best_theta, n)
    return GaussMixClass.GaussMix(w_opt, m_opt, s_opt)



def test():
    # tests the accuracy of the MomentsToMix function

    torch.manual_seed(421)
        
    w_truth = torch.tensor([0.3, 0.2, 0.5])
    m_truth = torch.tensor([-1.0, 1.0, 3.0])
    s_truth = torch.tensor([1.0, 2.0, 3.0])
    
    n = 3

    M_given = mixture_moments_torch(w_truth, m_truth, s_truth, n)
    print("Target moments:", M_given)

    best_loss = float('inf')
    best_theta = None

    # Multi-start: try many random initializations
    for trial in range(10):
        theta = torch.nn.Parameter(torch.randn(3*n - 1) * 2)  # wider spread
        optimizer = torch.optim.Adam([theta], lr=0.05)

        for step in range(2000):
            optimizer.zero_grad()
            w, m, s = unpack_theta_torch(theta, n)
            M_pred = mixture_moments_torch(w, m, s, n)
            loss = torch.sum((M_pred - M_given)**2)
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_theta = theta.detach().clone()
            print(f"Trial {trial}: loss = {loss.item():.6f}")

    # Unpack best solution
    w_opt, m_opt, s_opt = unpack_theta_torch(best_theta, n)

    x = torch.linspace(-8, 8, 400)

    mix_truth = GaussMixClass.GaussMix(w_truth, m_truth, s_truth)
    mix_opt = GaussMixClass.GaussMix(w_opt, m_opt, s_opt)

    f_truth = mix_truth.eval(x)
    plt.plot(x, f_truth, label="mix_truth")

    f_opt = mix_opt.eval(x)
    plt.plot(x, f_opt, label="mix_opt")

    print("\n=== TRUTH ===")
    print("Weights: ", w_truth)
    print("Means:   ", m_truth)
    print("Variances:", s_truth)

    print("\n=== BEST FIT ===")
    print("Weights: ", w_opt.detach())
    print("Means:   ", m_opt.detach())
    print("Variances:", s_opt.detach())
    print(f"Final loss: {best_loss:.2e}")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

    """
    print("\n=== DIFFERENCES ===")
    print("Weights: ", w_truth - w_opt.detach())
    print("Means:   ", m_truth - m_opt.detach())
    print("Variances:", s_truth - s_opt.detach())
    """

if __name__ == "__main__":
    test()