import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp
from tqdm import tqdm

mu = torch.nn.Parameter(torch.tensor([0.0]))
sigma = torch.nn.Parameter(torch.tensor([2.0]))
w = torch.nn.Parameter(torch.tensor([1.0]))
nu = torch.stack([w, mu, sigma])
x = torch.nn.Parameter(torch.tensor([0.0]))

def gaussian_mixture(x, nu):
    w, mu, sigma = nu[0], nu[1], nu[2]
    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

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

print(gaussian_mixture(x, nu))