import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import Masterarbeit.Archive.GaussMixClass as GaussMixClass
from src.logger_config import logger

def ELBO(mix_1: GaussMixClass.GaussMix, mix_2: GaussMixClass.GaussMix) -> float:
    """
    Generate the ELBO between 2 Gaussian mixtures
    
    :param mix_1: Gaussian mixture
    :type mix_1: GaussMixClass.GaussMix
    :param mix_2: Gaussian mixture
    :type mix_2: GaussMixClass.GaussMix
    :return: ELBO
    :rtype: float
    """
    # E_q[log p(layer,Data)] − E_q[log q(layer,lambda)].
    # p(layer, Data) = p(layer) · p(Data | layer)
    # z_1^l: one mix --> one output
    return likelihood(mix_1, mix_1) - likelihood(mix_1, mix_2)

def likelihood(mix_1: GaussMixClass.GaussMix, mix_2: GaussMixClass.GaussMix) -> float: 
    """
    Calculates the likelihood of two Gaussian mixtures with the upper bound from Jensen's inequality
    
    :param mix_1: Gaussian mixture
    :type mix_1: GaussMixClass.GaussMix
    :param mix_2: Gaussian mixture
    :type mix_2: GaussMixClass.GaussMix
    :return: likelihood
    :rtype: float
    """
    w_1 = mix_1.w
    w_2 = mix_2.w
    s_1 = mix_1.s
    s_2 = mix_2.s
    m_1 = mix_1.m
    m_2 = mix_2.m

    size_1 = w_1.size(0)
    size_2 = w_2.size(0)

    z_ab = torch.zeros(size_1, size_2)
    for i in range(0, size_1):
        for j in range(0, size_2):
            z_ab[i,j] = 1/math.sqrt(2*math.pi*(s_1[i] + s_2[i])) * math.exp(-0.5 * (m_1[i] + m_2[j])**2/(s_1[i] + s_2[i]))
    inner = torch.log(torch.einsum("b,ab->a", w_2, z_ab))
    return torch.einsum("a,a->", w_1, inner)

if __name__ == "__main__":
    w1 = torch.tensor([0.3, 0.4, 0.3])
    m1 = torch.tensor([10., 0., -10.])
    s1 = torch.tensor([1., 1., 1.])
    w2 = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.2])
    m2 = torch.tensor([10., 0., -1., -2, -5])
    s2 = torch.tensor([3., 1., 1., 1., 1.])

    mix_1 = GaussMixClass.GaussMix(w1, m1, s1)
    mix_2 = GaussMixClass.GaussMix(w2, m2, s2)

    print(ELBO(mix_1, mix_2))