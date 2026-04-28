import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger

def activation_input_moments(z, W, mix_size: int):
    """ This function uses the Gaussian mixtures from z and W to calculate the moments of the activation input a

    Args:
        z (_type_): array of GaussMixClass.GaussMix elements, describes the previous layer output
        W (_type_): double array of GaussMixClass.GaussMix elements, describes the layer weights
        mix_size (int): size of all mixes in the network

    Returns:
        _type_: _description_
    """
    z_moments = [z_i.mixture_moments(mix_size) for z_i in z]
    W_moments = [[w_ij.mixture_moments(mix_size) for w_ij in vector] for vector in W]

    a_size = len(W)
    z_size = len(z_moments)

    current_moments = [W_moments[j][z_size] for j in range(0, a_size)]

    for j in range(0, a_size):
        for i in range(0, z_size):
            old_moments = current_moments
            new_moments = z_moments[i] * W_moments[j][i]
            for k in range(0, a_size):
                current_moments[j][k] += math.factorial(a_size) / (math.factorial(k) * math.factorial(a_size - k)) *old_moments[k] * new_moments[a_size-k] # TODO do forumla
                # write for all dimensions what they do and mean in relation to the formula

    return current_moments

if __name__ == "__main__":
    mix_size = 3

    z = [GaussMixClass.GaussMix(torch.tensor([0.3, 0.4, 0.3]), torch.tensor([1., 2., 2.]), torch.tensor([1., 1., 0.5,])),
    GaussMixClass.GaussMix(torch.tensor([0.7, 0.2, 0.1]), torch.tensor([-1., -1., 0.]), torch.tensor([2., 2., 2.,])),
    GaussMixClass.GaussMix(torch.tensor([0.5, 0.05, 0.45]), torch.tensor([-2., -2., -2.]), torch.tensor([10., 0.5, 0.5,]))]

    W = [[GaussMixClass.GaussMix(torch.tensor([0.1, 0.1, 0.8]), torch.tensor([0., 1., -1.]), torch.tensor([1., 1., 1.,])),
    GaussMixClass.GaussMix(torch.tensor([0.2, 0.2, 0.6]), torch.tensor([-1., -1., -1.]), torch.tensor([1., 1., 1.,])),
    GaussMixClass.GaussMix(torch.tensor([0.33, 0.34, 0.33]), torch.tensor([-3., 0., 3.]), torch.tensor([2., 2., 2.,])),
    GaussMixClass.GaussMix(torch.tensor([0.45, 0.1, 0.45]), torch.tensor([-10., 0., 2.]), torch.tensor([10., 1., 10.,]))]]

    print(activation_input_moments(z, W, mix_size))