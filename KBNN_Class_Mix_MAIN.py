import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import GaussMixClass
from Pb_solver import mix_density_mult_approx

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp

def id(x):
    return x

class KBBN_mix_class:
    def __init__(self, dimensions, functions, starting_variance = 1):
        self.dimensions = dimensions
        self.functions = functions
        self.noise = 0.01

        assert len(dimensions) - len(functions) == 1, f"dimensions should have one more entry then functions"

        weights = [[[None for _ in range(dimensions[l] + 1)]
                        for _ in range(dimensions[l + 1])]
                        for l in range(len(dimensions) - 1)]

        for l in range(0, len(dimensions) - 1):
            for j in range(0, dimensions[l+1]):
                for i in range(0, dimensions[l] + 1):
                    weights[l][j][i] = GaussMixClass.GaussMix(torch.tensor([0.5, 0.5]), torch.tensor([-1., 1.]), torch.tensor([1., 1.]))

        self.weights = weights

    def static_output(self, x: torch.tensor):
        dimensions = self.dimensions
        weights = self.weights
        functions = self.functions

        w = [[[None for _ in range(dimensions[l] + 1)]
                    for _ in range(dimensions[l + 1])]
                    for l in range(len(dimensions) - 1)]
        for l in range(0, len(dimensions) - 1):
            for j in range(0, dimensions[l+1]):
                for i in range(0, dimensions[l] + 1):
                    w[l][j][i] = weights[l][j][i].eval_random()

        layer_input = x
        for l in range(0, len(dimensions) - 1):
            layer_output = torch.zeros(dimensions[l+1])
            f = functions[l]
            for j in range(0, dimensions[l+1]):
                layer_output[j] = torch.matmul(layer_input, torch.tensor(w[l][j][0:dimensions[l]])) + w[l][j][-1] # Bias
            layer_input = f(layer_output)

        return layer_input
    
    def uncertainty_quantification(self, x: torch.tensor) -> GaussMixClass.GaussMix:
        # 1D input and output
        dimensions = self.dimensions
        weights = self.weights
        f = self.functions

        z_l = [None] * dimensions[0]
        z_l[0] = GaussMixClass.GaussMix(torch.tensor([1]), torch.tensor([x[0]]), torch.tensor([0.1]))

        # in each layer l
        for l in range(0, len(dimensions)-1):
            a_l = [None] * dimensions[l+1]

            # get each a^(l+1)
            for j in range(0, dimensions[l+1]):
                a_l[j] = z_l[0].approx_mul(weights[l][j][0])
                for i in range(0, dimensions[l]):
                    a_l[j] = a_l[j].approx_add(z_l[i].approx_mul(weights[l][j][i]))
                a_l[j] = a_l[j].approx_add(weights[l][j][-1]) # Bias

            z_l = [None] * dimensions[l+1]

            # get each z^(l+1)
            for j in range(0, dimensions[l+1]):
                z_l[j] = a_l[j].approx_activation(f[l])

        return z_l[0]
    
    def forward_pass(self, x: torch.tensor):
        # 1D input and output
        dimensions = self.dimensions
        weights = self.weights
        f = self.functions

        output = [[None, None] for _ in range(len(dimensions) - 1)]

        z_l = [None] * dimensions[0]
        z_l[0] = GaussMixClass.GaussMix(torch.tensor([1]), torch.tensor([x[0]]), torch.tensor([0.1]))

        # in each layer l
        for l in range(0, len(dimensions)-1):
            a_l = [None] * dimensions[l+1]

            # get each a^(l+1)
            for j in range(0, dimensions[l+1]):
                a_l[j] = z_l[0].approx_mul(weights[l][j][0])
                for i in range(0, dimensions[l]):
                    a_l[j] = a_l[j].approx_add(z_l[i].approx_mul(weights[l][j][i]))
                a_l[j] = a_l[j].approx_add(weights[l][j][-1]) # Bias

            z_l = [None] * dimensions[l+1]

            # get each z^(l+1)
            for j in range(0, dimensions[l+1]):
                z_l[j] = a_l[j].approx_activation(f[l])
            output[l][0] = [gm.clone() for gm in a_l] 
            output[l][1] = [gm.clone() for gm in z_l]

        return output
    
    def backward_pass(self, x: torch.tensor, y: torch.tensor, data_variance: float):
        forward_pass_data = self.forward_pass(x)
        dimensions = self.dimensions

        L = dimensions[-1]
        z_l_fp = forward_pass_data[L-1][1]
        a_l_fp = forward_pass_data[L][0]

        beta_l = [GaussMixClass.GaussMix(torch.tensor([1.0]), torch.tensor([-y[i]]), torch.tensor([data_variance])) for i in range(0, L)]
        z_l_bp = [mix_density_mult_approx(z_l_fp[i], beta_l[i]) for i in range(0, L)]

        # in each layer l backwards
        for l in range(len(dimensions) - 2, 0, -1):
            z_l_fp = forward_pass_data[l-1][1]
            a_l_fp = forward_pass_data[l][0]
            # section for a^l
            a_l_bp = [mix_density_mult_approx(z_l_bp[i], a_l_fp[i]) for i in range(0, dimensions[l])] # TODO change mix_density_mult_approx so it takes 2 mixtures as arguments
            # section for and w^l
            # update w here
            # section for and z^l
            z_l_bp = None
    
def test_1(): 
    dimensions = [1, 3, 1]
    functions = [relu, id]
    my_KBNN = KBBN_mix_class(dimensions, functions)

    x = torch.tensor([1.])

    print(my_KBNN.static_output(x))

def test_2(size: int):
    dimensions = [1, 3, 1]
    functions = [relu, id]
    my_KBNN = KBBN_mix_class(dimensions, functions)

    x = torch.tensor([1.])
    y = torch.zeros(size)

    for i in range(0, size):
        y[i] = my_KBNN.static_output(x).item()

    plt.hist(y, bins=20)
    plt.show()

def test_3():
    dimensions = [1, 3, 1]
    functions = [relu, id]
    my_KBNN = KBBN_mix_class(dimensions, functions)

    mix = my_KBNN.uncertainty_quantification(torch.tensor([1.]))

    mix2 = my_KBNN.forward_pass(torch.tensor([1.]))

    x = torch.linspace(-60, 60, 400)

    f_1 = mix.eval(x)
    plt.plot(x, f_1, label="mix_1")

    print("================================")
    print(mix.m.size(0))
    print(torch.sum(mix.w))

    dx = x[1] - x[0]
    f_1_int = dx * (f_1.sum() - 0.5*f_1[0] - 0.5*f_1[-1])
    print(f_1_int)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((0, 0.025))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # creatze histogram
    # test_2(1000)
    # create prediction
    test_3()