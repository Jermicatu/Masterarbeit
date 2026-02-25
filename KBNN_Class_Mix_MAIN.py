import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import GaussMixClass

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

        weights = [[[None for _ in range(dimensions[l])]
                        for _ in range(dimensions[l + 1])]
                        for l in range(len(dimensions) - 1)]
        for l in range(0, len(dimensions) - 1):
            for j in range(0, dimensions[l+1]):
                for i in range(0, dimensions[l]):
                    weights[l][j][i] = GaussMixClass.GaussMix(torch.tensor([0.5, 0.5]), torch.tensor([-1., 1.]), torch.tensor([1., 1.]))

        self.weights = weights

    def static_output(self, x: torch.tensor):
        # TODO add Bias
        dimensions = self.dimensions
        weights = self.weights
        functions = self.functions

        w = [[[None for _ in range(dimensions[l])]
                    for _ in range(dimensions[l + 1])]
                    for l in range(len(dimensions) - 1)]
        for l in range(0, len(dimensions) - 1):
            for j in range(0, dimensions[l+1]):
                for i in range(0, dimensions[l]):
                    w[l][j][i] = weights[l][j][i].eval_random()

        layer_input = x
        for l in range(0, len(dimensions) - 1):
            layer_output = torch.zeros(dimensions[l+1])
            f = functions[l]
            for j in range(0, dimensions[l+1]):
                layer_output[j] = torch.matmul(layer_input, torch.tensor(w[l][j]))
            layer_input = f(layer_output)

        return layer_input
    
    def uncertainty_quantification(self):

        return NotImplementedError
    
def test_1(): 
    dimensions = [1, 100, 1]
    functions = [relu, id]
    my_KBNN = KBBN_mix_class(dimensions, functions)

    x = torch.tensor([1.])

    print(my_KBNN.static_output(x))

def test_2(size: int):
    dimensions = [1, 100, 1]
    functions = [relu, id]
    my_KBNN = KBBN_mix_class(dimensions, functions)

    x = torch.tensor([1.])
    y = torch.zeros(size)

    for i in range(0, size):
        y[i] = my_KBNN.static_output(x).item()

    plt.hist(y, bins=20)
    plt.show()

if __name__ == "__main__":
    test_2(1000)