import torch

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp


def evaluate_functions(function_name):
    if function_name == "id":
        return lambda val: val
    if function_name == "sigmoid":
        return lambda val: torch.sigmoid(val)
    
class Layer:
    def __init__(self, input_size, dimension, variance, activation_function_name):
        self.mean_weights = torch.normal(0, 1, size = (input_size+1, dimension), dtype=torch.float32)
        self.sigma_weights = variance * torch.diag_embed(torch.ones(dimension, input_size+1, dtype=torch.float32))
        self.activation_function_name = activation_function_name

class NetworkClass:
    def __init__(self, input_size, dimensions, function_names, starting_variance = 1):
        network = [None] * len(dimensions)

        network[0] = Layer(input_size, dimensions[0], starting_variance, function_names[0])

        for i in range(1, len(dimensions)):
            network[i] = Layer(dimensions[i-1]+1, dimensions[i], starting_variance, function_names[i])

        self.input_size = input_size
        self.dimensions = dimensions
        self.function_names = function_names
        self.noise = 0.01
        self.network = network

        self.num_layers = len(dimensions)

    def meanOutput(self, input_vector):
        vector = input_vector
        if len(vector.size()) == 0:
            vector = vector.unsqueeze(0)
        vector = vector.to(torch.float32)

        for i in range(self.num_layers):
            evalue = torch.flatten(
                torch.matmul(
                    torch.cat((vector, torch.ones(1)), 0), self.network[i].mean_weights)) 
            vector = evaluate_functions(self.function_names[i])(evalue)

        return vector

    def forwardPass(self, vector):
        forwardPassData = [None] * self.num_layers

        mean_z_old = vector
        sigma_z_old = torch.zeros(self.input_size+1, self.input_size+1, dtype=torch.float32)

        for i in range(self.num_layers):


            if self.function_names[i] == "sigmoid":
                ...#


                