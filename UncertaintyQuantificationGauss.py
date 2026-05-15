import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger

class PaperBNN:
    def __init__(self, dimensions, functions, starting_variance = 1, noise = 0.01):
        self.dimensions = dimensions
        self.functions = functions
        self.noise = noise
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions-1)
        self.m = [torch.zeros(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        self.s = [5 * torch.ones(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        
    def forwardPass(self, m_q, s_q, x_data, y_data):
        dimensions = self.dimensions
        functions = self.functions
        L = len(dimensions-1)
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        for i in range(len(x_data)):
            z_l = x_data[i]
            s_z_l = torch.zeros(z_l.size(0))
            y = y_data[i]
            for l in range(L):
                
                
    
if __name__ == "__main__":
    print("INITIATE TESTS:")