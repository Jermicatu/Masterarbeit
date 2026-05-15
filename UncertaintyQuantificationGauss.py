import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import GaussMixClass
from logger_config import logger

class PaperBNN:
    def __init__(self, dimensions, functions, starting_variance = 5, data_variance = 1):
        self.dimensions = dimensions
        self.functions = functions
        self.noise = noise
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions-1)
        self.m = [torch.zeros(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        self.s = [starting_variance * torch.ones(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        
    def forwardPass(self, m_q, s_q, x_data, y_data):
        # Calculates the expected log probability of the data under given weights with mean m_q and var s_q
        dimensions = self.dimensions
        functions = self.functions
        data_variance = self.data_variance
        L = len(dimensions-1)
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        sol = 0

        for i in range(len(x_data)):
            m_z_l = x_data[i]
            s_z_l = torch.zeros(m_z_l.size(0))
            y = y_data[i]
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(self.m[l][1:], m_z_l) + m_q[l][0]
                s_a_l = (s_q[l][0] + torch.matmul(m_q[l][1:] ** 2, s_z_l)
                + torch.matmul(s_q[l][1:], m_z_l ** 2) + torch.matmul(s_q[l][1:], s_z_l))

                m_z_l = (f(m_a_l + torch.sqrt(s_a_l)) + f(m_a_l - torch.sqrt(s_a_l))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l))**2 + f(m_a_l - torch.sqrt(s_a_l))**2) / 2 - m_z_l ** 2

            sol += 0.5*math.log(2*math.pi*data_variance) - ((y - m_z_l)**2 - s_z_l) / (2 * data_variance)

        return sol
                
    def ELBO(self, m_q, s_q, x_data, y_data):
        # Calculates the ELBO of all our weights in the network between the current p and new q distributions
        ELBO = None

        return ELBO

    
if __name__ == "__main__":
    print("INITIATE TESTS:")