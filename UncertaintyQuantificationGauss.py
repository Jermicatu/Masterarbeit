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
        self.data_variance = data_variance
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions) - 1
        self.m = [torch.zeros(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        self.s = [starting_variance * torch.ones(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        
    def forwardPass(self, m_q, s_q, x_data, y_data):
        # Calculates the expected log probability of the data under given weights with mean m_q and var s_q
        dimensions = self.dimensions
        functions = self.functions
        data_variance = self.data_variance
        L = len(dimensions) - 1
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        sol = 0

        for i in range(len(x_data)):
            m_z_l = x_data[i:i+1]
            s_z_l = torch.zeros(1)
            y = y_data[i]
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(m_q[l][:,1:], m_z_l) + m_q[l][:,0]
                s_a_l = (s_q[l][:,0] + torch.matmul(m_q[l][:,1:] ** 2, s_z_l)
                + torch.matmul(s_q[l][:,1:], m_z_l ** 2) + torch.matmul(s_q[l][:,1:], s_z_l))

                m_z_l = (f(m_a_l + torch.sqrt(s_a_l)) + f(m_a_l - torch.sqrt(s_a_l))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l))**2 + f(m_a_l - torch.sqrt(s_a_l))**2) / 2 - m_z_l ** 2

            sol += - 0.5 * math.log(2 * math.pi * data_variance) - ((y - m_z_l)**2 + s_z_l) / (2 * data_variance)

        return sol
                
    def ELBO(self, m_q, s_q, x_data, y_data):
        # Calculates the ELBO of all our weights in the network between the current p and new q distributions
        L = len(self.dimensions) - 1
        m_p = self.m
        s_p = self.s

        my_ELBO = self.forwardPass(m_q, s_q, x_data, y_data)

        flat_mq = torch.cat([ml.flatten() for ml in m_q])
        flat_sq = torch.cat([sl.flatten() for sl in s_q])
        flat_mp = torch.cat([ml.flatten() for ml in m_p])
        flat_sp = torch.cat([sl.flatten() for sl in s_p])

        for mq,sq,mp,sp in zip(flat_mq, flat_sq, flat_mp, flat_sp):
            my_ELBO += 0.5 * torch.log(sp) + ((mq-mp)**2 + sq) / (2 * sp) - 0.5 * torch.log(sq) - 0.5

        return my_ELBO
    
    def train(self, x_data, y_data, lr_m=0.02, lr_s=0.01, epochs=500):

        m_q = [m.clone().detach().requires_grad_(True) for m in self.m]
        log_s_q = [torch.log(s.clone().detach()).requires_grad_(True) for s in self.s]

        for epoch in range(epochs):
            s_q_fixed = [torch.exp(ls).detach() for ls in log_s_q]

            loss_m = -self.ELBO(m_q, s_q_fixed, x_data, y_data)
            loss_m.backward()

            with torch.no_grad():
                for m in m_q:
                    if m.grad is not None:
                        m -= lr_m * m.grad      # ascent:  m <- m + lr * grad_ELBO

            for m in m_q:
                if m.grad is not None:
                    m.grad.zero_()

            m_q_fixed = [m.detach() for m in m_q]
            s_q_var = [torch.exp(ls) for ls in log_s_q]

            loss_s = -self.ELBO(m_q_fixed, s_q_var, x_data, y_data)
            loss_s.backward()

            with torch.no_grad():
                for ls in log_s_q:
                    if ls.grad is not None:
                        ls -= lr_s * ls.grad    # ascent on log(s)

            for ls in log_s_q:
                if ls.grad is not None:
                    ls.grad.zero_()

            loss_s = -self.ELBO(m_q_fixed, s_q_var, x_data, y_data)
            loss_s.backward()


    

def test1():
    dimensions = [1, 2, 1]
    functions = [torch.tanh, torch.tanh]
    network = PaperBNN(dimensions, functions)

    x_data = torch.linspace(-3, 3, steps=50)
    y_data = torch.cos(x_data)

    m_q = network.m
    s_q = network.s

    network.ELBO(m_q, s_q, x_data, y_data)

    
if __name__ == "__main__":
    print("INITIATE TESTS:")
    test1()