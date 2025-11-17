import torch
import math
import matplotlib.pyplot as plt
import numpy as np

class GaussMixProduct:
    mix1 = None
    mix2 = None

    def __init__(self, weight1, mean1, sigma1, weight2, mean2, sigma2):
        self.mix1 = GaussMix(weight1, mean1, sigma1)
        self.mix2 = GaussMix(weight2, mean2, sigma2)

    def eval_tilde(self, x, gamma, noise):
        sol1 = self.mix1.eval_gamma(x, gamma, noise)
        sol2 = self.mix2.eval(x)
        return sol1 * sol2

class GaussMix:
    w = None
    m = None
    s = None

    def __init__(self, weight, mean, sigma):
        """
        Save the weight, mean and variance each as a torch vector of same length
        """
        if weight.dim() != 1:
            raise AttributeError("dimension of weight is not one!")
        if mean.dim() != 1:
            raise AttributeError("dimension of mean is not one!")
        if sigma.dim() != 1:
            raise AttributeError("dimension of sigma is not one!")
        if not (weight.size(0) == mean.size(0) and weight.size(0) == sigma.size(0)):
            raise AttributeError("inputs don't have the same length!")

        self.w = weight
        self.m = mean
        self.s = sigma

    def __add__(self, other):
        if not isinstance(other, GaussMix):
            return None # NotImplementedError
        if self.w.size(0) != other.w.size(0):
            raise AttributeError("Gauss mixes don't have the same length!")
        
        return GaussMix(self.w + other.w, self.m + other.m, self.s + other.s)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if type(other) == int or type(other) == float or type(other):
            return GaussMix(self.w * other, self.m * other, self.s * other)
        else:
            return NotImplementedError
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def split(self, position):
        if type(position) != int:
            return NotImplementedError
        if position > self.w.size(0):
            raise AttributeError("Can't split there! Postition is larger than the size of the Gauss mix!")
        
        return NotImplementedError
        # TODO

    def eval(self, x):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
        Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian mix f defined via eta
        """
        # assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device)), f"w must sum to 1, but got {w.sum().item()}"

        x = x[:, None]

        m_ = self.m[None, :]
        s_ = self.s[None, :]
        w_ = self.w[None, :]

        # Gaussian formula
        coef = 1.0 / torch.sqrt(2 * math.pi * s_)
        exponent = torch.exp(-0.5 * (x - m_) ** 2 / s_)
        gaussians = coef * exponent  # shape (K, L)

        # Weighted sum over components
        return (w_ * gaussians).sum(dim=1)

    def eval_individual(self, x):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
        Output: f(x) for each individual Gaussian eta_i
        """
        f_i = torch.zeros(x.size(0), self.w.size(0))
        for i in range(0,x.size(0)):
            for j in range(0, self.w.size(0)):
                f_i[i, j] = self.w[j] * math.exp(-0.5 * (x[i] - self.m[j])**2 / self.s[j]) / math.sqrt(2 * math.pi * self.s[j])
        
        return f_i
    
    def eval_gamma(self, x, gamma, noise):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
                @param gamma: float in range [0,1]
                @param noise: float 
        Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian mix f defined via eta, gamma and noise
        """
        factor = ((1 + noise)/(gamma+noise))**2

        x = x[:, None]
        m_ = self.m[None, :]
        s_ = self.s[None, :]
        w_ = self.w[None, :]

        # coef = 1.0 / torch.sqrt(2. * math.pi * s_)
        coef = 1.0
        exponent = torch.exp(-0.5 * (x - m_) ** 2. / (s_ * factor))
        gaussians = coef * exponent

        return (w_ * gaussians).sum(dim=1)
    
    def eval_gamma_diff(self, x, gamma, noise):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
                @param gamma: float in range [0,1]
                @param noise: float 
        Output: f'(x) one dimensional vector of same shape as x where f'(x) is a gaussian mix f defined via eta, gamma and noise
        """
        factor = ((1 + noise)/(gamma + noise))**2

        x = x[:, None]
        m_ = self.m[None, :]
        s_ = self.s[None, :]
        w_ = self.w[None, :]

        coef = (noise - gamma) * (x - m_) / ((1 + noise)**2 * s_) # * torch.sqrt(2. * math.pi * s_))
        exponent = torch.exp(-0.5 * (x - m_) ** 2 / (s_*factor))
        gaussians = coef * exponent 

        return (w_ * gaussians).sum(dim=1)

if __name__ == "__main__":
    """
    Test here
    """
    w1 = torch.tensor([0.2, 0.4, 0.2])
    m1 = torch.tensor([10., 0., -10.])
    s1 = torch.tensor([1., 1., 1.])
    w2 = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.2])
    m2 = torch.tensor([10., 0., -1., -2, -5])
    s2 = torch.tensor([3., 1., 1., 1., 1.])

    my_Gauss = GaussMixProduct(w1, m1, s1, w2, m2, s2)

    x = torch.linspace(-30, 30, 400)
    f_1 = my_Gauss.mix2.eval(x)
    plt.plot(x, f_1, label="f_1")

    individual = my_Gauss.mix2.eval_individual(x)

    for i in range(0, individual.size(1)):
        plt.plot(x, individual[:,i], "r--", label="f_1 individual")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

    gamma = 1
    noise = 0.00001
    f_gamma = my_Gauss.mix2.eval_gamma(x, gamma, noise)
    plt.plot(x, f_gamma, label="f_gamma")

    f_gamma_diff = my_Gauss.mix2.eval_gamma_diff(x, gamma, noise)
    plt.plot(x, f_gamma_diff, label="f_gamma_diff")
    
    dx = x[1] - x[0]
    plt.plot(x, np.gradient(f_gamma, dx), label="f_gamma_diff_np")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

    f_1 = my_Gauss.mix1.eval(x)
    plt.plot(x, f_1, label="f_1")

    f_2 = my_Gauss.mix2.eval(x)
    plt.plot(x, f_2, label="f_2")

    f_tild_0 = my_Gauss.eval_tilde(x, 0, noise)
    plt.plot(x, f_tild_0, label="f_tild_0")

    f_tild_1 = my_Gauss.eval_tilde(x, 1, noise)
    plt.plot(x, f_tild_1, label="f_tild_1")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

    print("====================================")

"""
ANDI TIPPS
P-matrix file
delta P file

if __name__ == "main"
    hier testen

1) Was ist main? Welche Schritte?

GaussApprox Ordner
- Approx
   -Gaussian_mix (Klasse)
    -solver (rk4/euler)
    -splitter (in margin, add_componen)

canvas board
matcha.io

"""