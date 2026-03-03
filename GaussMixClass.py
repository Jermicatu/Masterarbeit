import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from logger_config import logger


class GaussMixProduct:
    mix1 = None
    mix2 = None

    def __init__(self, weight1: torch.tensor, mean1: torch.tensor, sigma1: torch.tensor, 
                 weight2: torch.tensor, mean2: torch.tensor, sigma2: torch.tensor):
        """
        Docstring for __init__
        
        :param self: Description
        :param weight1: Description
        :type weight1: torch.tensor
        :param mean1: Description
        :type mean1: torch.tensor
        :param sigma1: Description
        :type sigma1: torch.tensor
        :param weight2: Description
        :type weight2: torch.tensor
        :param mean2: Description
        :type mean2: torch.tensor
        :param sigma2: Description
        :type sigma2: torch.tensor
        """
        self.mix1 = GaussMix(weight1, mean1, sigma1)
        self.mix2 = GaussMix(weight2, mean2, sigma2)

    def eval_tilde(self, x: torch.tensor, gamma: float, noise: float) -> torch.tensor:
        """
        Docstring for eval_tilde
        
        :param self: Description
        :param x: Description
        :type x: torch.tensor
        :param gamma: Description
        :type gamma: float
        :param noise: Description
        :type noise: float
        :return: Description
        :rtype: Any
        """
        sol1 = self.mix1.eval_gamma(x, gamma, noise)
        sol2 = self.mix2.eval(x)

        return sol1 * sol2

class GaussMix:
    w = None
    m = None
    s = None

    def __init__(self, weight: torch.tensor, mean: torch.tensor, sigma: torch.tensor):
        """
        Docstring for __init__
        
        :param self: Description
        :param weight: Description
        :type weight: torch.tensor
        :param mean: Description
        :type mean: torch.tensor
        :param sigma: Description
        :type sigma: torch.tensor        
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

    def __add__(self, other: "GaussMix"):
        """
        Docstring for __add__
        
        :param self: Description
        :param other: Description
        :type other: "GaussMix"
        """
        if not isinstance(other, GaussMix):
            return None # NotImplementedError
        if self.w.size(0) != other.w.size(0):
            raise AttributeError("Gauss mixes don't have the same length!")
        
        return GaussMix(self.w + other.w, self.m + other.m, self.s + other.s)
    
    def __radd__(self, other: "GaussMix"):
        """
        Docstring for __radd__
        
        :param self: Description
        :param other: Description
        :type other: "GaussMix"
        """
        return self.__add__(other)
    
    def __mul__(self, other: "GaussMix"):
        """
        Docstring for __mul__
        
        :param self: Description
        :param other: Description
        :type other: "GaussMix"
        """
        if type(other) == int or type(other) == float or type(other):
            return GaussMix(self.w * other, self.m * other, self.s * other)
        else:
            return NotImplementedError
        
    def __rmul__(self, other: "GaussMix"):
        """
        Docstring for __rmul__
        
        :param self: Description
        :param other: Description
        :type other: "GaussMix"
        """
        return self.__mul__(other)
    
    def approx_mul(self, mix: "GaussMix") -> "GaussMix":
        """
        Docstring for approx_mul
        
        :param self: Description
        :param mix: Description
        :type mix: "GaussMix"
        :return: Description
        :rtype: GaussMix
        """
        size_1 = self.m.size(0)
        size_2 = mix.m.size(0)

        w_new = torch.zeros(size_1 * size_2)
        m_new = torch.zeros(size_1 * size_2)
        s_new = torch.zeros(size_1 * size_2)
        
        for i in range(0, size_1):
            for j in range(0, size_2):
                w_new[i*size_2 + j] = self.w[i]*mix.w[j]
                m_new[i*size_2 + j] = self.m[i]*mix.m[j]
                s_new[i*size_2 + j] = math.sqrt((self.s[i]*mix.s[j])**2 + (self.s[i]*mix.m[j])**2 + (self.m[i]*mix.s[j])**2)

        return GaussMix(w_new, m_new, s_new)
    
    def approx_add(self, mix: "GaussMix") -> "GaussMix":
        """
        Docstring for approx_add
            
        :param self: Description
        :param mix: Description
        :type mix: "GaussMix"
        :return: Description
        :rtype: GaussMix
        """
        size_1 = self.m.size(0)
        size_2 = mix.m.size(0)

        w_new = torch.zeros(size_1 * size_2)
        m_new = torch.zeros(size_1 * size_2)
        s_new = torch.zeros(size_1 * size_2)
            
        for i in range(0, size_1):
            for j in range(0, size_2):
                w_new[i*size_2 + j] = self.w[i] * mix.w[j]
                m_new[i*size_2 + j] = self.m[i] + mix.m[j]
                s_new[i*size_2 + j] = math.sqrt(self.s[i]**2 + mix.s[j]**2)

        return GaussMix(w_new, m_new, s_new)
    
    def approx_activation(self, f) -> "GaussMix":
        """
        Docstring for approx_activation
        
        :param self: Description
        :param f: Description
        :return: Description
        :rtype: GaussMix
        """
        size_1 = self.m.size(0)

        w_new = torch.zeros(size_1)
        m_new = torch.zeros(size_1)
        s_new = torch.zeros(size_1)
            
        for i in range(0, size_1):
            f_1 = f(torch.tensor([self.s[i] + self.m[i]]))
            f_2 = f(torch.tensor([- self.s[i] + self.m[i]]))
            m_new[i] = 0.5 * (f_1      + f_2)
            s_new[i] = 0.5 * (f_1 ** 2 + f_2 ** 2) - m_new[i] ** 2
        s_new = torch.clamp(s_new, min = 0.01)
        return GaussMix(self.w, m_new, s_new)
    
    def split(self, position: int):
        """
        Docstring for split
        
        :param self: Description
        :param position: Description
        :type position: int
        """
        if type(position) != int:
            return NotImplementedError
        if position > self.w.size(0):
            raise AttributeError("Can't split there! Postition is larger than the size of the Gauss mix!")
        
        return NotImplementedError
        # TODO

    def eval(self, x: torch.tensor) -> torch.tensor:
        """
        Docstring for eval
        
        :param self: Description
        :param x: Description
        :type x: torch.tensor
        :return: Description
        :rtype: Any
        """
        x = x[:, None]

        m_ = self.m[None, :]
        s_ = self.s[None, :] ** 2
        w_ = self.w[None, :]

        # Gaussian formula
        coef = 1.0 / torch.sqrt(2 * math.pi * s_)
        exponent = torch.exp(-0.5 * (x - m_) ** 2 / s_)
        gaussians = coef * exponent  # shape (K, L)

        # Weighted sum over components
        return (w_ * gaussians).sum(dim=1)
    
    def eval_random(self):
        m_ = self.m
        s_ = self.s
        w_ = self.w

        dist = torch.distributions.Normal(m_, s_)
        sample = dist.sample() * w_

        return sample.sum(dim=0).item()

    def eval_individual(self, x):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
        Output: f(x) for each individual Gaussian eta_i
        """
        f_i = torch.zeros(x.size(0), self.w.size(0))
        for i in range(0,x.size(0)):
            for j in range(0, self.w.size(0)):
                f_i[i, j] = self.w[j] * math.exp(-0.5 * (x[i] - self.m[j])**2 / (self.s[j] ** 2)) / math.sqrt(2 * math.pi * (self.s[j] ** 2))
        
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
        s_ = self.s[None, :] ** 2
        w_ = self.w[None, :]

        # coef = 1.0 / torch.sqrt(2. * math.pi * s_)
        coef = 1.0
        exponent = torch.exp(-0.5 * (x - m_) ** 2. / (s_ * factor))
        gaussians = coef * exponent

        return (w_ * gaussians).sum(dim=1)
    
    def eval_gamma_ver2(self, x, gamma, noise):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
                @param gamma: float in range [0,1]
                @param noise: float 
        Output: f(x) one dimensional vector of same shape as x where f(x) is a gaussian mix f defined via eta, gamma and noise
        """

        x = x[:, None]
        m_ = self.m[None, :]
        s_ = self.s[None, :] ** 2
        w_ = self.w[None, :]

        coef = 1.0 / torch.sqrt(2. * math.pi * s_)
        exponent = torch.exp(-0.5 * (x - m_) ** 2. / s_ )
        gaussians = (coef * exponent) ** gamma

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
        s_ = self.s[None, :] ** 2
        w_ = self.w[None, :]

        coef = - (noise + gamma) * (x - m_) ** 2 / ((1 + noise)**2 * s_) # * torch.sqrt(2. * math.pi * s_))
        exponent = torch.exp(-0.5 * (x - m_) ** 2 / (s_*factor))
        gaussians = coef * exponent 

        return (w_ * gaussians).sum(dim=1)
    
    def eval_gamma_diff_ver2(self, x, gamma, noise):
        """
        Input:  @param x: one dimensional tensor stating the x-values for which we want an output
                @param eta: two dimensional tensor of form [w, mu, sigma]
                @param gamma: float in range [0,1]
                @param noise: float 
        Output: f'(x) one dimensional vector of same shape as x where f'(x) is a gaussian mix f defined via eta, gamma and noise
        """

        x = x[:, None]
        m_ = self.m[None, :]
        s_ = self.s[None, :] ** 2
        w_ = self.w[None, :]

        coef = 1.0 / torch.sqrt(2. * math.pi * s_)
        exponent = torch.exp(-0.5 * (x - m_) ** 2. / s_ )
        exponent = torch.clamp(exponent, min = 1.0e-6)
        gaussians = torch.log(coef * exponent) * ((coef * exponent) ** gamma)

        return (w_ * gaussians).sum(dim=1)
    
def id(x):
    return x

def test_1():
    w1 = torch.tensor([0.3, 0.4, 0.3])
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

def test_2():
    w1 = torch.tensor([0.3, 0.4, 0.3])
    m1 = torch.tensor([10., 0., -10.])
    s1 = torch.tensor([1., 1., 1.])

    mix1 = GaussMix(w1, m1, s1)

    gammas = torch.linspace(0, 1, 100)
    noise = 0.01
    f_gamma = torch.linspace(0, 1, 100)
    f_gamma_diff = torch.linspace(0, 1, 100)
    for i in range(0, 100):
        f_gamma[i] = mix1.eval_gamma(torch.tensor([0]), gammas[i], noise)
        f_gamma_diff[i] = mix1.eval_gamma_diff(torch.tensor([0]), gammas[i], noise)
            
    plt.plot(gammas, f_gamma, label="f_gamma")
    plt.plot(gammas, f_gamma_diff, label="f_gamma_diff")
        
    dgamma = gammas[1] - gammas[0]
    plt.plot(gammas, np.gradient(f_gamma, dgamma), label="f_gamma_diff_np")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def test_3():
    w1 = torch.tensor([0.3, 0.4, 0.3])
    m1 = torch.tensor([10., 0., -10.])
    s1 = torch.tensor([1., 1., 1.])
    w2 = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.2])
    m2 = torch.tensor([10., 0., -1., -2, -5])
    s2 = torch.tensor([3., 1., 1., 1., 1.])

    my_Gauss = GaussMixProduct(w1, m1, s1, w2, m2, s2)
    noise = 0.01
    x = torch.linspace(-30, 30, 400)

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

def test_4():
    w1 = torch.tensor([0.5, 0.5])
    m1 = torch.tensor([-2., 2.])
    s1 = torch.tensor([1., 1.])

    w2 = torch.tensor([0.333, 0.334, 0.333])
    m2 = torch.tensor([-4., 0., 4.])
    s2 = torch.tensor([1., 1., 1.])

    w3 = 0.5 * torch.tensor([0.333, 0.333, 0.334, 0.334, 0.333, 0.333])
    m3 = torch.tensor([-8., -8., 0., 0., 8., 8.])
    s3 = torch.sqrt(torch.tensor([21., 21., 5., 5., 21., 21.]))

    mix1 = GaussMix(w1, m1, s1)
    mix2 = GaussMix(w2, m2, s2)
    mix3 = GaussMix(w3, m3, s3)

    mix_mul = mix1.approx_mul(mix2)

    x = torch.linspace(-14, 14, 400)

    f_1 = mix1.eval(x)
    plt.plot(x, f_1, label="mix_1")

    f_2 = mix2.eval(x)
    plt.plot(x, f_2, label="mix_2")

    f_3 = mix3.eval(x)
    plt.plot(x, f_3, label="mix_sol")

    f_mul = mix_mul.eval(x)
    plt.plot(x, f_mul, label="mix_mul")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def test_5():
    w1 = torch.tensor([0.5, 0.5])
    m1 = torch.tensor([-2., 2.])
    s1 = torch.tensor([1., 1.])

    w2 = torch.tensor([0.333, 0.334, 0.333])
    m2 = torch.tensor([-4., 0., 4.])
    s2 = torch.tensor([1., 1., 1.])

    w3 = 0.5 * torch.tensor([0.333, 0.333, 0.334, 0.334, 0.333, 0.333])
    m3 = torch.tensor([-6., -2., -2., 2., 2., 6.])
    s3 = torch.sqrt(torch.tensor([2., 2., 2., 2., 2., 2.]))

    mix1 = GaussMix(w1, m1, s1)
    mix2 = GaussMix(w2, m2, s2)
    mix3 = GaussMix(w3, m3, s3)

    mix_add = mix1.approx_add(mix2)

    x = torch.linspace(-14, 14, 400)

    f_1 = mix1.eval(x)
    plt.plot(x, f_1, label="mix_1")

    f_2 = mix2.eval(x)
    plt.plot(x, f_2, label="mix_2")

    f_3 = mix3.eval(x)
    plt.plot(x, f_3, label="mix_sol")

    f_add = mix_add.eval(x)
    plt.plot(x, f_add, label="mix_add")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def test_6():
    w1 = torch.tensor([0.5, 0.5])
    m1 = torch.tensor([-3., 3.])
    s1 = torch.tensor([1., 1.])

    mix1 = GaussMix(w1, m1, s1)
    relu = torch.relu

    mix_actiation = mix1.approx_activation(id)

    x = torch.linspace(-14, 14, 400)

    f_1 = mix1.eval(x)
    plt.plot(x, f_1, label="f_1")

    f_activation = mix_actiation.eval(x)
    plt.plot(x, f_activation, label="f_activation")

    print(mix_actiation.w)
    print(mix_actiation.m)
    print(mix_actiation.s)

    individual = mix_actiation.eval_individual(x)

    for i in range(0, individual.size(1)):
        plt.plot(x, individual[:,i], "r--", label="f_activation individual")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def test_7():
    w1_test = torch.tensor([0.5, 0.5])
    m1_test = torch.tensor([-2., 2.])
    s1_test = torch.tensor([1., 1.])

    w2_test = torch.tensor([0.333, 0.334, 0.333])
    m2_test = torch.tensor([-4., 0., 4.])
    s2_test = torch.tensor([1., 1., 1.])

    test_mix = GaussMixProduct(w1_test, m1_test, s1_test, w2_test, m2_test, s2_test)

    x = torch.linspace(-14, 14, 400)
    dx = x[1] - x[0]

    f_1_g_0 = test_mix.eval_tilde(x, 0, 0.01)
    f_1_g_0 = f_1_g_0 / (dx * (f_1_g_0.sum()- 0.5*f_1_g_0[0] - 0.5*f_1_g_0[-1]))
    plt.plot(x, f_1_g_0, label="0")
    f_1_g_033 = test_mix.eval_tilde(x, 0.3, 0.01)
    f_1_g_033 = f_1_g_033 / (dx * (f_1_g_033.sum()- 0.5*f_1_g_033[0] - 0.5*f_1_g_033[-1]))
    plt.plot(x, f_1_g_033, label="0.33")
    f_1_g_066 = test_mix.eval_tilde(x, 0.66, 0.01)
    f_1_g_066 = f_1_g_066 / (dx * (f_1_g_066.sum()- 0.5*f_1_g_066[0] - 0.5*f_1_g_066[-1]))
    plt.plot(x, f_1_g_066, label="0.66")
    f_1_g_1 = test_mix.eval_tilde(x, 1, 0.01)
    f_1_g_1 = f_1_g_1 / (dx * (f_1_g_1.sum()- 0.5*f_1_g_1[0] - 0.5*f_1_g_1[-1]))
    plt.plot(x, f_1_g_1, label="1")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def test_8():
    w1 = torch.tensor([0.5, 0.5])
    m1 = torch.tensor([-2., 2.])
    s1 = torch.tensor([1., 1.])

    mix1 = GaussMix(w1, m1, s1)

    print(mix1.eval_random())


if __name__ == "__main__":
    """
    Test here
    """

    # test_8()

    # tests the individual Gaussians    
    # test_1()

    # tests the gamma differential function
    test_2()

    # tests the f_tilde for gamma 0 and 1
    # test_3()

    # tests approx_mul
    # test_4()

    # tests approx_add
    # test_5()

    # tests approx_activation
    test_6()

    # Tbh I frogor what this was for sth sth normalized functions
    # test_7()

"""
ANDI TIPPS

canvas board
matcha.io
obsidian
"""