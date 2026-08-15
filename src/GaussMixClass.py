import torch
import math

DEFAULT_MAX_MIX = 100

class GaussMixProduct:
    mix1 = None
    mix2 = None

    def __init__(self, weight1: torch.tensor, mean1: torch.tensor, sigma1: torch.tensor, 
                 weight2: torch.tensor, mean2: torch.tensor, sigma2: torch.tensor, max_mix: int = DEFAULT_MAX_MIX):
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
        :param max_mix: Description
        :type Max_mix: int
        """
        self.mix1 = GaussMix(weight1, mean1, sigma1, max_mix)
        self.mix2 = GaussMix(weight2, mean2, sigma2, max_mix)

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

    def __init__(self, weight: torch.tensor, mean: torch.tensor, sigma: torch.tensor, max_mix: int = DEFAULT_MAX_MIX):
        """
        Docstring for __init__
        
        :param self: Description
        :param weight: Description
        :type weight: torch.tensor
        :param mean: Description
        :type mean: torch.tensor
        :param sigma: Description
        :type sigma: torch.tensor
        :param max_mix: Description
        :type Max_mix: int      
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
        self.max_mix = max_mix

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

    def mean(self):
        return torch.matmul(self.w, self.m)
    
    def mixture_moments(self, n: int):
        """creates 3n-1 moments of the mixture up to 

        Args:
            n (int): size of the Gauss mix we want to approximate

        Returns:
            torch.tensor: tensor of 3n-1 first moments of the mixture
        """
        w = self.w
        m = self.m
        s = self.s
        max_moment = 3 * n - 1

        def gaussian_moments(m_single, s_single):
            """
            Raw non-central moments M_1 .. M_{max_moment} of the mixture.
            Uses the recurrence:  M_k = mu * M_{k-1} + (k-1) * v * M_{k-2}
            """
            m_list = [torch.tensor(1.)]
            if max_moment >= 1:
                m_list.append(m_single)
            if max_moment >= 2:
                m_list.append(m_single**2 + s_single)
            for k in range(2, max_moment):      # k here corresponds to (k+1)-th moment
                m_list.append(m_single * m_list[k-1] + k * s_single * m_list[k-2])
        
            return torch.stack(m_list)

        parts = [w[i] * gaussian_moments(m[i], s[i]) for i in range(n)]

        return torch.stack(parts).sum(dim=0)
    
    def clone(self):
        return GaussMix(self.w.clone(), self.m.clone(), self.s.clone(), self.max_mix)
    
    def normalize(self):
        w = self.w

        sum = torch.sum(w)
        self.w = w/sum

    def add_const(self, x: float):
        self.m += x

    def mul_const(self, x: float):
        self.m *= x
        self.s *= x

    def approx_mul(self, mix: "GaussMix", max_mix: int = DEFAULT_MAX_MIX) -> "GaussMix":
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

        return GaussMix(w_new, m_new, s_new, max_mix)
    
    def approx_add(self, mix: "GaussMix", max_mix: int = DEFAULT_MAX_MIX) -> "GaussMix":
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

        return GaussMix(w_new, m_new, s_new, max_mix)
    
    def approx_activation(self, f, max_mix: int = DEFAULT_MAX_MIX) -> "GaussMix":
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
        return GaussMix(self.w, m_new, s_new, max_mix)
    
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