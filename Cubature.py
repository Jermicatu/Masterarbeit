import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import random

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp

def f1(x):
  return 0.45 * (cos(x+1) + 1)

def f2(x):
  return 0.5 * (sig(x+1) - sig(x-1) + sig(x))

def f3(x):
  return cos(x)

def f4(x):
  return x**3

def id(x):
  return x

def f5(x):
  return torch.sum(x)

def simple_cubature(f, d):
  """
  This code is for approximating an integral of the form \int\limits_{\mathbb{R}^d} f(x) \exp (-x^T x) dx
  Input:  @param f: function f(x)
          @param d: dimension of the input for the function f
  Output: @output sol: approximation of the integral
  """
  d = 5  # dimension
  I = torch.eye(d)
  sol = sum(f(I[:, i]) + f(-I[:, i]) for i in range(0, d))
  sol = sol/(2*d)
  return sol

def cubature_approx(f, mean, var):
  """
  This code is for approximating E(f(a)) and var(f(a))
  Input:  @param f: function f(x)
          @param: var: one dimensional torch vector of form [sigma_1^2, ..., sigma_d^2]
          @param: mean: one dimensional torch vector of form [mu_1, ..., mu_d]
  Output: @output mean_sol: approximation E(f(a))
          @output var_sol: approximation E(f^2(a)) - E(f(a))^2
  """
  assert var.size() == mean.size(), f"Shape mismatch between var and mean"

  mean_sol = f(torch.sqrt(var) + mean) + f(-torch.sqrt(var) + mean)
  mean_sol = mean_sol/math.sqrt(math.pi)

  var_sol = f(torch.sqrt(var) + mean)**2 + f(-torch.sqrt(var) + mean)**2
  var_sol = var_sol/math.sqrt(math.pi) - mean_sol**2

  return mean_sol, var_sol

def cubature_approx_mixture(f, mean, var):
  """
  This code is for approximating E(f(a)) and var(f(a)) where a is a Gaussian mixture
  Input:  @param f: function f(x) that takes each element in a tensor individualy
          @param: var: two dimensional torch matrix where each column is of the form [sigma_1^2, ..., sigma_d^2]
          @param: mean: two dimensional torch matrix of where each column is of the form [mu_1, ..., mu_d]
  Output: @output mean_sol: approximation E(f(a))
          @output var_sol: approximation E(f^2(a)) - E(f(a))^2
  """
  assert var.size() == mean.size(), f"Shape mismatch between var and mean"

  d = var.size(1)

  v = torch.sqrt(var)
  term1 = f(v + mean)
  term2 = f(-v + mean)

  mean_sol = torch.sum(term1 + term2, dim=1)
  mean_sol = mean_sol/math.sqrt(math.pi)

  var_sol = torch.sum(term1**2 + term2**2, dim=1)
  var_sol = var_sol/math.sqrt(math.pi) - mean_sol**2

  return mean_sol, var_sol


print(cubature_approx_mixture(relu, torch.tensor([[1, 1], [-1, 1], [1, 1], [-1, 1], [0, 10]]), torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])))