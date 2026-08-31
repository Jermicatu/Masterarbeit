import torch
import math
import random
from .utils import id

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu
from torch import erf as erf
from torch import exp as exp

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

class Network_Class:
  def __init__(self, input_size, dimensions, functions, starting_variance = 1):
    """
      Input:  @param Size: An np.array that tells the width of each layer
                           example: [3, 5, 3] an network where the first layer has width 3, 2nd 5 and 3rd 3.
              @param functions: functions for all layers in the network
                                example: [sig, sig, id]
      Saves a network where each layer saves the weights and the variance matrix as a batch plus saviang the activation function.
      The form is [weights, variance, activation function]
    """
    network = [None] * len(dimensions)

    # +1 for the bias

    network[0] = [torch.normal(0, 1, size = (input_size+1, dimensions[0]), dtype=torch.float32), starting_variance * torch.diag_embed(torch.ones(dimensions[0], input_size+1, dtype=torch.float32)), functions[0]]

    for i in range(1,len(dimensions)):
      network[i] = [torch.normal(0, 1, size = (dimensions[i-1]+1, dimensions[i]), dtype=torch.float32), starting_variance * torch.diag_embed(torch.ones(dimensions[i],dimensions[i-1]+1, dtype=torch.float32)), functions[i]]
      

    self.input_size = input_size
    self.dimensions = dimensions
    self.functions = functions
    self.noise = 0.01
    self.network = network

  def meanOutput(self, x):
    """
    Input:  @param x: vector, input for the network
    Output: @output y: vector, the mean output from our network
    """
    network = self.network
    dimensions = self.dimensions

    if len(x.size()) == 0:
      x = x.unsqueeze(0)
    x = x.to(torch.float32)

    # iterate through each layer 
    for i in range(0, len(dimensions)):
      f = network[i][2]
      x = f(torch.flatten(torch.matmul(torch.cat((x, torch.ones(1)), 0), network[i][0])))
    return x

  def staticOutput(self, x):
    """
    Input:  @param x: vector, input for the network
    Output: @output y: vector, the approximation from our network
    """
    network = self.network
    dimensions = self.dimensions

    if len(x.size()) == 0:
      x = x.unsqueeze(0)
    x = x.to(torch.float32)

    # iterate through each layer
    for i in range(0, len(dimensions)):
      f = network[i][2]

      m = torch.distributions.multivariate_normal.MultivariateNormal(network[i][0].mT, network[i][1])
      x = f(torch.matmul(torch.cat((x, torch.ones(1)), 0), m.sample().mT))

    return x
  
  def forwardPass(self, x):
    """
    Input:  @param x: vector, input for the network
    Output: @forwardPassData: the variables [m_a, s_a, m_z_new, s_z_new] for each layer to then be used in the train function
    """
    network = self.network
    dimensions = self.dimensions
    input_size = self.input_size
    forwardPassData = [None] * len(dimensions)

    m_z_old = x
    s_z_old = torch.zeros(input_size+1, input_size+1, dtype=torch.float32)

    # iterate through each layer
    for i in range(0, len(dimensions)):
      m_w, s_w, f = network[i]

      A = torch.einsum('in,ij,jn->n', m_w, s_z_old, m_w)
      B = torch.einsum('i,nij,j->n', m_z_old, s_w, m_z_old)
      C = torch.einsum('nij,jk->nik', s_w, s_z_old)
      C = torch.einsum('nii->n', C)
      
      m_a = torch.flatten(torch.matmul(m_z_old, m_w))
      s_a = torch.maximum(A + B + C, torch.ones(dimensions[i], dtype=torch.float32) * self.noise)

      if f == torch.sigmoid:
        l = math.sqrt(math.pi/8)
        t = torch.sqrt(1 + (l**2) * s_a).pow_(-1)
        m_z_new = sig(m_a * t)
        s_z_new = torch.maximum(m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - t), torch.ones(dimensions[i], dtype=torch.float32) * self.noise)
      if f == id:
        m_z_new = m_a
        s_z_new = s_a + self.noise
      if f == torch.relu:
        alpha = m_a / torch.sqrt(s_a)
        p_a = torch.sqrt(s_a)/math.sqrt(2 * math.pi) * exp(-0.5 * alpha**2)
        probit = 0.5 * (1 + erf(alpha / math.sqrt(2)))
        m_z_new = m_a * probit + p_a
        s_z_new = torch.maximum((torch.pow(m_a, 2) + s_a) * probit + m_a * p_a - m_z_new ** 2 + torch.ones(dimensions[i], dtype=torch.float32) * self.noise, torch.ones(dimensions[i], dtype=torch.float32) * self.noise)

      # simple cubature approx
      #m_z_new, s_z_new = cubature_approx(f, m_a, s_a)
      #s_z_new = torch.maximum(s_z_new + torch.ones(dimensions[i], dtype=torch.float32) * self.noise, torch.ones(dimensions[i], dtype=torch.float32) * self.noise)
      
      m_z_old = torch.cat((m_z_new, torch.tensor([1], dtype=torch.float32)), 0)
      s_z_old = torch.block_diag(torch.diag(s_z_new), torch.tensor([[0]], dtype=torch.float32))

      # print([m_a, s_a, m_z_new, s_z_new])
      forwardPassData[i] = [m_a, s_a, m_z_new, s_z_new]

    
    return forwardPassData

  def train(self, data):
    """
    Input:  @param data: The data to train the network with in form of [x_data, y_data]
    """
    x_data, y_data = data
    network = self.network
    dimensions = self.dimensions
    
    # iterate through each data point
    indices = list(range(len(x_data)))
    random.shuffle(indices)
    for j in indices:
    #for j in range(0, len(x_data)):
      x = x_data[j].unsqueeze(0)
      y = y_data[j].unsqueeze(0)

      x_ = torch.cat((x, torch.ones(1, dtype=torch.float32)), 0)
      # forward pass
      forwardPassData = self.forwardPass(x_)

      m_z_plus = y
      s_z_plus = torch.zeros(y.size(0), dtype=torch.float32)

      # iterate through each layer
      for i in range(len(dimensions)-1, -1, -1):
        perc_count = dimensions[i]
        
        # values differt in the very first layer
        if i == 0:
          m_z_minus_prev = x
          s_z_minus_prev = torch.zeros(x.size(0), dtype=torch.float32)
          w_count = self.input_size + 1
        else:
          m_z_minus_prev = forwardPassData[i-1][2]
          s_z_minus_prev = forwardPassData[i-1][3]
          w_count = dimensions[i-1] + 1

        assert m_z_plus.flatten().shape == torch.Size([perc_count]), f"Shape mismatch in m_z_plus"
        assert s_z_plus.shape == torch.Size([perc_count]), f"Shape mismatch in s_z_plus"
        assert m_z_minus_prev.shape == torch.Size([w_count-1]), f"Shape mismatch in m_z_minus_prev"
        assert s_z_minus_prev.shape == torch.Size([w_count-1]), f"Shape mismatch in s_z_minus_prev"

        m_a_minus, s_a_minus, m_z_minus, s_z_minus = forwardPassData[i]
        assert m_a_minus.shape == torch.Size([perc_count]), f"Shape mismatch in m_a_minus"
        assert s_a_minus.shape == torch.Size([perc_count]), f"Shape mismatch in s_a_minus"
        assert m_z_minus.shape == torch.Size([perc_count]), f"Shape mismatch in m_z_minus"
        assert s_z_minus.shape == torch.Size([perc_count]), f"Shape mismatch in s_z_minus"
        m_w, s_w, f = network[i]
        
        if f == torch.sigmoid:
          l = math.sqrt(math.pi/8)
          t = torch.sqrt(1 + (l**2) * s_a_minus)
          cov_a_z = ((l * s_a_minus) / t) * (1 / math.sqrt(2 * math.pi)) * torch.exp(-((l * m_a_minus / t)**2) / 2)
        elif f == id:
          cov_a_z = m_a_minus ** 2 + s_a_minus - m_z_minus * m_a_minus
        elif f == torch.relu:
          alpha = m_a_minus / torch.sqrt(s_a_minus)
          p_a = torch.sqrt(s_a_minus)/math.sqrt(2 * math.pi) * exp(-0.5 * alpha**2)
          probit = 0.5 * (1 + erf(alpha / math.sqrt(2)))
          cov_a_z = torch.maximum((torch.pow(m_a_minus, 2) + s_a_minus) * probit + m_a_minus * p_a - m_a_minus * m_z_minus, torch.ones(dimensions[i], dtype=torch.float32) * 1e-12)
        
        assert cov_a_z.shape == torch.Size([perc_count]), f"Shape mismatch in cov_a_z"

        k = cov_a_z / s_z_minus
        
        da = k * (m_z_plus.flatten() - m_z_minus)
        Da = (k **2) * (s_z_plus - s_z_minus)
        
        C_wza_top = s_w @ torch.cat((m_z_minus_prev, torch.tensor([1], dtype=torch.float32)), 0)
        C_wza_bot = torch.cat((s_z_minus_prev, torch.tensor([0], dtype=torch.float32)), 0).unsqueeze(-1) * m_w
                
        Ca_inv = 1 / s_a_minus
        
        L_up = C_wza_top * torch.outer(Ca_inv, torch.ones((w_count)))
        L_low = C_wza_bot * Ca_inv.unsqueeze(0).repeat(w_count, 1)
        
        self.network[i][0] = self.network[i][0] + torch.t(L_up * torch.outer(da, torch.ones((w_count))))
        m_z_plus = m_z_minus_prev + L_low[:-1] @ da
        
        E = L_up * torch.outer(Da, torch.ones((w_count), dtype=torch.float32))
        F = L_up.unsqueeze(-1) @ torch.ones((1, w_count), dtype=torch.float32)
        
        self.network[i][1] = self.network[i][1] + E.unsqueeze(1).repeat(1, w_count, 1) * F
        
        G = L_low ** 2 * Da.unsqueeze(0).repeat(w_count, 1)
        
        s_z_plus = s_z_minus_prev + G[:-1] @ torch.ones((perc_count), dtype=torch.float32)

        
        

    self.network = network
