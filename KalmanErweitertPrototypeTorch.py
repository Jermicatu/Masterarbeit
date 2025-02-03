import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu

def f(x):
  return 2 * (sig(x+1))

def id(x):
  return x

def generateData(f, a, b, points):
  """
  Input:  @param f: function f(x) TODO multiple dimensions: for now one dimensional 
          @param a: creates the intervall with b 
          @param b: creates the intervall with a
          @param points: is the amount of equidistant points on the intervall [a, b]
          @output x: the tensor of the x values used to sample y
          @output y: the y values we get through f(x) + pertubation
  """
  x = torch.linspace(a, b, points, dtype=torch.float64)
  y = f(x) + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*0.01)
  return x, y

def staticPerceptron(m_w, W_variance, f, X):
  """
  Input:  @param m_w: tensor, with dtype=torch.float64, is the mean of the weights (plus bias) as a vector
          @param W_variance: tensor, with dtype=torch.float64, is the covariance matrix of the weights (plus bias)
          @param f: is the activation function of the perceptron
          @param X: torch vecotor of type torch.float64, is the input for the perceptron
  Output: @output sol: tensor, the static output of the perceptron
  """
  X_bias = torch.cat((torch.tensor([1]), X), 0)
  X_bias = torch.tensor(X_bias.clone().detach(), dtype=torch.float64)
  W = torch.distributions.multivariate_normal.MultivariateNormal(m_w, W_variance).sample()
  sol = f(torch.matmul(W , X_bias))
  return sol


def forwardPass(network, x):
  """
  Input:  @param network: 
          @param Data:
  Output: @forwardPassData
  """
  forwardPassData = [None] * len(network)

  # TODO check if bias is missing
  m_z_old = x
  s_z_old = torch.zeros(m_z_old.size(0), m_z_old.size(0), dtype=torch.float64)

  # iterate through each layer
  for i in range(0, len(network)):
    m_a = torch.zeros(len(network[i]), dtype=torch.float64)
    s_a = torch.zeros(len(network[i]), len(network[i]), dtype=torch.float64)
    m_z_new = torch.zeros(len(network[i]), dtype=torch.float64)
    s_z_new = torch.zeros(len(network[i]), len(network[i]), dtype=torch.float64)
    
    # iterate through each perceptron
    for j in range(0, len(network[i])):
      m_w, s_w, f = network[i][j]
      m_a[j] = torch.matmul(m_w, m_z_old)
      s_a[j][j] = torch.matmul(torch.matmul(m_w, s_z_old), m_w) + torch.matmul(torch.matmul(m_z_old, s_w), m_z_old) + torch.trace(torch.matmul(s_w, s_z_old))
      if f == sig:
        a_var = s_a[j][j]
        a_mean = m_a[j]
        l = math.sqrt(math.pi/8)
        t = math.sqrt(1 + (l**2) * a_var)
        m_z_new[j] = sig(a_mean / t)
        s_z_new[j][j] = max(m_z_new[j] * (1 - m_z_new[j]) * (1 - 1/t) + 10**(-6), torch.tensor([10**(-6)], dtype=torch.float64))
        #s_z_new[j][j] = m_z_new[j] * (1 - m_z_new[j]) * (1 - 1/t)
      if f == id:
        m_z_new = m_a
        s_z_new = s_a
      if f == tanh:
        a_var = s_a[j][j]
        a_mean = m_a[j]
        l = math.sqrt(math.pi/8)
        t = math.sqrt(1 + (l**2) * a_var)
        m_z_new[j] = 2 * sig(2 * a_mean / t) - 1
        s_z_new[j][j] = max((1 / t) * (m_z_new[j] + 1) * m_z_new[j] + m_z_new[j] * (1 - m_z_new[j])  + 10**(-6), torch.tensor([10**(-6)], dtype=torch.float64))
        # TODO

    m_z_old = torch.cat((torch.tensor([1], dtype=torch.float64), m_z_new), 0)
    s_z_old = torch.block_diag(torch.tensor([[0]], dtype=torch.float64), s_z_new) # variance of bias is zero
    forwardPassData[i] = [m_a, s_a, m_z_new, s_z_new]
    print("m_a: ", m_a)
    print("m_z_new: ", m_z_new)

    #m_z_old = torch.cat((torch.tensor([1], dtype=torch.float64), m_z_new), 0)
    #s_z_old = torch.block_diag(torch.tensor([[0]], dtype=torch.float64), s_z_new)
    #if i ==len(network)-1:
    #  forwardPassData[i] = [m_a, s_a, m_z_new, s_z_new]
    #else:
    #  forwardPassData[i] = [m_a, s_a, m_z_old, s_z_old]

  return forwardPassData

def backwardPass(network, data):
  # TODO
  """
  Input:  @param network: 
          @param Data:
  Output: @network
  """
  x_data, y_data = data
  # iterate through each data point
  for k in range(0, len(x_data)):
    x = x_data[k].unsqueeze(0)
    y = y_data[k].unsqueeze(0)
    x = torch.cat((torch.tensor([1], dtype=torch.float64), x.clone().detach()), 0)
    # forward pass
    forwardPassData = forwardPass(network, x)

    m_z_plus = y
    s_z_plus = torch.zeros(y.size(0), y.size(0), dtype=torch.float64)
    # iterate through each layer
    for i in range(len(network)-1, -1, -1):
      perc_count = len(network[i])
      
      if i == 0:
        m_z_minus_prev = x
        s_z_minus_prev = torch.zeros(x.size(0), x.size(0), dtype=torch.float64)
      else:
        m_z_minus_prev = torch.cat((torch.tensor([1], dtype=torch.float64), forwardPassData[i-1][2]), 0)
        s_z_minus_prev = torch.block_diag(torch.tensor([[0]], dtype=torch.float64), forwardPassData[i-1][3])

      m_a_minus, s_a_minus, m_z_minus, s_z_minus = forwardPassData[i]

      m_a_plus = torch.zeros(perc_count, dtype=torch.float64)
      s_a_plus = torch.zeros(perc_count, perc_count, dtype=torch.float64)
      # iterate through each perceptron

      for j in range(0, perc_count):
        m_w, s_w, f = network[i][j]

        cov_a_z = torch.zeros(perc_count, dtype=torch.float64)
        for n in range(0, m_z_plus.size(0)):
          if f == sig:
            l = math.sqrt(math.pi/8)
            t = math.sqrt(1 + (l**2) * s_a_minus[n][n])
            cov_a_z[n] = ((l * s_a_minus[n][n]) / t) * (1 / math.sqrt(2 * math.pi)) * math.exp(-((l * m_a_minus[n].item() / t)**2) / 2)
          elif f == id:
            cov_a_z[n] = 1
          if f == tanh:
            None # TODO

        print((i, j))
        k_n = torch.matmul(torch.inverse(s_z_minus), cov_a_z).unsqueeze(0).mT
        m_a_plus[j] = m_a_minus[j] + torch.matmul(k_n.mT, (m_z_plus - m_z_minus)).squeeze(0)
        s_a_plus[j][j] = max(s_a_minus[j][j] + torch.matmul(torch.matmul(k_n.mT, (s_z_plus - s_z_minus)), k_n) + 10**(-6), 10**(-6))
        print("cov_a_z: ", cov_a_z)
        print("k_n: ", k_n)
        print("s_a_plus: ", s_a_plus)
        print("s_z_minus: ", s_z_minus)
        
        if j==0:
          C_wza_top = torch.matmul(s_w, m_z_minus_prev.unsqueeze(0).mT)
          C_wza_bot = torch.matmul(s_z_minus_prev, m_w.unsqueeze(0).mT)
          m_w_big = m_w
          s_w_big = s_w
        else:
          print(C_wza_bot)
          print(torch.matmul(s_z_minus_prev, m_w.unsqueeze(0).mT))
          C_wza_top = torch.block_diag(C_wza_top, torch.matmul(s_w, m_z_minus_prev.unsqueeze(0).mT))
          C_wza_bot = torch.cat((C_wza_bot, torch.matmul(s_z_minus_prev, m_w.unsqueeze(0).mT)),1)
          m_w_big = torch.cat((m_w_big, m_w), 0)
          s_w_big = torch.block_diag(s_w_big, s_w)
      
      C_wza = torch.cat((C_wza_top, C_wza_bot), 0)
      L = torch.matmul(C_wza, torch.inverse(s_a_minus))
      
      print(torch.cat((m_w_big, m_z_minus_prev), 0))
      print(torch.matmul(L, (m_a_plus - m_a_minus)))

      m_big = torch.cat((m_w_big, m_z_minus_prev), 0) + torch.matmul(L, (m_a_plus - m_a_minus))
      c_big = torch.block_diag(s_w_big, s_z_minus_prev) + torch.matmul(torch.matmul(L, (s_a_plus - s_a_minus)), L.mT)

      m_w_plus, m_z_plus = torch.split(m_big, [len(m_w) * perc_count, len(m_big) - len(m_w) * perc_count])
      m_z_plus = torch.split(m_z_plus, [1, len(m_z_plus)-1])[1]
      c_w_plus = torch.split(torch.split(c_big, [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count])[0], [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count], 1)[0]
      s_z_plus = torch.split(torch.split(c_big, [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count])[1], [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count], 1)[1]
      s_z_plus = torch.split(torch.split(s_z_plus, [1, s_z_plus.size(0)-1])[1], [1, s_z_plus.size(0)-1], 1)[1]
      

      for j in range(0, perc_count):
        network[i][j][0], m_w_plus = torch.split(m_w_plus, [len(m_w), len(m_w_plus) - len(m_w)])
        network[i][j][1] = torch.split(torch.split(c_w_plus, [len(m_w), c_w_plus.size(0) - len(m_w)])[0], [len(m_w), c_w_plus.size(0) - len(m_w)], 1)[0] # + 10**(-6) * torch.eye(len(m_w), dtype=torch.float64)
        c_w_plus = torch.split(torch.split(c_w_plus, [c_w_plus.size(0)-network[i][j][1].size(0), network[i][j][1].size(0)])[1], [c_w_plus.size(0)-network[i][j][1].size(0), network[i][j][1].size(0)], 1)[1]
        #print("network[i][j][0]: ", network[i][j][0])
        #print("network[i][j][1]: ", network[i][j][1])
  return network

def createNetwork(input_size, network_size, functions):
  """
  Input:  @param Size: An np.array that tells the width of each layer
                       example: [3, 5, 3] an network where the first layer has width 3, 2nd 5 and 3rd 3.
          @param functions: functions for all layers in the network
                            example: np.array([sig, sig, id])
  Output: @output network: The Neural network where network[i][j] has the parameters needed for the perceptron at the ith layer and jth width
                           The parameters are of the form [m_w, W_variance, f]
  """
  network = [None] * len(network_size)

  network[0] = [None] * (network_size[0])
  for j in range(0, network_size[0]):
    # +1 for the bias
    network[0][j] = [torch.rand(input_size+1) - 0.5*torch.ones(input_size+1, dtype=torch.float64), torch.eye(input_size+1, dtype=torch.float64), functions[0]]

  for i in range(1,len(network_size)):
    network[i] = [None] * (network_size[i])
    for j in range(0, network_size[i]):
      network[i][j] = [torch.rand(network_size[i-1]+1) - 0.5*torch.ones(network_size[i-1]+1, dtype=torch.float64), torch.eye(network_size[i-1]+1, dtype=torch.float64), functions[i]]
  
  return network

def staticNetworkOutput(network, x):
  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
  if len(x.size()) == 0:
    x = x.unsqueeze(0)
  x = torch.tensor(x.clone().detach(), dtype=torch.float64)
  # iterate through each layer
  for i in range(0, len(network)):
    # iterate through each perceptron
    y = torch.zeros(len(network[i]))
    for j in range(0, len(network[i])):
      m_w, W_variance, f = network[i][j]
      y[j] = staticPerceptron(m_w, W_variance, f, x)
    x = y
  return y

data = generateData(sig, -3, 3, 100)
network = createNetwork(1, [1], [id])
print("network: ", network)
network = backwardPass(network, data)
print("network: ", network)
W_variance = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)

perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
for i in range(0, data[0].size(0)):
  perceptron_Plot[i] = staticNetworkOutput(network , data[0][i])

plt.plot(data[0], data[1])
plt.plot(data[0], perceptron_Plot)
plt.show()