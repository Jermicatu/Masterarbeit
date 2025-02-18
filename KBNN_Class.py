import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from torch import tanh as tanh
from torch import cos as cos
from torch import sigmoid as sig
from torch import relu as relu

def f1(x):
  return 0.45 * (cos(x+1) + 1)

def f2(x):
  return 0.5 * (sig(x+1) - sig(x-1) + sig(x))

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

class Network_Class:
  def __init__(self, input_size, dimensions, functions):
    
    network = [None] * len(dimensions)

    # +1 for the bias
    network[0] = [torch.rand(dimensions[0], input_size+1, 1) - 0.5*torch.ones(dimensions[0], input_size+1, 1, dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[0], input_size+1, dtype=torch.float64)), functions[0]]

    for i in range(1,len(dimensions)):
      network[i] = [torch.rand(dimensions[i], dimensions[i-1]+1, 1) - 0.5*torch.ones(dimensions[i], dimensions[i-1]+1, 1, dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[i],dimensions[i-1]+1, dtype=torch.float64)), functions[i]]
      

    self.input_size = input_size
    self.dimensions = dimensions
    self.functions = functions
    self.network = network

  def staticOutput(self, x):
    network = self.network
    dimensions = self.dimensions
    input_size = self.input_size

    if len(x.size()) == 0:
      x = x.unsqueeze(0)
    x = x.to(torch.float64)
    # iterate through each layer
    x = torch.cat((torch.ones(1), x), 0)
    x = x.repeat(dimensions[0], 1, 1)
    w_core = torch.normal(0, 1, size=(dimensions[0], input_size+1, 1), dtype=torch.float64)
    x = torch.bmm(x, torch.bmm(network[0][1], w_core) + network[0][0])
    x = torch.flatten(x)
    
    for i in range(1, len(dimensions)):
      x = torch.cat((torch.ones(1), x), 0)
      x = x.repeat(dimensions[i], 1, 1)
      w_core = torch.normal(0, 1, size=(dimensions[i], dimensions[i-1]+1, 1), dtype=torch.float64)
      x = torch.bmm(x, torch.bmm(network[i][1], w_core) + network[i][0])
      x = torch.flatten(x)
    return x
  
  def __forwardPass(self, x):
    network = self.network
    dimensions = self.dimensions
    input_size = self.input_size
    forwardPassData = [None] * len(dimensions)
    m_z_old = x
    s_z_old = torch.zeros(input_size+1, input_size+1, dtype=torch.float64)


    # iterate through each layer
    for i in range(0, len(dimensions)):
      m_w, s_w, f = network[i]

      m_a = torch.flatten(torch.matmul(m_z_old, m_w))
      s_a = torch.flatten(torch.matmul(torch.transpose(m_w, 1, 2), torch.matmul(s_z_old, m_w))) + torch.matmul(m_z_old, torch.matmul(s_w, m_z_old).mT) + torch.vmap(torch.trace)(torch.matmul(s_w, s_z_old))
      
      if f == sig:
        l = math.sqrt(math.pi/8)
        t = torch.sqrt(1 + (l**2) * s_a).pow_(-1)
        m_z_new = sig(m_a * t)
        s_z_new = torch.maximum(m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - torch.ones(dimensions[i]) * t) + torch.ones(dimensions[i]) * 10**(-6), torch.ones(dimensions[i], dtype=torch.float64) * 10**(-6))
      if f == id:
        m_z_new = m_a
        s_z_new = s_a

      m_z_old = torch.cat((torch.tensor([1], dtype=torch.float64), m_z_new), 0)
      s_z_old = torch.block_diag(torch.tensor([[0]], dtype=torch.float64), torch.diag(s_z_new)) # variance of bias is zero

      forwardPassData[i] = [m_a, s_a, m_z_new, s_z_new]
    
    return forwardPassData

  def train(self, data):
    x_data, y_data = data
    network = self.network
    dimensions = self.dimensions

    # iterate through each data point
    for k in range(0, len(x_data)):
      x = x_data[k].unsqueeze(0)
      y = y_data[k].unsqueeze(0)
      x = torch.cat((torch.ones(1, dtype=torch.float64), x), 0)
      # forward pass
      forwardPassData = self.__forwardPass(x)

      m_z_plus = y
      s_z_plus = torch.zeros(y.size(0), dtype=torch.float64)
      # iterate through each layer
      for i in range(len(dimensions)-1, -1, -1):
        perc_count = dimensions[i]
          
        if i == 0:
          m_z_minus_prev = x
          s_z_minus_prev = torch.zeros(x.size(0), dtype=torch.float64)
        else:
          m_z_minus_prev = torch.cat((torch.tensor([1], dtype=torch.float64), forwardPassData[i-1][2]), 0)
          s_z_minus_prev = torch.cat((torch.tensor([0], dtype=torch.float64), forwardPassData[i-1][3]), 0)

        m_a_minus, s_a_minus, m_z_minus, s_z_minus = forwardPassData[i]

        m_a_plus = torch.zeros(perc_count, dtype=torch.float64)
        s_a_plus = torch.zeros(perc_count, dtype=torch.float64)

        m_w, s_w, f = network[i]
        
        # Has to be changed if the activation function can vary in a layer
        if f == sig:
          l = math.sqrt(math.pi/8)
          t = torch.sqrt(1 + (l**2) * s_a_minus)
          cov_a_z = ((l * s_a_minus) / t) * (1 / math.sqrt(2 * math.pi)) * torch.exp(-((l * m_a_minus / t)**2) / 2)
        elif f == id:
          cov_a_z = s_a_minus

        k_n = 0.9 * (torch.pow(s_z_minus, -1) * cov_a_z)
        m_a_plus = m_a_minus + k_n * (m_z_plus - m_z_minus)
        s_a_plus = torch.maximum(s_a_minus + torch.matmul(torch.matmul(k_n, (s_z_plus - torch.diag(s_z_minus))), k_n) + torch.ones(s_a_minus.size(0)) * 10**(-6), torch.ones(s_a_minus.size(0)) * 10**(-6))
        
        C_wza_top = torch.matmul(s_w, m_z_minus_prev)
        C_wza_bot = torch.matmul(s_z_minus_prev, m_w)
        print(s_w)
        print(m_z_minus_prev)
        print(C_wza_top)
        """
        if j==0:
          C_wza_top = torch.matmul(s_w, m_z_minus_prev.unsqueeze(0).mT)
          C_wza_bot = torch.matmul(s_z_minus_prev, m_w)          
          m_w_big = m_w
          s_w_big = s_w
        else:
          C_wza_top = torch.block_diag(C_wza_top, torch.matmul(s_w, m_z_minus_prev.unsqueeze(0).mT))
          C_wza_bot = torch.cat((C_wza_bot, torch.matmul(s_z_minus_prev, m_w.unsqueeze(0).mT)),1)
          m_w_big = torch.cat((m_w_big, m_w), 0)
          s_w_big = torch.block_diag(s_w_big, s_w)
        """ 
        C_wza = torch.cat((C_wza_top, C_wza_bot), 0)
        L = torch.matmul(C_wza, torch.inverse(s_a_minus))

        m_big = torch.cat((m_w_big, m_z_minus_prev), 0) + torch.matmul(L, (m_a_plus - m_a_minus))
        c_big = torch.block_diag(s_w_big, s_z_minus_prev) + torch.matmul(torch.matmul(L, torch.diag(s_a_plus - s_a_minus)), L.mT)

        m_w_plus, m_z_plus = torch.split(m_big, [len(m_w) * perc_count, len(m_big) - len(m_w) * perc_count])
        m_z_plus = torch.split(m_z_plus, [1, len(m_z_plus)-1])[1]
        c_w_plus = torch.split(torch.split(c_big, [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count])[0], [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count], 1)[0]
        s_z_plus = torch.split(torch.split(c_big, [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count])[1], [len(m_w) * perc_count, len(c_big) - len(m_w) * perc_count], 1)[1]
        s_z_plus = torch.split(torch.split(s_z_plus, [1, s_z_plus.size(0)-1])[1], [1, s_z_plus.size(0)-1], 1)[1]
      

    for j in range(0, perc_count):
      network[i][j][0], m_w_plus = torch.split(m_w_plus, [len(m_w), len(m_w_plus) - len(m_w)])
      network[i][j][1] = torch.split(torch.split(c_w_plus, [len(m_w), c_w_plus.size(0) - len(m_w)])[0], [len(m_w), c_w_plus.size(0) - len(m_w)], 1)[0] # + 10**(-6) * torch.eye(len(m_w), dtype=torch.float64)
      c_w_plus = torch.split(torch.split(c_w_plus, [c_w_plus.size(0)-network[i][j][1].size(0), network[i][j][1].size(0)])[1], [c_w_plus.size(0)-network[i][j][1].size(0), network[i][j][1].size(0)], 1)[1]


torch.manual_seed(10)
data = generateData(f2, -5, 5, 1000)
network = Network_Class(1, [5, 1], [sig, id])
print("network: ", network)
plt.plot(data[0], data[1], label = "f")

for k in range(1, 11):
  network.train(data)
  perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
  mse = 0
  # print("network ", k, " : ", network)
  for i in range(0, data[0].size(0)):
    perceptron_Plot[i] = network.staticOutput(network , data[0][i])
    mse += (data[1][i] - perceptron_Plot[i])**2
  mse = mse/data[0].size(0)
  print("mse: ", mse)
  plt.plot(data[0], perceptron_Plot, label='approx Nr. {k}'.format(k=k))

myPrediction = torch.zeros(data[0].size(0), dtype=torch.float64)
for i in range(0, data[0].size(0)):
  myPrediction[i] = f2(data[0][i])
plt.plot(data[0], myPrediction, label='My Prediction')

print("network: ", network)
plt.legend(loc='best')
plt.ylim(-0.1, 0.6)
plt.show()