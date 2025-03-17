#import torch
#import numpy as np
#import math
#import matplotlib.pyplot as plt

#from torch import tanh as tanh
#from torch import cos as cos
#from torch import sigmoid as sig
#from torch import relu as relu

import torch
import math
import matplotlib.pyplot as plt
import time
import numpy as np

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
  y = f(x) + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*0.1)
  return x, y

class Network_Class:
  def __init__(self, input_size, dimensions, functions):
    """
      Input:  @param Size: An np.array that tells the width of each layer
                           example: [3, 5, 3] an network where the first layer has width 3, 2nd 5 and 3rd 3.
              @param functions: functions for all layers in the network
                                example: [sig, sig, id]
    """
    network = [None] * len(dimensions)

    # +1 for the bias
    #  - 0.5*torch.ones(dimensions[0], input_size+1, 1, dtype=torch.float64)

    network[0] = [torch.normal(0, 1, size = (dimensions[0], input_size, 1), dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[0], input_size, dtype=torch.float64)), functions[0]]

    for i in range(1,len(dimensions)):
      network[i] = [torch.normal(0, 1, size = (dimensions[i], dimensions[i-1], 1), dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[i],dimensions[i-1], dtype=torch.float64)), functions[i]]
      

    self.input_size = input_size
    self.dimensions = dimensions
    self.functions = functions
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
    x = x.to(torch.float64)

    # iterate through each layer 
    for i in range(0, len(dimensions)):
      f = network[i][2]
      x = f(torch.flatten(torch.matmul(x, network[i][0])))
    return x
    

  def staticOutput(self, x):
    """
    Input:  @param x: vector, input for the network
    Output: @output y: vector, the approximation from our network
    """
    network = self.network
    dimensions = self.dimensions
    input_size = self.input_size

    if len(x.size()) == 0:
      x = x.unsqueeze(0)
    x = x.to(torch.float64)

    # iterate through each layer
    for i in range(0, len(dimensions)):
      f = network[i][2]
      w_core = torch.normal(0, 1, size=(dimensions[i], dimensions[i-1], 1), dtype=torch.float64)
      x = f(torch.flatten(torch.matmul(x, torch.bmm(network[i][1], w_core) + network[i][0])))
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
    s_z_old = torch.zeros(input_size, input_size, dtype=torch.float64)
    # s_z_old = torch.eye(input_size+1, input_size+1, dtype=torch.float64) * 1e-3

    # iterate through each layer
    for i in range(0, len(dimensions)):
      m_w, s_w, f = network[i]

      m_a = torch.flatten(torch.matmul(m_z_old, m_w))
      s_a = torch.flatten(torch.matmul(torch.transpose(m_w, 1, 2), torch.matmul(s_z_old, m_w))) + torch.matmul(m_z_old, torch.matmul(s_w, m_z_old).mT) + torch.vmap(torch.trace)(torch.matmul(s_w, s_z_old))
      if f == sig:
        l = math.sqrt(math.pi/8)
        t = torch.sqrt(1 + (l**2) * s_a).pow_(-1)
        m_z_new = sig(m_a * t)
        # s_z_new = m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - t)
        s_z_new = torch.maximum(m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - t) + torch.ones(dimensions[i]) * 10**(-6), torch.ones(dimensions[i], dtype=torch.float64) * 10**(-6))
      if f == id:
        m_z_new = m_a
        s_z_new = s_a
      if f == relu:
        p_a = torch.pow(s_a, 0.5)/math.sqrt(2 * math.pi) * exp(-0.5 * torch.pow(m_a, 2) * torch.pow(s_a, -1))
        probit = 0.5 * (1 + erf(m_a * torch.pow(s_a, -1/2) / math.sqrt(2)))
        m_z_new = m_a * probit + p_a
        s_z_new = torch.maximum((torch.pow(m_a, 2) + torch.pow(s_a, 2)) * probit + m_a * p_a - torch.pow(m_z_new, 2) + torch.ones(dimensions[i]) * 10**(-6), torch.ones(dimensions[i], dtype=torch.float64) * 10**(-6))

      m_z_old = m_z_new
      s_z_old = torch.diag(s_z_new)

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
    for k in range(0, len(x_data)):
    #for k in range(len(x_data)-1, -1, -1):
      x = x_data[k].unsqueeze(0)
      y = y_data[k].unsqueeze(0)
      # forward pass
      forwardPassData = self.forwardPass(x)

      m_z_plus = y
      s_z_plus = torch.zeros(y.size(0), y.size(0), dtype=torch.float64)
      #s_z_plus = torch.eye(y.size(0), dtype=torch.float64) * 1e-2

      # iterate through each layer
      for i in range(len(dimensions)-1, -1, -1):
        perc_count = dimensions[i]
        
        # values differt in the very first layer
        if i == 0:
          m_z_minus_prev = x
          s_z_minus_prev = torch.ones(x.size(0), dtype=torch.float64) * 1e-6
          s_z_minus_prev = torch.zeros(x.size(0), dtype=torch.float64)
          w_count = self.input_size
        else:
          m_z_minus_prev = forwardPassData[i-1][2]
          s_z_minus_prev = forwardPassData[i-1][3]
          w_count = dimensions[i-1]

        m_a_minus, s_a_minus, m_z_minus, s_z_minus = forwardPassData[i]

        m_w, s_w, f = network[i]
        
        if f == sig:
          l = math.sqrt(math.pi/8)
          t = torch.sqrt(1 + (l**2) * s_a_minus)
          cov_a_z = ((l * s_a_minus) / t) * (1 / math.sqrt(2 * math.pi)) * torch.exp(-((l * m_a_minus / t)**2) / 2)
        elif f == id:
          cov_a_z = s_a_minus
        elif f == relu:
          cov_a_z = s_z_minus + torch.pow(m_z_minus, 2) - m_a_minus * m_z_minus
        
        # (0.45 * (len(x_data) - k)/len(x_data) + 0.1)
        # 0.9
        K = torch.cat([torch.pow(s_z_minus, -1).unsqueeze(0)*cov_a_z[i] for i in range(0, cov_a_z.size(0))])

        
        m_a_plus = m_a_minus + torch.matmul(K, (m_z_plus.flatten() - m_z_minus))
        # s_a_plus = s_a_minus + torch.einsum("ij,jk,ki->i", K, (s_z_plus - torch.diag(s_z_minus)), K.mT) 
        s_a_plus = torch.maximum(s_a_minus + torch.einsum("ij,jk,ki->i", K, (s_z_plus - torch.diag(s_z_minus)), K.mT) + torch.ones(s_a_minus.size(0)) * 10**(-6), torch.ones(s_a_minus.size(0)) * 10**(-6))

        C_wza_top = torch.block_diag(*[torch.matmul(s_w, m_z_minus_prev.unsqueeze(0).mT)[i] for i in range(0, perc_count)])
        C_wza_bot = (s_z_minus_prev * torch.transpose(m_w, 1, 2)).squeeze(1).mT
        C_wza = torch.cat((C_wza_top, C_wza_bot), 0)
        
        L = C_wza * torch.pow(s_a_minus, -1)

        m_w_big = m_w.flatten().unsqueeze(0).mT
        s_w_big = torch.block_diag(*[s_w[i] for i in range(0, perc_count)])

        m_big = torch.cat((m_w_big, m_z_minus_prev.unsqueeze(0).mT), 0) + torch.matmul(L, (m_a_plus - m_a_minus).unsqueeze(0).mT)
        c_big = torch.block_diag(s_w_big, torch.diag(s_z_minus_prev)) + torch.matmul(torch.matmul(L, torch.diag(s_a_plus - s_a_minus)), L.mT)

        m_w_plus, m_z_plus = torch.split(m_big, [w_count * perc_count, len(m_big) - w_count * perc_count])
        c_w_plus = torch.split(torch.split(c_big, [w_count * perc_count, len(c_big) - w_count * perc_count])[0], [w_count * perc_count, len(c_big) - w_count * perc_count], 1)[0]
        s_z_plus = torch.split(torch.split(c_big, [w_count * perc_count, len(c_big) - w_count * perc_count])[1], [w_count * perc_count, len(c_big) - w_count * perc_count], 1)[1]
      

        for j in range(0, perc_count):
          network[i][0][j], m_w_plus = torch.split(m_w_plus, [w_count, m_w_plus.size(0) - w_count])
          network[i][1][j] = torch.split(torch.split(c_w_plus, [w_count, c_w_plus.size(0) - w_count])[0], [w_count, c_w_plus.size(0) - w_count], 1)[0] # + 10**(-6) * torch.eye(len(m_w), dtype=torch.float64)
          c_w_plus = torch.split(torch.split(c_w_plus, [c_w_plus.size(0)-network[i][1][j].size(0), network[i][1][j].size(0)])[1], [c_w_plus.size(0)-network[i][1][j].size(0), network[i][1][j].size(0)], 1)[1]
    self.network = network

# Calculate the start time
start = time.time()

torch.manual_seed(111111)
data = generateData(f4, -1, 1, 100)
network = Network_Class(1, [10, 1], [relu, id])

print(network.network)

#print(network.network)
plt.plot(data[0], data[1], '.', label = "f")

n_plots = 2
color = iter(plt.cm.rainbow(np.linspace(0, 1, 100)))

for k in range(0, 100):
  network.train([torch.tensor([data[0][k]]), torch.tensor([data[1][k]])])
  perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
  perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)
  mse = 0
  # print("network ", k, " : ", network)
  for i in range(0, data[0].size(0)):
    perceptron_Plot[i] = network.meanOutput(data[0][i])
    perceptron_Plot_var[i] = network.forwardPass(data[0][i].unsqueeze(0))[len(network.dimensions)-1][3]
    mse += (data[1][i] - perceptron_Plot[i])**2
  mse = mse/data[0].size(0)
  print("mse Nr. ", k, " : ", mse)
  c = next(color)
  plt.plot(data[0], perceptron_Plot, label='epoch Nr. {k}'.format(k=k), color=c)
  plt.fill_between(data[0], perceptron_Plot-perceptron_Plot_var, perceptron_Plot + perceptron_Plot_var, alpha=0.2, color=c)
  

# Calculate the end time and time taken
end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")

print("network: ", network.network)

# plt.legend(loc='best')
plt.ylim(-1.5, 1.5)
plt.show()