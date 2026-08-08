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

def generateData(f, a, b, points):
  """
  Input:  @param f: function f(x) TODO multiple dimensions: for now one dimensional 
          @param a: creates the intervall with b 
          @param b: creates the intervall with a
          @param points: is the amount of equidistant points on the intervall [a, b]
  Output: @output x: the tensor of the x values used to sample y
          @output y: the y values we get through f(x) + pertubation
  """
  x = torch.linspace(a, b, points, dtype=torch.float64)
  y = f(x) + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)) * 8
  return x, y

class Network_Class:
  def __init__(self, input_size, dimensions, functions):
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

    network[0] = [torch.normal(0, 1, size = (input_size+1, dimensions[0]), dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[0], input_size+1, dtype=torch.float64)), functions[0]]

    for i in range(1,len(dimensions)):
      network[i] = [torch.normal(0, 1, size = (dimensions[i-1]+1, dimensions[i]), dtype=torch.float64), torch.diag_embed(torch.ones(dimensions[i],dimensions[i-1]+1, dtype=torch.float64)), functions[i]]
      

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
    x = x.to(torch.float64)

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
    x = x.to(torch.float64)

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
    s_z_old = torch.zeros(input_size+1, input_size+1, dtype=torch.float64)

    # iterate through each layer
    for i in range(0, len(dimensions)):
      m_w, s_w, f = network[i]

      A = torch.einsum('in,ij,jn->n', m_w, s_z_old, m_w)
      B = torch.einsum('i,nij,j->n', m_z_old, s_w, m_z_old)
      C = torch.einsum('nij,jk->nik', s_w, s_z_old)
      C = torch.einsum('nii->n', C)
      
      m_a = torch.flatten(torch.matmul(m_z_old, m_w))
      s_a = torch.maximum(A + B + C, torch.ones(dimensions[i], dtype=torch.float64) * self.noise)


      if f == sig:
        l = math.sqrt(math.pi/8)
        t = torch.sqrt(1 + (l**2) * s_a).pow_(-1)
        m_z_new = sig(m_a * t)
        s_z_new = torch.maximum(m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - t), torch.ones(dimensions[i], dtype=torch.float64) * self.noise)
      if f == id:
        m_z_new = m_a
        s_z_new = s_a + self.noise
      if f == relu:
        alpha = m_a / torch.sqrt(s_a)
        p_a = torch.sqrt(s_a)/math.sqrt(2 * math.pi) * exp(-0.5 * alpha**2)
        probit = 0.5 * (1 + erf(alpha / math.sqrt(2)))
        m_z_new = m_a * probit + p_a
        s_z_new = torch.maximum((torch.pow(m_a, 2) + s_a) * probit + m_a * p_a - m_z_new ** 2 + torch.ones(dimensions[i], dtype=torch.float64) * self.noise, torch.ones(dimensions[i], dtype=torch.float64) * self.noise)

      m_z_old = torch.cat((m_z_new, torch.tensor([1], dtype=torch.float64)), 0)
      s_z_old = torch.block_diag(torch.diag(s_z_new), torch.tensor([[0]], dtype=torch.float64))

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

      x_ = torch.cat((x, torch.ones(1, dtype=torch.float64)), 0)
      # forward pass
      forwardPassData = self.forwardPass(x_)

      m_z_plus = y
      s_z_plus = torch.zeros(y.size(0), dtype=torch.float64)

      # iterate through each layer
      for i in range(len(dimensions)-1, -1, -1):
        perc_count = dimensions[i]
        
        # values differt in the very first layer
        if i == 0:
          m_z_minus_prev = x
          s_z_minus_prev = torch.zeros(x.size(0), dtype=torch.float64)
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
        
        if f == sig:
          l = math.sqrt(math.pi/8)
          t = torch.sqrt(1 + (l**2) * s_a_minus)
          cov_a_z = ((l * s_a_minus) / t) * (1 / math.sqrt(2 * math.pi)) * torch.exp(-((l * m_a_minus / t)**2) / 2)
        elif f == id:
          cov_a_z = m_a_minus ** 2 + s_a_minus - m_z_minus * m_a_minus
        elif f == relu:
          alpha = m_a_minus / torch.sqrt(s_a_minus)
          p_a = torch.sqrt(s_a_minus)/math.sqrt(2 * math.pi) * exp(-0.5 * alpha**2)
          probit = 0.5 * (1 + erf(alpha / math.sqrt(2)))
          cov_a_z = torch.maximum((torch.pow(m_a_minus, 2) + s_a_minus) * probit + m_a_minus * p_a - m_a_minus * m_z_minus, torch.ones(dimensions[i], dtype=torch.float64) * 1e-12)
       
        assert cov_a_z.shape == torch.Size([perc_count]), f"Shape mismatch in cov_a_z"

        k = cov_a_z / s_z_minus
        
        da = k * (m_z_plus.flatten() - m_z_minus)
        Da = (k **2) * (s_z_plus - s_z_minus)
        
        C_wza_top = s_w @ torch.cat((m_z_minus_prev, torch.tensor([1], dtype=torch.float64)), 0)
        C_wza_bot = torch.cat((s_z_minus_prev, torch.tensor([0], dtype=torch.float64)), 0).unsqueeze(-1) * m_w
                
        Ca_inv = 1 / s_a_minus
        
        L_up = C_wza_top * torch.outer(Ca_inv, torch.ones((w_count)))
        L_low = C_wza_bot * Ca_inv.unsqueeze(0).repeat(w_count, 1)
        
        self.network[i][0] = self.network[i][0] + torch.t(L_up * torch.outer(da, torch.ones((w_count))))
        m_z_plus = m_z_minus_prev + L_low[:-1] @ da
        
        E = L_up * torch.outer(Da, torch.ones((w_count), dtype=torch.float64))
        F = L_up.unsqueeze(-1) @ torch.ones((1, w_count), dtype=torch.float64)
        
        self.network[i][1] = self.network[i][1] + E.unsqueeze(1).repeat(1, w_count, 1) * F
        
        G = L_low ** 2 * Da.unsqueeze(0).repeat(w_count, 1)
        
        s_z_plus = s_z_minus_prev + G[:-1] @ torch.ones((perc_count), dtype=torch.float64)

        
        

    self.network = network

def sample(network, data, times, place):
  """
  Input:  @param network: A netork class
          @param data: The data to train the network with in form of [x_data, y_data]
          @times: The amount of samples taken
          @place: Place is the number of the data point you want to sample for. 
                  For example: in a dataset with 800 points 0 is the first point and 799 would be the last.
  """
  x = data[0][place]
  m_a_minus, s_a_minus, m_z_minus, s_z_minus = network.forwardPass(torch.cat((x.unsqueeze(0), torch.ones(1, dtype=torch.float64)), 0))[len(network.dimensions) - 1]
  samples = [None] * times
  for i in range(0, times):
    samples[i] = network.staticOutput(x).item()
  print(network.forwardPass(torch.cat((x.unsqueeze(0), torch.ones(1, dtype=torch.float64)), 0))[len(network.dimensions) - 1])
  print("mean: ", np.mean(samples))
  print("var: ", np.var(samples))
  print("my mean: ", m_z_minus)
  print("my var: ", s_z_minus)

  print("mean error: ", (np.mean(samples) - m_z_minus).item())
  print("var error: ", (np.var(samples) - s_z_minus).item())

  plt.hist(samples)
  plt.show()

  return None

def simpleTestNetwork(network, data):
  """
  Input:  @param network: A netork class
          @param data: The data to train the network with in form of [x_data, y_data]
  The code used for testing. used to not be a function but I just but it here for simplicity
  """

  # Test the speed of the BNN algorithm
  start = time.time()
  network.train(data)
  end = time.time()
  length = end - start
  print("It took", length, "seconds!")

  # Plot data
  plt.plot(data[0], data[1], '.', label = "data")

  # Save data to plot the prediction of the BNN
  perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float64)
  perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
  perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)

  for i in range(0, data[0].size(0)):
    perceptron_Plot_static[i] = network.staticOutput(data[0][i])
    perceptron_Plot[i] = network.meanOutput(data[0][i])
    perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])

  # PLot the prediction of the BNN
  plt.plot(data[0], perceptron_Plot, label="BNN prediction", color="r")
  plt.fill_between(data[0], perceptron_Plot-perceptron_Plot_var, perceptron_Plot + perceptron_Plot_var, alpha=0.2, color="r")

  # Limit plot and show it
  plt.legend(loc='best')
  plt.ylim(-85, 85)
  plt.show()

  return None

def testNetwork(network, data):
  """
  Input:  @param network: A netork class
          @param data: The data to train the network with in form of [x_data, y_data]
  The code used for testing. used to not be a function but I just but it here for simplicity
  """
  # Calculate the start time
  start = time.time()
  plt.plot(data[0], data[1], '.', label = "data")

  n_plots = 2
  color = iter(plt.cm.rainbow(np.linspace(0, 1, n_plots)))

  perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float64)
  perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
  perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)
  mse = 0
  for i in range(0, data[0].size(0)):
    perceptron_Plot_static[i] = network.staticOutput(data[0][i])
    perceptron_Plot[i] = network.meanOutput(data[0][i])
    perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])
    mse += (data[1][i] - perceptron_Plot[i])**2
  mse = mse/data[0].size(0)
  # print("mse Nr. ", 0, " : ", mse)
  c = next(color)
  plt.plot(data[0], perceptron_Plot, label='epoch Nr. {k}'.format(k=0), color=c)
  plt.fill_between(data[0], perceptron_Plot-perceptron_Plot_var, perceptron_Plot + perceptron_Plot_var, alpha=0.2, color=c)

  for k in range(1, n_plots):
    network.train(data)
    perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)
    mse = 0
    for i in range(0, data[0].size(0)):
      perceptron_Plot_static[i] = network.staticOutput(data[0][i])
      perceptron_Plot[i] = network.meanOutput(data[0][i])
      perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])
      mse += (data[1][i] - perceptron_Plot[i])**2
    mse = mse/data[0].size(0)
    # print("mse Nr. ", k, " : ", mse)
    c = next(color)
    plt.plot(data[0], perceptron_Plot, label='epoch Nr. {k}'.format(k=k), color=c)
    plt.fill_between(data[0], perceptron_Plot-perceptron_Plot_var, perceptron_Plot + perceptron_Plot_var, alpha=0.2, color=c)

  # Calculate the end time and time taken
  end = time.time()
  length = end - start

  print("It took", length, "seconds!")

  plt.legend(loc='best', fontsize=36)
  plt.ylim(-85, 85)
  plt.savefig("my_plot.png", bbox_inches='tight')
  #plt.show()
  return None





torch.manual_seed(500098)
data = generateData(f4, -4, 4, 800)
network = Network_Class(1, [100, 1], [relu, id])


simpleTestNetwork(network, data)
sample(network, data, 100000, 0)