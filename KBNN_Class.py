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

  def staticNetworkOutput(self, x):
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
  
  def forwardPass(self, x):
    network = self.network
    dimensions = self.dimensions
    input_size = self.input_size
    m_z_old = x
    s_z_old = torch.zeros(input_size+1, dtype=torch.float64)


    # iterate through each layer
    for i in range(0, len(network)):
      m_w, s_w, f = network[i]
      
      m_z_old = m_z_old.repeat(dimensions[i], 1, 1)
      s_z_old = s_z_old.repeat(dimensions[i], dimensions[i], 1, 1)

      print(torch.transpose(m_w, 1, 2))
      print(s_z_old)
      m_a = torch.flatten(torch.bmm(m_z_old, m_w))
      s_a = torch.flatten(torch.bmm(torch.bmm(torch.transpose(m_w, 1, 2), s_z_old), m_w) + torch.bmm(torch.bmm(torch.transpose(m_z_old, 1, 2), s_w), m_z_old)) + torch.vmap(torch.trace)(torch.bmm(s_w, s_z_old))

      if f == sig:
        l = math.sqrt(math.pi/8)
        t = torch.sqrt(1 + (l**2) * s_a).pow_(-1)
        m_z_new = sig(m_a * t)
        s_z_new = max(m_z_new * (torch.ones(dimensions[i]) - m_z_new) * (torch.ones(dimensions[i]) - torch.ones(dimensions[i]) * t) + torch.ones(dimensions[i]) * 10**(-6), torch.ones(dimensions[i]) * 10**(-6), dtype=torch.float64)
      if f == id:
        m_z_new = m_a
        s_z_new = s_a

    m_z_old = torch.cat((torch.tensor([1], dtype=torch.float64), m_z_new), 0)
    s_z_old = torch.block_diag(torch.tensor([[0]], dtype=torch.float64), s_z_new) # variance of bias is zero
    forwardPassData[i] = [m_a, s_a, m_z_new, s_z_new]

 
    forwardPassData = [None] * len(network)

    return forwardPassData
    

my_network = Network_Class(1, [3, 1], [sig, id])
print(my_network.network)
print(my_network.staticNetworkOutput(torch.tensor([5])))
my_network.forwardPass(torch.tensor([1, 5], dtype=torch.float64))