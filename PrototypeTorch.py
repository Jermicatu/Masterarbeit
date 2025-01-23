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

def staticPerceptron(W_mean, W_variance, f, X):
  """
  Input:  @param W_mean: tensor, with dtype=torch.float64, is the mean of the weights (plus bias) as a vector
          @param W_variance: tensor, with dtype=torch.float64, is the covariance matrix of the weights (plus bias)
          @param f: is the activation function of the perceptron
          @param X: torch vecotor of type torch.float64, is the input for the perceptron
  Output: @output sol: tensor, the static output of the perceptron
  """
  X_bias = torch.cat((torch.tensor([1]), X), 0)
  X_bias = torch.tensor(X_bias, dtype=torch.float64)
  W = torch.distributions.multivariate_normal.MultivariateNormal(W_mean, W_variance).sample()
  sol = f(torch.matmul(W , X_bias))
  return sol

def forwardPassPerceptron(W_mean, W_variance, f, X):
  """
  Input:  @param W_mean: tensor vector of type torch.float64, mean of the weights (plus bias) as a vector
          @param W_variance: tensor matrix of type torch.float64, covariance matrix (plus bias) of the weights
          @param f: activation function of the perceptron
          @param X: tensor vector of type torch.float64, Input for the perceptron
  Output: @output mean_a: float, the mean of a = W*X
          @output sigma_a_squared: float, the sigma**2 of a = W*X
          @output mean_y: float, the mean of y = f(a)
          @output sigma_y_squared: float, the sigma**2 of y = f(a)
  """
  if len(X.size()) == 0:
    X = X.unsqueeze(0)
  X_bias = torch.cat((torch.tensor([1], dtype=torch.float64), X), 0).unsqueeze(0).T
  mean_a = torch.matmul(W_mean, X_bias)
  sigma_a_squared = max(torch.matmul(torch.matmul(X_bias.mT, W_variance), X_bias) + 10**(-6), torch.tensor([10**(-6)], dtype=torch.float64))
  if f == sig:
    t = torch.sqrt(1 + (math.pi / 8) * sigma_a_squared)
    mean_y = sig(mean_a / t)
    sigma_y_squared = max(mean_y * (1 - mean_y) * (1 - 1/t) + 10**(-6), torch.tensor([10**(-6)], dtype=torch.float64))
  print(" mean_a; ", mean_a.item(), " sigma_a_squared; ", sigma_a_squared.item(), " mean_y; ", mean_y.item(), " sigma_y_squared; ", sigma_y_squared.item())
  print("W_mean; ", W_mean, " W_variance; ", W_variance, " X_bias: ", X_bias)
  #print("sigma_a_squared: ", sigma_a_squared)
  return mean_a, sigma_a_squared, mean_y, sigma_y_squared

def backwardPassPerceptron(W_mean_start, W_variance_start, f, data):
  """
  Input:  @param W_mean_start: tensor, mean of the starting weights (plus bias) as a vector
          @param W_variance_start: tensor, covariance matrix of the starting weights (plus bias)
          @param f: activation function of the perceptron
          @param data: Training data of the form [x_1, ..., x_n], [y_1, ..., y_n]
  Output: @output W_mean_end: tensor, the new mean vector
          @output W_variance_end: tensor, the new variance matrix
  """
  W_mean_i = W_mean_start
  W_variance_i = W_variance_start
  X, Y = data
  
  for i in range(0,len(X)):
    mean_a, sigma_a_squared, mean_y, sigma_y_squared = forwardPassPerceptron(W_mean_i, W_variance_i, f, X[i])
    
    if f == sig:
      # TODO Question: is N(a;0,1) the normal distribution f(a) = (1/math.sqrt(2*math.pi))*math.exp(-(a**2)/2)
      l = math.sqrt(math.pi/8)
      t = math.sqrt(1 + (l**2) * sigma_a_squared)
      cov_y_a_squared = ((l * sigma_a_squared) / t) * (1 / math.sqrt(2 * math.pi)) * math.exp(-((l * mean_a.item() / t)**2) / 2)
    
    k_i = cov_y_a_squared / sigma_y_squared
    mean_i = mean_a + k_i * (Y[i] - mean_y)
    sigma_i_squared = sigma_a_squared - k_i * cov_y_a_squared
    # [1, X[i]] because of bias
    l_i = torch.matmul(W_variance_i, (torch.tensor([[1], [X[i]]], dtype=torch.float64))) / (sigma_a_squared + 10**(-3))
    # squeeze to have a vector not a matrix
    W_mean_i = W_mean_i + (l_i * (mean_i - mean_a)).squeeze()
    W_variance_i = W_variance_i + torch.matmul(l_i, l_i.mT) * (sigma_i_squared - sigma_a_squared)  + torch.mul(torch.eye(W_mean_i.size(0)), 10^(-3))

  W_mean_end = W_mean_i
  W_variance_end = W_variance_i

  return W_mean_end, W_variance_end

def backwardPassPerceptronOnce(mean_a, sigma_a_squared, mean_y, sigma_y_squared, W_mean_start, W_variance_start, f, X, Y):
  """
  Input:  @param mean_a: float, the mean of a = W*X
          @param sigma_a_squared: float, the sigma**2 of a = W*X
          @param mean_y: float, the mean of y = f(a)
          @param sigma_y_squared: float, the sigma**2 of y = f(a)
          @param W_mean_start: tensor, mean of the starting weights (plus bias) as a vector
          @param W_variance_start: tensor, covariance matrix of the starting weights (plus bias)
          @param f: activation function of the perceptron
          @param x: tensor vector of type torch.float64, input for the perceptron
          @param y: tensor of type torch.float64, output for the perceptron
  Output: @output W_mean_end: tensor, the new mean vector
          @output W_variance_end: tensor, the new variance matrix
  """
  if len(X.size()) == 0:
    X = X.unsqueeze(0)
  X_bias = torch.cat((torch.tensor([1], dtype=torch.float64), X), 0).unsqueeze(0).T

  if f == sig:
    # TODO Question: is N(a;0,1) the normal distribution f(a) = (1/math.sqrt(2*math.pi))*math.exp(-(a**2)/2)
    l = math.sqrt(math.pi/8)
    t = math.sqrt(1 + (l**2) * sigma_a_squared)
    cov_y_a_squared = ((l * sigma_a_squared) / t) * (1 / math.sqrt(2 * math.pi)) * math.exp(-((l * mean_a.item() / t)**2) / 2)
  
  k_i = cov_y_a_squared / (sigma_y_squared*2) 
  mean_i = mean_a + k_i * (Y - mean_y)
  sigma_i_squared = sigma_a_squared - k_i * cov_y_a_squared
  l_i = torch.matmul(W_variance_start, X_bias) / sigma_a_squared
  # squeeze to have a vector not a matrix
  W_mean_end = W_mean_start + (l_i * (mean_i - mean_a)).squeeze()
  W_variance_end = W_variance_start + torch.matmul(l_i, l_i.mT) * (sigma_i_squared - sigma_a_squared) + torch.eye(W_mean_end.size(0), dtype=torch.float64) * 10**(-6)
  print("X: ", X, " mean_a; ", mean_a, " sigma_a_squared; ", sigma_a_squared, " mean_y; ", mean_y, " sigma_y_squared; ", sigma_y_squared)
  print("l_i :", l_i)
  #print("W_mean_start :", W_mean_start)
  print("W_variance_start :", W_variance_start)
  print("W_mean_start :", W_mean_start)
  print("torch.matmul(l_i, l_i.mT): ", torch.matmul(l_i, l_i.mT))
  print("sigma_i_squared: ", sigma_i_squared)
  print("sigma_a_squared: ", sigma_a_squared)
  print("cov_y_a_squared: ", cov_y_a_squared)
  print("sigma_y_squared: ", sigma_y_squared)
  print("k_i: ", k_i)
  #print("W_mean_end :", W_mean_end)
  print("W_variance_end :", W_variance_end)

  return W_mean_end, W_variance_end

def createNetwork(input_size, network_size, f = relu):
  """
  Input:  @param Size: An np.array that tells the width of each layer
                       example: np.array([3, 5, 3]) an network where the first layer has width 3, 2nd 5 and 3rd 3.
          @param f: function for all perceptrons in the network
                    default is the relu function
  Output: @output network: The Neural network where network[i][j] has the parameters needed for the perceptron at the ith layer and jth width
                           The parameters are of the form [W_mean, W_variance, f]
  """
  network = [None] * len(network_size)

  network[0] = [None] * (network_size[0])
  for j in range(0, network_size[0]):
    # +1 for the bias
    network[0][j] = [torch.rand(input_size+1) - 0.5*torch.ones(input_size+1, dtype=torch.float64), torch.eye(input_size+1, dtype=torch.float64), f]

  for i in range(1,len(network_size)):
    network[i] = [None] * (network_size[i])
    for j in range(0, network_size[i]):
      network[i][j] = [torch.rand(network_size[i-1]+1) - 0.5*torch.ones(network_size[i-1]+1, dtype=torch.float64), torch.eye(network_size[i-1]+1, dtype=torch.float64), f]
  return network

def trainNetwork(data, network):
  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
  x_data, y_data = data
  # iterate through each data point
  for k in range(0, len(x_data)):
    x = x_data[k]
    y = y_data[k]

    # forward pass
    # iterate through each layer
    forwardPassData = [None] * len(network)
    for i in range(0, len(network)):
      #iterate through each perceptron
      forwardPassData[i] = [None] * len(network[i])
      for j in range(0, len(network[i])):
        W_mean_i, W_variance_i, f = network[i][j]
        forwardPassData[i][j] = forwardPassPerceptron(W_mean_i, W_variance_i, f, x)
      # new x for next layer
      x = torch.zeros(len(network[i]))
      for j in range(0, len(network[i])):
        x[j] = forwardPassData[i][j][2]

    # backwarf pass
    # iterate through each layer but now backwards
    for i in range(len(network)-1, 0, -1):
      # get x
      x = torch.zeros(len(network[i-1]))
      for j in range(0, len(network[i-1])):
        x[j] = forwardPassData[i-1][j][2]

      if len(y.size()) == 0:
        y = y.unsqueeze(0)
      #iterate through each perceptron
      for j in range(0, len(network[i])):
        mean_a, sigma_a_squared, mean_y, sigma_y_squared = forwardPassData[i][j]
        W_mean_old, W_variance_old, f = network[i][j]
        W_mean, W_variance = backwardPassPerceptronOnce(mean_a, sigma_a_squared, mean_y, sigma_y_squared, W_mean_old, W_variance_old, f, x, y[j])
        network[i][j][0] = W_mean
        network[i][j][1] = W_variance
      y = x
    
    # TODO last part of loop where our x is from the data
    x = x_data[k]
    if len(x.size()) == 0:
      x = x.unsqueeze(0)
    if len(y.size()) == 0:
      y = y.unsqueeze(0)
    #iterate through each perceptron
    for j in range(0, len(network[0])):
      mean_a, sigma_a_squared, mean_y, sigma_y_squared = forwardPassData[0][j]
      W_mean_old, W_variance_old, f = network[i][j]
      W_mean, W_variance = backwardPassPerceptronOnce(mean_a, sigma_a_squared, mean_y, sigma_y_squared, W_mean_old, W_variance_old, f, x, y[j])
      network[i][j][0] = W_mean
      network[i][j][1] = W_variance

  return network

def staticNetworkOutput(network, x):
  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
  if len(x.size()) == 0:
    x = x.unsqueeze(0)
  x = torch.tensor(x, dtype=torch.float64)
  # iterate through each layer
  for i in range(0, len(network)):
    # iterate through each perceptron
    y = torch.zeros(len(network[i]))
    for j in range(0, len(network[i])):
      W_mean, W_variance, f = network[i][j]
      y[j] = staticPerceptron(W_mean, W_variance, f, x)
    x = y
  return y

data = generateData(f, -3, 3, 100)
network = createNetwork(1, [1,2,1], sig)
network = trainNetwork(data, network)
W_mean = torch.tensor([1, -1], dtype=torch.float64)
W_variance = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)

perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
for i in range(0, data[0].size(0)):
  perceptron_Plot[i] = staticNetworkOutput(network , data[0][i])

plt.plot(data[0], data[1])
plt.plot(data[0], perceptron_Plot)
plt.show()