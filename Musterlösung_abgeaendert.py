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
from tqdm import tqdm

class KBNN:
    """This is the implementation of KBNN"""
    def __init__(self, layers, bias=False):
        """Init function which specifies the network architecture and whether to include a bias or not"""
        self.layers = layers
        if not bias:
            self.no_bias = True
        else:
            self.no_bias = False
        self.mw = []
        self.Cw = []
        for i in range(1, len(self.layers)):
            if self.no_bias:
                self.mw.append(torch.randn((self.layers[i-1], self.layers[i])))
                self.Cw.append(torch.zeros((self.layers[i], self.layers[i-1], self.layers[i-1])))
                for j in range(layers[i]):
                    self.Cw[-1][j, :, :] = torch.diag(torch.ones(self.layers[i-1]))
            else:
                self.mw.append(torch.randn((self.layers[i - 1] + 1, self.layers[i])))
                self.Cw.append(torch.zeros((self.layers[i], self.layers[i - 1] + 1, self.layers[i - 1] + 1)))
                for j in range(layers[i]):
                    self.Cw[-1][j, :, :] = torch.diag(torch.ones(self.layers[i - 1] + 1))
        self.my = [*torch.zeros(len(self.layers) - 1)]
        self.Cy = [*torch.zeros(len(self.layers) - 1)]
        self.ma = [*torch.zeros(len(self.layers) - 1)]
        self.Ca = [*torch.zeros(len(self.layers) - 1)]
        self.Cay = [*torch.zeros(len(self.layers) - 1)]

        self.act_fct = ["relu"] * (len(self.layers) - 1)
        self.act_fct[-1] = "linear"

        self.noise = 0.1

        self.H_matr = []

    def meanOutput(self, x):
        """
        Input:  @param x: vector, input for the network
        Output: @output y: vector, the mean output from our network
        """

        if len(x.size()) == 0:
            x = x.unsqueeze(0)
            x = x.to(float)

        # iterate through each layer 
        for i in range(0, len(self.layers)-1):
            f = self.act_fct[i]

            if f == "relu":
                x = torch.relu(torch.flatten(torch.matmul(torch.cat((x, torch.ones(1)), 0), self.mw[i].to(float))))
            elif f == "linear":
                x = torch.flatten(torch.matmul(torch.cat((x, torch.ones(1)), 0), self.mw[i].to(float)))
        return x
    
    def staticOutput(self, x):
        """
        Input:  @param x: vector, input for the network
        Output: @output y: vector, the approximation from our network
        """

        if len(x.size()) == 0:
            x = x.unsqueeze(0)
            x = x.to(float)

        # iterate through each layer
        for i in range(0, len(self.layers)-1):
            f = self.act_fct[i]

            m = torch.distributions.multivariate_normal.MultivariateNormal(self.mw[i].to(torch.float64).mT, self.Cw[i].to(torch.float64))
            if f == "relu":
                x = torch.relu(torch.matmul(torch.cat((x, torch.ones(1)), 0), m.sample().mT))
            elif f == "linear":
                x = torch.matmul(torch.cat((x, torch.ones(1)), 0), m.sample().mT)
        return x
    
    def forward_pass(self, x, training=False):
        """This is the implementation of the forward pass. Note that it can handle batches."""

        x = x.float()
        n_samples = x.size(0)

        if self.no_bias:
            mz_ = x
        else:
            mz_ = torch.cat((x, torch.ones((n_samples, 1))), 1)

        Cz_ = torch.zeros((n_samples, mz_.size(1), mz_.size(1)))

        ma, Ca, my, Cy = self.ma, self.Ca, self.my, self.Cy

        for i in range(len(self.layers) - 1):
            activation = self.act_fct[i]
            ma[i] = mz_.float().mm(self.mw[i].float())
            self.ma[i] = ma[i]

            A = torch.matmul(torch.t(self.mw[i].float()), torch.matmul(Cz_, self.mw[i].float()))
            A = torch.diagonal(A, dim1=1, dim2=2)
            C = torch.diagonal(Cz_, dim1=-2, dim2=-1).mm(torch.t(torch.diagonal(self.Cw[i], dim1=2)))
            B = torch.einsum('nmi,ni->nm', torch.einsum('mij,nj->nmi', self.Cw[i], mz_), mz_)

            Ca[i] = A + B + C
            Ca[i][Ca[i] <= 0] = self.noise

            self.Ca[i] = Ca[i]

            if activation == "relu":
                [alpha, beta] = [0, 1]
                diff = beta - alpha
                diff2 = beta ** 2 - alpha ** 2
                E1 = ma[i]
                E2 = ma[i] ** 2 + Ca[i]

                # Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                # pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(Ca[i]))

                Cg = torch.sqrt(Ca[i])/math.sqrt(2 * math.pi) * exp(-0.5 * torch.pow(ma[i], 2) * torch.pow(Ca[i], -1))
                pmCa = 0.5 * (1 + erf(ma[i] * torch.pow(Ca[i], -1/2) / math.sqrt(2)))

                my[i] = alpha * E1 + diff * (E1 * pmCa + Cg)
                Cy[i] = alpha ** 2 * E2 + diff2 * (E2 * pmCa + ma[i] * Cg) - my[i] ** 2 + self.noise

                Cya = alpha * E2 + diff * (E2 * pmCa + ma[i] * Cg) - my[i] * ma[i]

            elif activation == "linear":
                my[i] = ma[i]
                Cy[i] = Ca[i] + self.noise

                Cya = ma[i] ** 2 + Ca[i] - my[i] * ma[i]


            self.my[i] = my[i]
            self.Cy[i] = Cy[i]
            self.Cay[i] = Cya

            mz = my[i]
            if self.no_bias:
                mz_ = mz
            else:
                mz_ = torch.cat((mz, torch.ones((n_samples, 1))), 1)

            Cz = Cy[i]

            if self.no_bias:
                Cz_ = torch.diag_embed(Cz)
            else:
                Cz_ = torch.diag_embed(torch.cat((Cz, torch.zeros((n_samples, 1))), 1))

        if training:
            my = [m[0] for m in my]
            Cy = [C[0] for C in Cy]
            ma = [m[0] for m in ma]
            Ca = [C[0] for C in Ca]
            return my, Cy, ma, Ca
        return my[-1], Cy[-1], ma[-1], Ca[-1]

    @torch.no_grad()
    def train(self, ds_x, ds_y):
        """ This function implements the online learning of KBNN.

        @param ds_x: This parameter should be X_train
        @param ds_y: This parameter should be y_train
        """

        assert ds_x.size(-1) == self.layers[0], f"Last data dimension has to be the same as input layer size.\n" \
                                                f" last dim = {ds_x.shape[-1]} \n input layer size = {self.layers[0]} "
        pairs = list(zip(ds_x, ds_y))
        random.shuffle(pairs)
        for x, y in tqdm(pairs, total=ds_y.size(0), disable=True):
            my, Cy, ma, Ca = self.forward_pass(torch.unsqueeze(x, 0), training=True)

            my_new = y
            Cy_new = torch.zeros(y.size())

            for i in reversed(range(len(self.layers) - 1)):
               # load information for each layer
                if self.no_bias:
                    ni = self.layers[i]
                else:
                    ni = self.layers[i] + 1
                no = self.layers[i + 1]
                activation = self.act_fct[i]
                mz = my[i - 1]
                Cz = Cy[i - 1]
                if i == 0:
                    mz = x
                    Cz = torch.zeros(mz.size())

                if self.no_bias:
                    mz_ = mz
                    Cz_ = Cz
                else:
                    mz_ = torch.cat((mz, torch.ones(1)), 0)
                    Cz_ = torch.cat((Cz, torch.zeros(1)), 0)

                if activation == "relu":
                    [alpha, beta] = [0, 1]
                    diff = beta - alpha
                    E2 = ma[i] ** 2 + Ca[i]

                    # Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                    # pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(Ca[i]))

                    Cg = torch.sqrt(Ca[i])/math.sqrt(2 * math.pi) * exp(-0.5 * torch.pow(ma[i], 2) * torch.pow(Ca[i], -1))
                    pmCa = 0.5 * (1 + erf(ma[i] * torch.pow(Ca[i], -1/2) / math.sqrt(2)))

                    Cya = alpha * E2 + diff * (E2 * pmCa + ma[i] * Cg) - my[i] * ma[i]

                elif activation == "linear":
                    Cya = ma[i] ** 2 + Ca[i] - my[i] * ma[i]

                self.Cay[i] = Cya
                
                # Kalman Filter Step for a
                k = Cya / Cy[i]

                da = k * (my_new - my[i])
                Da = (k ** 2) * (Cy_new - Cy[i])

                Cwa = self.Cw[i] @ mz_.float()
                Cz_new = Cz_.unsqueeze(-1).repeat(1, no)
                Cza = Cz_new * self.mw[i]

                Ca_inv = 1 / Ca[i]

                L_up = Cwa * torch.outer(Ca_inv, torch.ones((ni)))
                L_low = Cza * Ca_inv.unsqueeze(0).repeat(ni, 1)
                
                self.mw[i] = self.mw[i] + torch.t(L_up * torch.outer(da, torch.ones((ni))))
                if self.no_bias:
                    my_new = mz + L_low @ da
                else:
                    my_new = mz + L_low[:-1] @ da
                

                E = L_up * torch.outer(Da, torch.ones((ni)))
                F = L_up.unsqueeze(-1) @ torch.ones((1, ni))
                
                self.Cw[i] = self.Cw[i] + E.unsqueeze(1).repeat(1, ni, 1) * F
                #for k in range(self.Cw[i].shape[0]):
                #    self.Cw[i][k] = torch.diag(torch.diag(self.Cw[i][k]))

                G = L_low ** 2 * Da.unsqueeze(0).repeat(ni, 1)
                
                if self.no_bias:
                    Cy_new = Cz + G @ torch.ones((no))
                else:
                    Cy_new = Cz + G[:-1] @ torch.ones((no))

def f3(x):
  return cos(x)

def f4(x):
  return x**3

def generateData(f, a, b, points, var, scaling):
  """
  Input:  @param f: function f(x) TODO multiple dimensions: for now one dimensional 
          @param a: creates the intervall with b 
          @param b: creates the intervall with a
          @param points: is the amount of equidistant points on the intervall [a, b]
  Output: @output x: the tensor of the x values used to sample y
          @output y: the y values we get through f(x) + pertubation
  """
  x = torch.linspace(a, b, points, dtype=torch.float64)
  y = (f(x) + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)) * var) * scaling
  x = x.to(float)
  y = y.to(float)
  return x, y

def sample(network, data, times, place):
  """
  Input:  @param network: A netork class
          @param data: The data to train the network with in form of [x_data, y_data]
          @times: The amount of samples taken
          @place: Place is the number of the data point you want to sample for. 
                  For example: in a dataset with 800 points 0 is the first point and 799 would be the last.
  """
  x = data[0][place]
  my, Cy, ma, Ca = network.forward_pass(x=torch.tensor([[data[0][place].item()]]), training=False)
  samples = [None] * times
  for i in range(0, times):
    samples[i] = network.staticOutput(x.unsqueeze(0)).item()

  print(x)
  
  print("mean: ", np.mean(samples))
  print("var: ", np.var(samples))

  print("mean error: ", (np.mean(samples) - my).item())
  print("var error: ", (np.var(samples) - Cy).item())

  plt.hist(samples)
  plt.show()

def plotNetwork(network, data, y_1, y_2, sclaing):
    plt.plot(data[0], data[1]/scaling, '.', label = "f")
    perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)

    for i in range(0, data[0].size(0)):
        perceptron_Plot[i] = network.meanOutput(data[0][i])
        my, Cy, ma, Ca = network.forward_pass(x=torch.tensor([[data[0][i].item()]]), training=False)
        perceptron_Plot_var[i] = 2 * torch.sqrt(Cy)
    plt.plot(data[0], perceptron_Plot/sclaing, label='epoch Nr. {k}'.format(k=0))
    plt.fill_between(data[0], (perceptron_Plot-perceptron_Plot_var)/sclaing, (perceptron_Plot + perceptron_Plot_var)/sclaing, alpha=0.2)
    plt.ylim(y_1, y_2)
    plt.show()



"""
myKBNN.mw[0] = torch.tensor([[ 0.8236, -0.7421, -0.5944, -0.3985, -1.4027,  0.7038,  0.0510,  0.8372,
         -0.3622, -0.7125],
        [ 0.6072,  0.4976, -0.4011, -0.7199, -0.2686, -0.3955,  1.9172,  0.2522,
         -0.1279,  0.1740]], dtype=torch.float64).float()
myKBNN.mw[1] = torch.tensor([[-0.9959],
        [ 1.1563],
        [-0.3992],
        [ 1.2153],
        [-0.8115],
        [-0.8848],
        [-0.0070],
        [-1.7700],
        [-1.1698],
        [-0.2593],
        [ 0.2692]], dtype=torch.float64).float()
"""

"""
data = (torch.tensor([-4.0000, -3.9192, -3.8384, -3.7576, -3.6768, -3.5960, -3.5152, -3.4343,
        -3.3535, -3.2727, -3.1919, -3.1111, -3.0303, -2.9495, -2.8687, -2.7879,
        -2.7071, -2.6263, -2.5455, -2.4646, -2.3838, -2.3030, -2.2222, -2.1414,
        -2.0606, -1.9798, -1.8990, -1.8182, -1.7374, -1.6566, -1.5758, -1.4949,
        -1.4141, -1.3333, -1.2525, -1.1717, -1.0909, -1.0101, -0.9293, -0.8485,
        -0.7677, -0.6869, -0.6061, -0.5253, -0.4444, -0.3636, -0.2828, -0.2020,
        -0.1212, -0.0404,  0.0404,  0.1212,  0.2020,  0.2828,  0.3636,  0.4444,
         0.5253,  0.6061,  0.6869,  0.7677,  0.8485,  0.9293,  1.0101,  1.0909,
         1.1717,  1.2525,  1.3333,  1.4141,  1.4949,  1.5758,  1.6566,  1.7374,
         1.8182,  1.8990,  1.9798,  2.0606,  2.1414,  2.2222,  2.3030,  2.3838,
         2.4646,  2.5455,  2.6263,  2.7071,  2.7879,  2.8687,  2.9495,  3.0303,
         3.1111,  3.1919,  3.2727,  3.3535,  3.4343,  3.5152,  3.5960,  3.6768,
         3.7576,  3.8384,  3.9192,  4.0000], dtype=torch.float64).float(), torch.tensor([-49.2614, -56.0479, -70.2470, -66.6657, -33.5495, -48.6480, -44.4800,
        -52.0064, -34.5884, -35.2051, -43.3419, -35.9590, -19.9231, -29.0145,
        -28.2866, -27.9268,   2.4009,  -8.3378, -19.1845, -22.6919, -14.5839,
        -17.0293,  -9.8139, -11.0184, -12.2487,  -1.5261,  -7.3148, -22.2545,
          6.6193,  -0.5937,  -1.9189,  10.6352,  -4.9213,  -8.2295,  11.6194,
         -3.0420,  14.0862,   1.2730,   6.6918, -19.8359,  -1.6356,   6.8496,
          4.1625, -13.2579, -14.6102,  -1.7011,  -4.2302, -13.5901,   6.5326,
          3.2771,   1.7299,   8.7203,   1.3925,   1.3568,  -9.0499,  -6.4789,
        -14.1969,   7.2401,  17.6370,  -1.0319,   8.5360,   6.7432,  11.1921,
         11.8800,  -2.5362,  14.4712,  12.2564,  14.7977,   7.1260,   8.7688,
         -2.4774,  -0.6965,   8.2758,  -5.1370,   5.2870,   3.9167,  -5.9182,
          9.8799,  27.1779,  14.7506,   8.4463,  13.4219,  21.3305,   8.3676,
         14.5414,  21.5296,  23.8031,  31.3858,  38.8362,  26.0820,  34.5406,
         44.3429,  39.0464,  44.4923,  50.5254,  47.9237,  54.0630,  57.0344,
         55.6574,  55.2323], dtype=torch.float64).float())
"""

scaling = 1
x_1 = -1
x_2 = 1
y_1 = 0
y_2 = 1.5
data_var = 0.1
net_var = 1
test_f = f3

myKBNN = KBNN([1, 100, 1], bias=True)

torch.manual_seed(500098)
data = generateData(test_f, x_1, x_2, 800, data_var, scaling)

x, y = data

myKBNN.train(x.unsqueeze(0).mT.float(), y.unsqueeze(0).mT.float())

plotNetwork(myKBNN, data, y_1, y_2, scaling)

sample(myKBNN, data, 100000, 90)