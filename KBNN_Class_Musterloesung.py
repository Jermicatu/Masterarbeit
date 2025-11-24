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

        self.noise = 0.01

        self.H_matr = []

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

                Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(Ca[i]))

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
        for x, y in tqdm(zip(ds_x, ds_y), total=ds_y.size(0), disable=True):

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

                    Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                    pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(Ca[i]))

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