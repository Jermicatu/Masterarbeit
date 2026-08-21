import torch
import math
from . import LayerMomentApprox
from torch.func import vmap

def id(x):
    return x

class myMixBNN:
    # featuring Variational interference
    def __init__(self, dimensions, functions, mix_size, starting_variance = 5, data_variance = 1):
        self.dimensions = dimensions
        self.functions = functions
        self.mix_size = mix_size
        self.data_variance = data_variance
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions) - 1
        self.m = [torch.randn(dimensions[l + 1], dimensions[l] + 1, mix_size) * 0.1 for l in range(L)]
        self.s = [starting_variance * torch.ones(dimensions[l + 1], dimensions[l] + 1, mix_size) for l in range(L)]
        self.w = [torch.ones(dimensions[l + 1], dimensions[l] + 1, mix_size) / mix_size for l in range(L)]

    def forwardPass_vectorized(self, m_q, s_q, w_q, x_data, y_data):
        # Calculates the expected log probability of the data under given weights with mean m_q and var s_q
        dimensions = self.dimensions
        functions = self.functions
        mix_size = self.mix_size
        data_variance = self.data_variance
        L = len(dimensions) - 1
            
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        def singlePass(x, y, m_q, s_q, w_q):
            m_z_start = x.unsqueeze(-1).expand(-1, mix_size)
            print(m_z_start.size())
            s_z_start = torch.zeros_like(m_z_start)
            w_z_start = torch.ones_like(m_z_start) / mix_size
            z_l_moments = LayerMomentApprox.MixToMoments(m_z_start, s_z_start, w_z_start)
            for l in range(L):
                a_l_moments = LayerMomentApprox.activation_input_approx_vectorized(z_l_moments, m_q[l], s_q[l], w_q[l])
                z_l_moments = LayerMomentApprox.activation_output_approx_vectorized(a_l_moments, functions[l])

            m_output = z_l_moments[:, 1]
            s_output = z_l_moments[:, 2] - m_output**2            
            return - torch.sum(0.5 * math.log(2 * math.pi * data_variance) + ((y - m_output)**2 + s_output) / (2 * data_variance))
        
        sol = vmap(singlePass, in_dims=(0, 0, None, None, None))(x_data, y_data, m_q, s_q, w_q)     
        return sol.sum()

    def forwardPass(self, m_q, s_q, w_q, x_data, y_data):
        # Calculates the expected log probability of the data under given weights with mean m_q and var s_q
        dimensions = self.dimensions
        functions = self.functions
        mix_size = self.mix_size
        data_variance = self.data_variance
        L = len(dimensions) - 1
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        sol = 0

        for i in range(x_data.size(0)):
            y = y_data[i]
            x = x_data[i]
            m_z_start = x.unsqueeze(-1).expand(-1, mix_size)
            s_z_start = torch.zeros_like(m_z_start)
            w_z_start = torch.ones_like(m_z_start) / mix_size
            z_l_moments = LayerMomentApprox.MixToMoments(m_z_start, s_z_start, w_z_start)
            for l in range(L):
                a_l_moments = LayerMomentApprox.activation_input_approx_vectorized(z_l_moments, m_q[l], s_q[l], w_q[l])
                z_l_moments = LayerMomentApprox.activation_output_approx_vectorized(a_l_moments, functions[l])

            m_output = z_l_moments[:, 1]
            s_output = z_l_moments[:, 2] - m_output**2
            sol -= torch.sum(0.5 * math.log(2 * math.pi * data_variance) + ((y - m_output)**2 + s_output) / (2 * data_variance))
            
        return sol.sum()

    def ELBO(self, m_q, s_q, w_q, x_data, y_data, kl_weight=1.0):
        # Calculates the ELBO of all our weights in the network between the current p and new q distributions
        L = len(self.dimensions) - 1
        m_p = self.m
        s_p = self.s
        w_p = self.w

        # I am using matched component KL as an approximation

        my_ELBO = self.forwardPass(m_q, s_q, w_q, x_data, y_data)

        flat_mq = torch.cat([ml.flatten() for ml in m_q])
        flat_sq = torch.cat([sl.flatten() for sl in s_q])
        flat_wq = torch.cat([wl.flatten() for wl in w_q])
        flat_mp = torch.cat([ml.flatten() for ml in m_p])
        flat_sp = torch.cat([sl.flatten() for sl in s_p])
        flat_wp = torch.cat([wl.flatten() for wl in w_p])

        kl = 0

        for mq,sq,wq,mp,sp,wp in zip(flat_mq, flat_sq, flat_wq, flat_mp, flat_sp, flat_wp):
            kl += wp * (torch.log(wp) - 0.5 * torch.log(2 * math.pi * sp) - ((mq-mp)**2 + sq) / (2 * sp)) 
            kl += wq * (torch.log(wq) - 0.5 * torch.log(2 * math.pi * sq) - 0.5)
        print("likelyhood: ", my_ELBO)
        print("KL: ", kl)
        return my_ELBO + kl_weight*kl
    
    def train(self, x_data, y_data, epochs=500, lr=0.01):
        m_q = [m.clone().detach().requires_grad_(True) for m in self.m]
        log_s_q = [torch.log(s.clone().detach() / 10).requires_grad_(True) for s in self.s]  # variance is always a tenth of that of s_p
        logit_w_q = [torch.zeros_like(w).requires_grad_(True) for w in self.w]

        optimizer = torch.optim.Adam(m_q + log_s_q + logit_w_q, lr=lr)

        for epoch in range(epochs):

            kl_weight = min(1.0, epoch / epochs)

            optimizer.zero_grad()
            s_q = [torch.exp(ls) for ls in log_s_q]
            w_q = [torch.softmax(lw, dim=-1) for lw in logit_w_q]
            loss = -self.ELBO(m_q, s_q, w_q, x_data, y_data, kl_weight)
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: ELBO = {-loss.item():.2f}")

        self.m = [m.detach().clone() for m in m_q]
        self.s = [torch.exp(ls).detach().clone() for ls in log_s_q]
        self.w = [w.detach().clone() for w in w_q]

    def predict(self, x_data):
        dimensions = self.dimensions
        functions = self.functions
        mix_size = self.mix_size
        L = len(dimensions) - 1

        m_p = self.m
        s_p = self.s
        mix_w_p = self.w

        data_size = x_data.size(0)

        output_mean = torch.zeros(data_size, dimensions[-1])
        output_var = torch.zeros(data_size, dimensions[-1])

        for i in range(x_data.size(0)):
            x = x_data[i] # TODO: adjust example
            m_z_start = x.unsqueeze(-1).expand(-1, mix_size)
            s_z_start = torch.zeros_like(m_z_start)
            w_z_start = torch.ones_like(m_z_start) / mix_size
            z_l_moments = LayerMomentApprox.MixToMoments(m_z_start, s_z_start, w_z_start)
            for l in range(L):
                # either calculate m, s and w or just 3*mix_size - 1 moments
                a_l_moments = LayerMomentApprox.activation_input_approx_vectorized(z_l_moments, m_p[l], s_p[l], mix_w_p[l]) 
                z_l_moments = LayerMomentApprox.activation_output_approx_vectorized(a_l_moments, functions[l])

            output_mean[i, :], output_var[i,:] = z_l_moments[:, 1], z_l_moments[:, 2] - z_l_moments[:, 1]**2

        return output_mean, output_var
