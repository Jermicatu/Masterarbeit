import torch
import math
import matplotlib.pyplot as plt

def id(x):
    return x

class PaperBNN:
    def __init__(self, dimensions, functions, starting_variance = 10, data_variance = 1):
        self.dimensions = dimensions
        self.functions = functions
        self.data_variance = data_variance
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions) - 1
        self.m = [torch.randn(dimensions[l + 1], dimensions[l] + 1) * 0.1 for l in range(L)]
        self.s = [starting_variance * torch.ones(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        
    def forwardPass(self, m_q, s_q, x_data, y_data):
        # Calculates the expected log probability of the data under given weights with mean m_q and var s_q
        dimensions = self.dimensions
        functions = self.functions
        data_variance = self.data_variance
        L = len(dimensions) - 1
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        sol = 0

        for i in range(len(x_data)):
            m_z_l = x_data[i:i+1]
            s_z_l = torch.zeros_like(m_z_l)
            y = y_data[i]
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(m_q[l][:,1:], m_z_l) + m_q[l][:,0]
                s_a_l = (s_q[l][:,0] + torch.matmul(m_q[l][:,1:] ** 2, s_z_l)
                + torch.matmul(s_q[l][:,1:], m_z_l ** 2) + torch.matmul(s_q[l][:,1:], s_z_l))

                m_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6)) + f(m_a_l - torch.sqrt(s_a_l + 1e-6))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6))**2 + f(m_a_l - torch.sqrt(s_a_l + 1e-6))**2) / 2 - m_z_l ** 2

            sol -= 0.5 * math.log(2 * math.pi * data_variance) + ((y - m_z_l)**2 + s_z_l) / (2 * data_variance)

        return sol.sum()

    def ELBO(self, m_q, s_q, x_data, y_data, kl_weight=1.0):
        # Calculates the ELBO of all our weights in the network between the current p and new q distributions
        L = len(self.dimensions) - 1
        m_p = self.m
        s_p = self.s

        my_ELBO = self.forwardPass(m_q, s_q, x_data, y_data)

        flat_mq = torch.cat([ml.flatten() for ml in m_q])
        flat_sq = torch.cat([sl.flatten() for sl in s_q])
        flat_mp = torch.cat([ml.flatten() for ml in m_p])
        flat_sp = torch.cat([sl.flatten() for sl in s_p])

        kl = 0

        for mq,sq,mp,sp in zip(flat_mq, flat_sq, flat_mp, flat_sp):
            kl += 0.5 * torch.log(sp) + ((mq-mp)**2 + sq) / (2 * sp) - 0.5 * torch.log(sq) - 0.5
        
        return my_ELBO - kl_weight * kl
    
    def train(self, x_data, y_data, epochs=500, lr=0.01):
        m_q = [m.clone().detach().requires_grad_(True) for m in self.m]
        log_s_q = [torch.log(s.clone().detach() / 10).requires_grad_(True) for s in self.s]  # variance is always a tenth of that of s_p

        optimizer = torch.optim.Adam(m_q + log_s_q, lr=lr)

        for epoch in range(epochs):

            kl_weight = min(1.0, epoch / epochs)

            optimizer.zero_grad()
            s_q = [torch.exp(ls) for ls in log_s_q]
            loss = -self.ELBO(m_q, s_q, x_data, y_data, kl_weight)
            loss.backward()
            optimizer.step()

            #if epoch % 50 == 0:
            #    print(f"Epoch {epoch}: ELBO = {-loss.item():.2f}")

        self.m = [m.detach().clone() for m in m_q]
        self.s = [torch.exp(ls).detach().clone() for ls in log_s_q]

    def predict(self, x_data):
        dimensions = self.dimensions
        functions = self.functions
        L = len(dimensions) - 1
        
        m_p = self.m
        s_p = self.s

        y_mean = torch.zeros(dimensions[-1], len(x_data))
        y_var = torch.ones(dimensions[-1], len(x_data))

        for i in range(len(x_data)):
            m_z_l = x_data[i:i+1]
            s_z_l = torch.zeros(1)
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(m_p[l][:,1:], m_z_l) + m_p[l][:,0]
                s_a_l = (s_p[l][:,0] + torch.matmul(m_p[l][:,1:] ** 2, s_z_l)
                + torch.matmul(s_p[l][:,1:], m_z_l ** 2) + torch.matmul(s_p[l][:,1:], s_z_l))

                m_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6)) + f(m_a_l - torch.sqrt(s_a_l + 1e-6))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6))**2 + f(m_a_l - torch.sqrt(s_a_l + 1e-6))**2) / 2 - m_z_l ** 2

            y_mean[:,i] = m_z_l
            y_var[:,i] = s_z_l

        return y_mean, y_var

def test1():
    dimensions = [1, 2, 1, 1]
    functions = [torch.tanh, torch.tanh, id]
    network = PaperBNN(dimensions, functions)

    x_data = torch.linspace(-3, 3, steps=50)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data)
    y_m, y_s = network.predict(x_data)

    y_pred = y_m.squeeze(0)
    y_s = y_s.squeeze(0)

    plt.plot(x_data, y_data, 'ro', label="cos(x)")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)


    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((-1, 1))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("INITIATE TESTS:")
    test1()