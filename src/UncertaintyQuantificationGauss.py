import torch
import math

def id(x):
    return x

class VI_BNN:
    def __init__(self, dimensions, functions, starting_variance = 5, data_variance = 1):
        """Initialize the VI BNN

        Args:
            dimensions (list): A list of integers describing the layer sizes [d_0, ..., d_L]
            functions (list): A list of activation functions [f_1, ..., f_L]
            starting_variance (int, optional): The starting variance of all network weights. Defaults to 5.
            data_variance (float, optional): The assumed data variance. Defaults to 1.
        """

        # save important variables for later use
        self.dimensions = dimensions
        self.functions = functions
        self.data_variance = data_variance
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, f"size of functions and dimensions - 1 do not match" 

        L = len(dimensions) - 1

        # initialize network weight means list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
        self.m = [torch.randn(dimensions[l + 1], dimensions[l] + 1) * 0.1 for l in range(L)]

        # initialize network weight variances list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
        self.s = [starting_variance * torch.ones(dimensions[l + 1], dimensions[l] + 1) for l in range(L)]
        
    def forwardPass(self, m_q, s_q, x_data, y_data):
        """Calculates the expected log probability of the data under given weights with mean m_q and var s_q

        Args:
            m_q (List): mean list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
            s_q (List): var list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
            x_data (torch.tensor): x values of the training data
            y_data (torch.tensor): y values of the training data

        Returns:
            float: expected log probability of the network output
        """

        # prepare variables
        dimensions = self.dimensions
        functions = self.functions
        data_variance = self.data_variance
        L = len(dimensions) - 1
        
        assert len(x_data) == len(y_data), f"size of x and y do not match"

        sol = 0

        # iterate over data set
        for i in range(len(x_data)):
            m_z_l = x_data[i:i+1]
            s_z_l = torch.zeros_like(m_z_l)
            y = y_data[i]

            # iterate over the layers
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(m_q[l][:,1:], m_z_l) + m_q[l][:,0]
                s_a_l = (s_q[l][:,0] + torch.matmul(m_q[l][:,1:] ** 2, s_z_l)
                + torch.matmul(s_q[l][:,1:], m_z_l ** 2) + torch.matmul(s_q[l][:,1:], s_z_l))

                # approximate mean and sigma values for current layer output
                m_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6)) + f(m_a_l - torch.sqrt(s_a_l + 1e-6))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6))**2 + f(m_a_l - torch.sqrt(s_a_l + 1e-6))**2) / 2 - m_z_l ** 2

            # add expected log probability of current data point to final value
            sol -= 0.5 * math.log(2 * math.pi * data_variance) + ((y - m_z_l)**2 + s_z_l) / (2 * data_variance)

        return sol.sum()

    def ELBO(self, m_q, s_q, x_data, y_data, kl_weight=1.0):
        """Calculates the ELBO of all our weights in the network between the current p and new q distributions

        Args:
            m_q (List): mean list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
            s_q (List): var list with torch tensors with sizes [(d_1, d_0 + 1), ..., (d_L, d_{L-1} + 1)]
            x_data (torch.tensor): x values of the training data
            y_data (torch.tensor): y values of the training data
            kl_weight (float, optional): weight of the KL part of the . Defaults to 1.0.

        Returns:
            float: ELBO of all the network weights
        """

        # prepare variables
        m_p = self.m
        s_p = self.s

        # get the expected log probability
        log_prob = self.forwardPass(m_q, s_q, x_data, y_data)

        # The since all variables are of the same structure we can flatten them and still jointly iterate over 
        flat_mq = torch.cat([ml.flatten() for ml in m_q])
        flat_sq = torch.cat([sl.flatten() for sl in s_q])
        flat_mp = torch.cat([ml.flatten() for ml in m_p])
        flat_sp = torch.cat([sl.flatten() for sl in s_p])

        kl = 0

        # iterate over all weights in the network
        for mq,sq,mp,sp in zip(flat_mq, flat_sq, flat_mp, flat_sp):
            kl += (- 0.5 * torch.log(2 * math.pi * sp) - ((mq-mp)**2 + sq) / (2 * sp)) 
            kl += (- 0.5 * torch.log(2 * math.pi * sq) - 0.5)

        # return the ELBO value
        return log_prob + kl_weight * kl
    
    def train(self, x_data, y_data, epochs=500, lr=0.01):
        """training algorithm for the network

        Args:
            x_data (torch.tensor): x values of the training data
            y_data (torch.tensor): y values of the training data
            epochs (int, optional): _description_. Defaults to 500.
            lr (float, optional): _description_. Defaults to 0.01.
        """

        # copy current network weights
        m_q = [m.clone().detach().requires_grad_(True) for m in self.m]
        # initial variance is a tenth of that of s_p
        log_s_q = [torch.log(s.clone().detach() / 10).requires_grad_(True) for s in self.s]

        # prepare for ELBO optimisaion 
        optimizer = torch.optim.Adam(m_q + log_s_q, lr=lr)

        # iterize over the epochs
        for epoch in range(epochs):

            kl_weight = min(1.0, epoch / epochs)

            optimizer.zero_grad()
            s_q = [torch.exp(ls) for ls in log_s_q]
            loss = -self.ELBO(m_q, s_q, x_data, y_data, kl_weight)
            loss.backward()
            optimizer.step()

            #if epoch % 50 == 0:
            #    print(f"Epoch {epoch}: ELBO = {-loss.item():.2f}")

        # set the mean and variance to the new values
        self.m = [m.detach().clone() for m in m_q]
        self.s = [torch.exp(ls).detach().clone() for ls in log_s_q]

    def predict(self, x_data):
        """Calculate the mean and variance values of the newtork output

        Args:
            x_data (torch.tensor): x values for which we want to know the network prediction

        Returns:
            torch.tensor, torch.tensor: predicted mean and variance values (y_mean, y_var) 
        """

        # prepare variables
        dimensions = self.dimensions
        functions = self.functions
        L = len(dimensions) - 1
        
        m_p = self.m
        s_p = self.s

        y_mean = torch.zeros(dimensions[-1], len(x_data))
        y_var = torch.ones(dimensions[-1], len(x_data))

        # iterate over x values
        for i in range(len(x_data)):
            m_z_l = x_data[i:i+1]
            s_z_l = torch.zeros_like(m_z_l)

            # iterate over layers
            for l in range(L):
                f = functions[l]
                m_a_l = torch.matmul(m_p[l][:,1:], m_z_l) + m_p[l][:,0]
                s_a_l = (s_p[l][:,0] + torch.matmul(m_p[l][:,1:] ** 2, s_z_l)
                + torch.matmul(s_p[l][:,1:], m_z_l ** 2) + torch.matmul(s_p[l][:,1:], s_z_l))

                # approximate mean and sigma values for current layer output
                m_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6)) + f(m_a_l - torch.sqrt(s_a_l + 1e-6))) / 2
                s_z_l = (f(m_a_l + torch.sqrt(s_a_l + 1e-6))**2 + f(m_a_l - torch.sqrt(s_a_l + 1e-6))**2) / 2 - m_z_l ** 2

            # save mean and variance values from the last layer
            y_mean[:,i] = m_z_l
            y_var[:,i] = s_z_l

        return y_mean, y_var
