import torch
import math
import matplotlib.pyplot as plt

def id(x):
    return x


class myMixBNN:
    """
    Bayesian Neural Network with Gaussian Mixture variational posterior.
    
    Each weight and bias is a Gaussian mixture of size mix_size.
    Propagation uses moment matching through nonlinear activations.
    """
    
    def __init__(self, dimensions, functions, mix_size, starting_variance=5.0, data_variance=1.0):
        self.dimensions = dimensions
        self.functions = functions
        self.mix_size = mix_size
        self.data_variance = data_variance
        self.starting_variance = starting_variance

        assert len(functions) == len(dimensions) - 1, \
            f"size of functions ({len(functions)}) and dimensions - 1 ({len(dimensions)-1}) do not match"

        L = len(dimensions) - 1
        
        # Variational parameters: for each layer l, shape (output_dim, input_dim+1, mix_size)
        # We store raw parameters and apply constraints during forward pass
        # m: means (unconstrained)
        # log_s: log-variances (ensures positivity)
        # logit_w: logit-weights (ensures simplex via softmax)
        
        self.m = [torch.randn(dimensions[l + 1], dimensions[l] + 1, mix_size) * 0.1 
                  for l in range(L)]
        self.log_s = [torch.log(torch.ones(dimensions[l + 1], dimensions[l] + 1, mix_size) * starting_variance) 
                      for l in range(L)]
        self.logit_w = [torch.zeros(dimensions[l + 1], dimensions[l] + 1, mix_size) 
                        for l in range(L)]
        
        # Prior: fixed at N(0, starting_variance) per paper
        self.prior_mean = 0.0
        self.prior_var = starting_variance

    def _get_mixture_params(self):
        """Return constrained mixture parameters (differentiable)."""
        params = []
        for l in range(len(self.m)):
            m = self.m[l]
            s = torch.exp(self.log_s[l])
            w = torch.softmax(self.logit_w[l], dim=-1)
            params.append((m, s, w))
        return params

    def forwardPass(self, x_data, y_data):
        """
        Differentiable forward pass computing expected log-likelihood.
        """
        dimensions = self.dimensions
        functions = self.functions
        mix_size = self.mix_size
        data_variance = self.data_variance
        L = len(dimensions) - 1
        
        params = self._get_mixture_params()
        
        sol = torch.tensor(0.0)
        
        for i in range(len(x_data)):
            x = x_data[i]
            
            # Input layer: deterministic, mixture with all mass in one component
            n_in = dimensions[0]
            m_z = torch.zeros(n_in, mix_size)
            s_z = torch.zeros(n_in, mix_size)
            w_z = torch.zeros(n_in, mix_size)
            
            for j in range(n_in):
                m_z[j, 0] = x
                s_z[j, 0] = 0.0
                w_z[j, 0] = 1.0
            
            # Propagate through layers
            for l in range(L):
                f = functions[l]
                n_out = dimensions[l + 1]
                n_in_curr = dimensions[l]
                
                m_w, s_w, w_w = params[l]
                
                # Pre-activation: compute for each output node
                m_a = torch.zeros(n_out, mix_size)
                s_a = torch.zeros(n_out, mix_size)
                w_a = torch.zeros(n_out, mix_size)
                
                for node in range(n_out):
                    # Collect all terms: bias + sum_j w_{ij} * z_j
                    terms_m = []
                    terms_s = []
                    terms_w = []
                    
                    # Bias term (index 0)
                    for k in range(mix_size):
                        if w_w[node, 0, k] > 1e-10:
                            terms_m.append(m_w[node, 0, k])
                            terms_s.append(s_w[node, 0, k])
                            terms_w.append(w_w[node, 0, k])
                    
                    # Weighted input terms
                    for j in range(n_in_curr):
                        for k_w in range(mix_size):
                            for k_z in range(mix_size):
                                w_prod = w_w[node, j+1, k_w] * w_z[j, k_z]
                                if w_prod > 1e-10:
                                    # Product of two independent Gaussians
                                    m_x = m_w[node, j+1, k_w]
                                    s_x = s_w[node, j+1, k_w]
                                    m_y = m_z[j, k_z]
                                    s_y = s_z[j, k_z]
                                    
                                    m_prod = m_x * m_y
                                    s_prod = m_x**2 * s_y + m_y**2 * s_x + s_x * s_y
                                    
                                    terms_m.append(m_prod)
                                    terms_s.append(torch.clamp(s_prod, min=1e-8))
                                    terms_w.append(w_prod)
                    
                    # Sum all terms: exact mean and variance
                    if len(terms_w) > 0:
                        terms_w_t = torch.stack(terms_w)
                        total_w = terms_w_t.sum()
                        
                        if total_w > 1e-10:
                            terms_m_t = torch.stack(terms_m)
                            terms_s_t = torch.stack(terms_s)
                            
                            # Normalize weights
                            w_norm = terms_w_t / total_w
                            
                            # Mean of sum
                            mean_sum = (w_norm * terms_m_t).sum()
                            
                            # Variance of sum (independent terms)
                            var_sum = (w_norm * (terms_s_t + terms_m_t**2)).sum() - mean_sum**2
                            var_sum = torch.clamp(var_sum, min=1e-8)
                            
                            # Store as single component (moment-matched)
                            m_a[node, 0] = mean_sum
                            s_a[node, 0] = var_sum
                            w_a[node, 0] = 1.0
                
                # Apply activation using cubature rule
                m_z_new = torch.zeros(n_out, mix_size)
                s_z_new = torch.zeros(n_out, mix_size)
                w_z_new = w_a.clone()
                
                for node in range(n_out):
                    for k in range(mix_size):
                        if w_a[node, k] > 1e-10:
                            mu = m_a[node, k]
                            var = s_a[node, k]
                            
                            # 2-point cubature for 1D Gaussian
                            std = torch.sqrt(var + 1e-8)
                            pt_plus = mu + std
                            pt_minus = mu - std
                            
                            h_plus = f(pt_plus)
                            h_minus = f(pt_minus)
                            
                            m_z_new[node, k] = 0.5 * (h_plus + h_minus)
                            s_z_new[node, k] = torch.clamp(
                                0.5 * (h_plus**2 + h_minus**2) - m_z_new[node, k]**2,
                                min=1e-8
                            )
                        else:
                            m_z_new[node, k] = 0.0
                            s_z_new[node, k] = 1e-8
                
                m_z = m_z_new
                s_z = s_z_new
                w_z = w_z_new
                
                # Normalize weights per node
                for node in range(n_out):
                    w_sum = w_z[node, :].sum()
                    if w_sum > 1e-10:
                        w_z[node, :] = w_z[node, :] / w_sum
            
            # Final output: compute overall mean and variance
            m_out = m_z[0, :]  # assuming single output
            s_out = s_z[0, :]
            w_out = w_z[0, :]
            
            pred_mean = (w_out * m_out).sum()
            pred_var = (w_out * (s_out + m_out**2)).sum() - pred_mean**2
            pred_var = torch.clamp(pred_var, min=0.0)
            
            # Log-likelihood contribution
            const = -0.5 * math.log(2.0 * math.pi * data_variance)
            sol = sol + (const - ((y_data[i] - pred_mean)**2 + pred_var) / (2.0 * data_variance))
        
        return sol

    def KL_jensen_bound(self):
        """
        Jensen upper bound for KL(q||p) where q is a Gaussian mixture and p is N(0, prior_var).
        
        KL(q||p) <= sum_k alpha_k * KL(N(mu_k, sigma_k^2) || N(0, prior_var))
        """
        prior_mean = self.prior_mean
        prior_var = self.prior_var
        
        kl_total = torch.tensor(0.0)
        
        params = self._get_mixture_params()
        
        for l in range(len(params)):
            m, s, w = params[l]
            
            # Per-component KL: KL(N(m_k, s_k) || N(0, prior_var))
            # = 0.5 * [log(prior_var/s_k) - 1 + s_k/prior_var + m_k^2/prior_var]
            kl_per_comp = 0.5 * (
                torch.log(prior_var / s) - 1.0 
                + s / prior_var 
                + m**2 / prior_var
            )
            
            # Jensen bound: weighted sum
            kl_layer = (w * kl_per_comp).sum()
            kl_total = kl_total + kl_layer
        
        return kl_total

    def ELBO(self, x_data, y_data, kl_weight=1.0):
        """
        Evidence Lower Bound = E_q[log p(D|W)] - kl_weight * KL(q||p)
        """
        likelihood = self.forwardPass(x_data, y_data)
        kl = self.KL_jensen_bound()
        
        return likelihood - kl_weight * kl

    def train(self, x_data, y_data, epochs=500, lr=0.01):
        """
        Joint Adam optimization of all mixture parameters.
        """
        # Collect all trainable parameters
        trainable = []
        for l in range(len(self.m)):
            self.m[l] = self.m[l].clone().detach().requires_grad_(True)
            self.log_s[l] = self.log_s[l].clone().detach().requires_grad_(True)
            self.logit_w[l] = self.logit_w[l].clone().detach().requires_grad_(True)
            trainable.extend([self.m[l], self.log_s[l], self.logit_w[l]])
        
        optimizer = torch.optim.Adam(trainable, lr=lr)
        
        for epoch in range(epochs):
            # KL annealing: slowly increase regularization
            kl_weight = min(1.0, epoch / 150.0)
            
            optimizer.zero_grad()
            loss = -self.ELBO(x_data, y_data, kl_weight)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0 or epoch == epochs - 1:
                elbo = -loss.item()
                print(f"Epoch {epoch:3d}: ELBO = {elbo:.2f}, KLw = {kl_weight:.2f}")
        
        # Detach after training
        for l in range(len(self.m)):
            self.m[l] = self.m[l].detach()
            self.log_s[l] = self.log_s[l].detach()
            self.logit_w[l] = self.logit_w[l].detach()

    def predict(self, x_data):
        """
        Non-differentiable prediction (same logic as forwardPass but no gradient tracking).
        """
        with torch.no_grad():
            dimensions = self.dimensions
            functions = self.functions
            mix_size = self.mix_size
            L = len(dimensions) - 1
            
            params = self._get_mixture_params()
            
            results_mean = []
            results_var = []
            
            for i in range(len(x_data)):
                x = x_data[i]
                
                n_in = dimensions[0]
                m_z = torch.zeros(n_in, mix_size)
                s_z = torch.zeros(n_in, mix_size)
                w_z = torch.zeros(n_in, mix_size)
                
                for j in range(n_in):
                    m_z[j, 0] = x
                    s_z[j, 0] = 0.0
                    w_z[j, 0] = 1.0
                
                for l in range(L):
                    f = functions[l]
                    n_out = dimensions[l + 1]
                    n_in_curr = dimensions[l]
                    
                    m_w, s_w, w_w = params[l]
                    
                    m_a = torch.zeros(n_out, mix_size)
                    s_a = torch.zeros(n_out, mix_size)
                    w_a = torch.zeros(n_out, mix_size)
                    
                    for node in range(n_out):
                        terms_m = []
                        terms_s = []
                        terms_w = []
                        
                        for k in range(mix_size):
                            if w_w[node, 0, k] > 1e-10:
                                terms_m.append(m_w[node, 0, k].item())
                                terms_s.append(s_w[node, 0, k].item())
                                terms_w.append(w_w[node, 0, k].item())
                        
                        for j in range(n_in_curr):
                            for k_w in range(mix_size):
                                for k_z in range(mix_size):
                                    w_prod = w_w[node, j+1, k_w].item() * w_z[j, k_z].item()
                                    if w_prod > 1e-10:
                                        m_x = m_w[node, j+1, k_w].item()
                                        s_x = s_w[node, j+1, k_w].item()
                                        m_y = m_z[j, k_z].item()
                                        s_y = s_z[j, k_z].item()
                                        
                                        m_prod = m_x * m_y
                                        s_prod = m_x**2 * s_y + m_y**2 * s_x + s_x * s_y
                                        
                                        terms_m.append(m_prod)
                                        terms_s.append(max(s_prod, 1e-8))
                                        terms_w.append(w_prod)
                        
                        if len(terms_w) > 0:
                            total_w = sum(terms_w)
                            if total_w > 1e-10:
                                w_norm = [w / total_w for w in terms_w]
                                mean_sum = sum(w * m for w, m in zip(w_norm, terms_m))
                                var_sum = sum(w * (s + m**2) for w, m, s in zip(w_norm, terms_m, terms_s)) - mean_sum**2
                                var_sum = max(var_sum, 1e-8)
                                
                                m_a[node, 0] = mean_sum
                                s_a[node, 0] = var_sum
                                w_a[node, 0] = 1.0
                    
                    m_z_new = torch.zeros(n_out, mix_size)
                    s_z_new = torch.zeros(n_out, mix_size)
                    w_z_new = w_a.clone()
                    
                    for node in range(n_out):
                        for k in range(mix_size):
                            if w_a[node, k] > 1e-10:
                                mu = m_a[node, k].item()
                                var = s_a[node, k].item()
                                
                                std = math.sqrt(var + 1e-8)
                                pt_plus = mu + std
                                pt_minus = mu - std
                                
                                h_plus = f(torch.tensor(pt_plus)).item()
                                h_minus = f(torch.tensor(pt_minus)).item()
                                
                                m_z_new[node, k] = 0.5 * (h_plus + h_minus)
                                s_z_new[node, k] = max(0.5 * (h_plus**2 + h_minus**2) - m_z_new[node, k]**2, 1e-8)
                            else:
                                m_z_new[node, k] = 0.0
                                s_z_new[node, k] = 1e-8
                    
                    m_z = m_z_new
                    s_z = s_z_new
                    w_z = w_z_new
                    
                    for node in range(n_out):
                        w_sum = w_z[node, :].sum().item()
                        if w_sum > 1e-10:
                            w_z[node, :] = w_z[node, :] / w_sum
                
                m_out = m_z[0, :]
                s_out = s_z[0, :]
                w_out = w_z[0, :]
                
                pred_mean = (w_out * m_out).sum().item()
                pred_var = (w_out * (s_out + m_out**2)).sum().item() - pred_mean**2
                pred_var = max(pred_var, 0.0)
                
                results_mean.append(pred_mean)
                results_var.append(pred_var)
            
            return torch.tensor(results_mean), torch.tensor(results_var)


def test_mixture_bnn():
    dimensions = [1, 2, 1]
    functions = [torch.tanh, id]
    mix_size = 3
    
    network = myMixBNN(dimensions, functions, mix_size, starting_variance=5.0)
    
    x_data = torch.linspace(-3, 3, steps=50)
    y_data = torch.cos(x_data)
    
    print("Training mixture BNN...")
    network.train(x_data, y_data, epochs=500, lr=0.01)
    
    # Predict
    x_test = torch.linspace(-3, 3, steps=200)
    y_m, y_s = network.predict(x_test)
    
    y_pred = y_m
    y_std = torch.sqrt(torch.clamp(y_s, min=0))
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data.numpy(), y_data.numpy(), c='red', s=20, label='Training data')
    plt.plot(x_test.numpy(), y_pred.numpy(), 'b-', label='Predictive mean')
    plt.fill_between(x_test.numpy(),
                     (y_pred - 2*y_std).numpy(),
                     (y_pred + 2*y_std).numpy(),
                     alpha=0.3, label='±2 std (epistemic)')
    plt.plot(x_test.numpy(), torch.cos(x_test).numpy(), 'g--', label='True cos(x)')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title(f'Mixture BNN (mix_size={mix_size})')
    plt.show()
    
    # Print final ELBO
    final_elbo = network.ELBO(x_data, y_data, kl_weight=1.0).item()
    print(f"Final ELBO: {final_elbo:.2f}")


if __name__ == "__main__":
    test_mixture_bnn()