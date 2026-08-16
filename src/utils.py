import torch

def id(x):
    return x

def RMSE(y_data, y_approx):
    # root mean square error
    assert y_data.size() == y_approx.size()

    return torch.sqrt(torch.mean((y_data - y_approx)**2))

def MAE(y_data, y_approx):
    # mean absolute error
    assert y_data.size() == y_approx.size()

    return torch.mean(torch.abs(y_data - y_approx))

def NLL(y_data, y_mu_approx, y_s_approx):
    # negative log likelyhood
    assert y_data.size() == y_mu_approx.size()

    return torch.mean((y_data - y_mu_approx)**2 / (2*y_s_approx) + 0.5 * torch.log(2*torch.pi*y_s_approx))

FUNCTIONS = {
    "tanh": torch.tanh,
    "sig": torch.sigmoid,
    "id": id,
}