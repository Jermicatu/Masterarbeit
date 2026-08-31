import torch

def id(x):
    return x

def RMSE(y_data, y_approx):
    """root mean square error

    Args:
        y_data (torch.tensor): original y values
        y_approx (torch.tensor): approximated y values

    Returns:
        float: the RMSE
    """

    assert y_data.shape == y_approx.shape

    return torch.sqrt(torch.mean((y_data - y_approx)**2))

def MAE(y_data, y_approx):
    """mean average error

    Args:
        y_data (torch.tensor): original y values
        y_approx (torch.tensor): approximated y values

    Returns:
        float: the MAE
    """

    assert y_data.shape == y_approx.shape

    return torch.mean(torch.abs(y_data - y_approx))

def NLL(y_data, y_mu_approx, y_s_approx):
    """negative log likelihood

    Args:
        y_data (torch.tensor): original y values
        y_mu_approx (torch.tensor): approximated y mean values
        y_s_approx (torch.tensor): approximated y variance values

    Returns:
        float: the NLL
    """

    assert y_data.shape == y_mu_approx.shape

    return torch.mean((y_data - y_mu_approx)**2 / (2*y_s_approx) + 0.5 * torch.log(2*torch.pi*y_s_approx))

ERROR_FUNCTIONS = {
    "RMSE": RMSE,
    "MAE": MAE,
    "NLL": NLL,
}

FUNCTIONS = {
    "tanh": torch.tanh,
    "sig": torch.sigmoid,
    "id": id,
    "relu": torch.relu
}