import torch

def id(x):
    return x

FUNCTIONS = {
    "tanh": torch.tanh,
    "sig": torch.sigmoid,
    "id": id,
}