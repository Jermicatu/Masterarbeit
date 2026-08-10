import torch

def id(x):
    return x

FUNCTIONS = {
    "tanh": torch.tanh,
    "id": id,
}