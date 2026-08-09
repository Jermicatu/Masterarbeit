import src.UncertaintyQuantificationGauss as VI
import src.UncertaintyQuantificationGaussMix as VI_mix
import torch
import matplotlib.pyplot as plt
import time

def id(x):
    return x

def test_VI():
    dimensions = [1, 2, 1, 1]
    functions = [torch.tanh, torch.tanh, id]
    network = VI.PaperBNN(dimensions, functions)

    x_data = torch.linspace(-3, 3, steps=50)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data)
    y_m, y_s = network.predict(x_data)
    print(y_m)
    print(y_s)

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


def test_VI_mix():
    dimensions = [1, 4, 1, 1]
    functions = [torch.tanh, torch.tanh, lambda x: x]
    network = VI_mix.myMixBNN(dimensions, functions, mix_size=1)


    x_data = torch.linspace(-3, 3, steps=20).unsqueeze(-1)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data)
    y_m, y_s = network.predict(x_data)
    print(y_m)
    print(y_s)

    y_pred = y_m.squeeze().detach().numpy()
    y_s = y_s.squeeze().detach().numpy()
    x_data = x_data.squeeze().detach().numpy()
    y_data = y_data.detach().numpy()

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
    # test_VI()
    test_VI_mix()