import src.KBNN_Class_MAIN as KBNN
import src.UncertaintyQuantificationGauss as VI
import src.UncertaintyQuantificationGaussMix as VI_mix
import src.experiments_utils as utils
import torch
import matplotlib.pyplot as plt
import time
import json

def id(x):
    return x

def test_KBNN_classic():
    """
    Input:  @param network: A netork class
            @param data: The data to train the network with in form of [x_data, y_data]
    The code used for testing. used to not be a function but I just but it here for simplicity
    """

    scaling = 50
    x_1 = -1
    x_2 = 1
    y_1 = 0
    y_2 = 1.5
    data_var = 0.1
    net_var = 1


    test_f = KBNN.f3

    data = KBNN.generateData(test_f, x_1, x_2, 800, data_var, scaling)
    network = KBNN.Network_Class(1, [100, 1], [KBNN.relu, KBNN.id], net_var)

    # Test the speed of the BNN algorithm
    start = time.time()
    network.train(data)
    end = time.time()
    length = end - start
    print("It took", length, "seconds!")

    # Plot data
    plt.plot(data[0], data[1]/scaling, '.', label = "data", color="r")

    print(data[0].size())
    print(data[1].size())

    # Save data to plot the prediction of the BNN
    perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)

    for i in range(0, data[0].size(0)):
        perceptron_Plot_static[i] = network.staticOutput(data[0][i])
        perceptron_Plot[i] = network.meanOutput(data[0][i])
        perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])

    # PLot the prediction of the BNN
    plt.plot(data[0], perceptron_Plot/scaling, label="KBNN prediction", color="b")
    plt.fill_between(data[0], (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.2, color="b")

    # Limit plot and show it
    plt.legend(loc='best')
    plt.ylim(y_1, y_2)
    plt.show()

    return None

def test_VI_classic():
    dimensions = [1, 2, 1, 1]
    functions = [torch.tanh, torch.tanh, id]
    network = VI.PaperBNN(dimensions, functions)

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

def test_VI_mix_classic():
    dimensions = [1, 4, 1, 1]
    functions = [torch.tanh, torch.tanh, lambda x: x]
    network = VI_mix.myMixBNN(dimensions, functions, mix_size=2)


    x_data = torch.linspace(-3, 3, steps=20).unsqueeze(-1)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data)
    y_m, y_s = network.predict(x_data)

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

def save_data():
    # this function has to be rewritten for each new dataset we want to make
    x_data = torch.linspace(-3, 3, steps=20)
    y_data = torch.cos(x_data)
    torch.save({
        "x_data": x_data,
        "y_data": y_data
    }, "Cos_data_20.pt")

def train_KBNN_json(name):
    # TODO thids function and the next

    with open(f"Tests/{name}.json", "r") as f:
        config = json.load(f)

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]

    data = torch.load(config["data_file"])
    x_data = data["x_data"]
    y_data = data["y_data"]

    net_var = 1

    network = KBNN.Network_Class(dimensions[0], dimensions[1:], functions, net_var)

    network.train(x_data, y_data)
    
def plot_KBNN_json(name):

    # Plot data
    plt.plot(data[0], data[1]/scaling, '.', label = "data", color="r")

    print(data[0].size())
    print(data[1].size())

    # Save data to plot the prediction of the BNN
    perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float64)
    perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float64)

    for i in range(0, data[0].size(0)):
        perceptron_Plot_static[i] = network.staticOutput(data[0][i])
        perceptron_Plot[i] = network.meanOutput(data[0][i])
        perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])

    # PLot the prediction of the BNN
    plt.plot(data[0], perceptron_Plot/scaling, label="KBNN prediction", color="b")
    plt.fill_between(data[0], (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.2, color="b")

    # Limit plot and show it
    plt.legend(loc='best')
    plt.ylim(y_1, y_2)
    plt.show()

def train_VI_json(name):

    with open(f"Tests/{name}.json", "r") as f:
        config = json.load(f)

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]
    epochs = config["epochs"]
    # learning_rate = config["learning_rate"]

    data = torch.load(config["data_file"])
    x_data = data["x_data"]
    y_data = data["y_data"]

    torch.manual_seed(seed)

    network = VI.PaperBNN(dimensions, functions)

    network.train(x_data, y_data, epochs)
    torch.save({
        "m": network.m,
        "s": network.s,
        "dimensions": network.dimensions,
    }, f"Tests/{name}.pt")

def plot_VI_json(name):

    with open(f"Tests/{name}.json", "r") as f:
        config = json.load(f)

    network_data = torch.load(f"Tests/{name}.pt")


    network = VI.PaperBNN(
        config["dimensions"],
        [utils.FUNCTIONS[func] for func in config["functions"]]
    )

    network.m = network_data["m"]
    network.s = network_data["s"]

    test_data = torch.load(config["data_file"])
    x_data = test_data["x_data"]
    y_data = test_data["y_data"]

    y_m, y_s = network.predict(x_data)

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

def train_VI_mix_json(name):

    with open(f"Tests/{name}.json", "r") as f:
        config = json.load(f)

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    mix_size = config["mix_size"]
    seed = config["seed"]
    epochs = config["epochs"]
    # learning_rate = config["learning_rate"]

    data = torch.load(config["data_file"])
    x_data = data["x_data"].unsqueeze(-1)
    y_data = data["y_data"].unsqueeze(-1)

    torch.manual_seed(seed)

    network = VI_mix.myMixBNN(dimensions, functions, mix_size)

    network.train(x_data, y_data, epochs)
    torch.save({
        "m": network.m,
        "s": network.s,
        "dimensions": network.dimensions,
    }, f"Tests/{name}.pt")

def plot_VI_mix_json(name):

    with open(f"Tests/{name}.json", "r") as f:
        config = json.load(f)

    network_data = torch.load(f"Tests/{name}.pt")


    network = VI_mix.myMixBNN(
        config["dimensions"],
        [utils.FUNCTIONS[func] for func in config["functions"]],
        config["mix_size"]
    )

    network.m = network_data["m"]
    network.s = network_data["s"]

    test_data = torch.load(config["data_file"])
    x_data = test_data["x_data"].unsqueeze(-1)
    y_data = test_data["y_data"].unsqueeze(-1)

    y_m, y_s = network.predict(x_data)

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
    test_KBNN_classic()