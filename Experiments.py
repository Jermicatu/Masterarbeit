import src.KBNN_Class_MAIN as KBNN
import src.UncertaintyQuantificationGauss as VI
import src.UncertaintyQuantificationGaussMix as VI_mix
import src.utils as utils
from src.utils import id
import random
import torch
import matplotlib.pyplot as plt
import time
import json



# TODO this does not work anymore - delete soon tm
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

    # Save data to plot the prediction of the BNN
    perceptron_Plot_static = torch.zeros(data[0].size(0), dtype=torch.float32)
    perceptron_Plot = torch.zeros(data[0].size(0), dtype=torch.float32)
    perceptron_Plot_var = torch.zeros(data[0].size(0), dtype=torch.float32)

    for i in range(0, data[0].size(0)):
        perceptron_Plot_static[i] = network.staticOutput(data[0][i])
        perceptron_Plot[i] = network.meanOutput(data[0][i])
        perceptron_Plot_var[i] = 2 * torch.sqrt(network.forwardPass(torch.cat((data[0][i].unsqueeze(0), torch.ones(1)), 0))[len(network.dimensions)-1][3])

    # PLot the prediction of the BNN
    plt.plot(data[0], perceptron_Plot/scaling, label="KBNN prediction")
    plt.fill_between(data[0], (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.5)

    # Limit plot and show it
    plt.legend(loc='best')
    plt.ylim(y_1, y_2)
    plt.show()

def test_VI_classic():
    torch.manual_seed(2)
    dimensions = [1, 10, 1, 1]
    functions = [torch.tanh, torch.tanh, id]
    network = VI.VI_BNN(dimensions, functions)

    x_data = torch.linspace(-3, 3, steps=20)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data, epochs=500)

    y_m, y_s = network.predict(x_data)
    
    y_pred = y_m.squeeze(0)
    y_s = y_s.squeeze(0)

    plt.plot(x_data, y_data, 'ro', label="data")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)


    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((-1, 1))
    plt.grid(True)
    plt.show()

def test_VI_mix_classic():
    torch.manual_seed(2)
    dimensions = [1, 20, 1, 1]
    functions = [torch.tanh, torch.tanh, lambda x: x]
    network = VI_mix.myMixBNN(dimensions, functions, mix_size=1)


    x_data = torch.linspace(-3, 3, steps=20).unsqueeze(-1)
    y_data = torch.cos(x_data)

    network.train(x_data, y_data)
    y_m, y_s = network.predict(x_data)

    y_pred = y_m.squeeze().detach().numpy()
    y_s = y_s.squeeze().detach().numpy()
    x_data = x_data.squeeze().detach().numpy()
    y_data = y_data.detach().numpy()

    plt.plot(x_data, y_data, 'ro', label="data")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)

    print(network.w)


    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((-1, 1))
    plt.grid(True)
    plt.show()

def test_function(x):
    return torch.cos(2*x)

def generate_data(filename, f, variance, start, end, steps):
    """Saves a dataset in the specified range

    Args:
        filename (string): Name of the resulting .pt file
        f (function): any function that takes a torch tensor as input and outputs a torch tensor
        variance (float): variance of the data noise, a variance of 0 leads to perfect data
        start (float): start of the inteval
        end (float): end of the interval
        steps (int): number of points in the resulting dataset
    """
    x_data = torch.linspace(start, end, steps=steps)
    y_data = f(x_data) + variance * torch.randn(steps)
    torch.save({
        "x_data": x_data,
        "y_data": y_data
    }, f"Test_data/{filename}.pt")

def open_KBNN_json(filename):
    """opens the named json in the KBNN folder and returns the configuration

    Args:
        filename (string): name of the json file we want to open in the KBNN folder

    Returns:
        dict: containing all variables needed to initialize a KBNN
    """
    with open(f"Tests/KBNN/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def open_VI_json(filename):
    """opens the named json in the VI folder and returns the configuration

    Args:
        filename (string): name of the json file we want to open in the VI folder

    Returns:
        dict: containing all variables needed to initialize a VI BNN
    """
    with open(f"Tests/VI/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def open_VI_mix_json(filename):
    """opens the named json in the VI_mix folder and returns the configuration

    Args:
        filename (string): name of the json file we want to open in the VI_mix folder

    Returns:
        dict: containing all variables needed to initialize a VI mix BNN
    """
    with open(f"Tests/VI_mix/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def train_KBNN_from_config_as(config, filename):
    """A given config dictionary specifies a dataset and a KBNN. 
    Train this KBNN on the dataset and save it under the given filename.

    Args:
        config (dict): A dictionary describing the KBNN parameters and a data set file. Usually taken from a relevant json file.
        filename (string): Name of the resulting file
    """

    scaling = config["scaling"]
    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]

    # As this network uses the random package we also set its seed, the other networks only require the torch seed
    torch.manual_seed(seed)
    random.seed(seed)

    data = torch.load(f"Test_data/{config["data_file"]}")
    x_data = data["x_data"]
    y_data = data["y_data"]

    net_var = 1

    network = KBNN.Network_Class(dimensions[0], dimensions[1:], functions, net_var)

    start = time.time()
    network.train([x_data, y_data*scaling])
    end = time.time()

    # Here we save only the variables relevant for ploting later
    torch.save({
        "network": network,
        "x_data": x_data,
        "y_data": y_data,
        "scaling": scaling,
        "train_time": end - start}, 
    f"Tests/KBNN/{filename}.pt")

def plot_KBNN_from_pt(filename):
    """Plots the given KBNN

    Args:
        filename (string): Name of the pt filename whose KBNN we want to plot
    """
    network_data = torch.load(
        f"Tests/KBNN/{filename}.pt",
        weights_only=False)

    KBNN_network = network_data["network"]
    x_data = network_data["x_data"]
    y_data = network_data["y_data"]
    scaling = network_data["scaling"]

    # Plot data
    plt.plot(x_data, y_data, '.', label = "data", color="r")

    # Save data to plot the prediction of the BNN
    perceptron_Plot_static = torch.zeros(x_data.size(0), dtype=torch.float32)
    perceptron_Plot = torch.zeros(x_data.size(0), dtype=torch.float32)
    perceptron_Plot_var = torch.zeros(x_data.size(0), dtype=torch.float32)

    for i in range(0, x_data.size(0)):
        perceptron_Plot_static[i] = KBNN_network.staticOutput(x_data[i])
        perceptron_Plot[i] = KBNN_network.meanOutput(x_data[i])
        perceptron_Plot_var[i] = 2 * torch.sqrt(KBNN_network.forwardPass(torch.cat((x_data[i].unsqueeze(0), torch.ones(1)), 0))[len(KBNN_network.dimensions)-1][3])

    # PLot the prediction of the BNN
    plt.plot(x_data, perceptron_Plot/scaling, label="KBNN prediction")
    plt.fill_between(x_data, (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.5)

    # Limit plot and show it
    plt.legend(loc='best')
    plt.ylim((-1.1, 1.1))
    plt.grid(True)
    plt.show()

def train_VI_from_config_as(config, filename):
    """A given config dictionary specifies a dataset and a VI BNN. 
    Train this VI BNN on the dataset and save it under the given filename.

    Args:
        config (dict): A dictionary describing the VI BNN parameters and a data set file. Usually taken from a relevant json file.
        filename (string): Name of the resulting file
    """

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]
    epochs = config["epochs"]
    # learning_rate = config["learning_rate"]

    data = torch.load(f"Test_data/{config["data_file"]}")
    x_data = data["x_data"]
    y_data = data["y_data"]

    torch.manual_seed(seed)

    network = VI.VI_BNN(dimensions, functions)

    start = time.time()
    network.train(x_data, y_data, epochs)
    end = time.time()

    torch.save({
        "network": network,
        "x_data": x_data,
        "y_data": y_data,
        "train_time": end - start
    }, f"Tests/VI/{filename}.pt")

def plot_VI_from_pt(filename):
    """Plots the given VI BNN

    Args:
        filename (string): Name of the pt filename whose VI BNN we want to plot
    """

    network_data = torch.load(
        f"Tests/VI/{filename}.pt",
        weights_only=False)

    VI_network = network_data["network"]
    x_data = network_data["x_data"]
    y_data = network_data["y_data"]

    y_m, y_s = VI_network.predict(x_data)

    y_pred = y_m.squeeze().detach().numpy()
    y_s = y_s.squeeze().detach().numpy()
    x_data = x_data.squeeze().detach().numpy()
    y_data = y_data.detach().numpy()

    plt.plot(x_data, y_data, 'ro', label="data")
    #plt.plot(x_data, y_pred, label="Network output")
    #plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim((-1.1, 1.1))
    plt.grid(True)
    plt.show()

def train_VI_mix_from_config_as(config, filename):
    """A given config dictionary specifies a dataset and a VI mix BNN. 
    Train this VI mix BNN on the dataset and save it under the given filename.

    Args:
        config (dict): A dictionary describing the VI mix BNN parameters and a data set file. Usually taken from a relevant json file.
        filename (string): Name of the resulting file
    """

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    mix_size = config["mix_size"]
    seed = config["seed"]
    epochs = config["epochs"]

    data = torch.load(f"Test_data/{config["data_file"]}")
    x_data = data["x_data"].unsqueeze(-1)
    y_data = data["y_data"].unsqueeze(-1)

    torch.manual_seed(seed)

    network = VI_mix.myMixBNN(dimensions, functions, mix_size)

    start = time.time()
    network.train(x_data, y_data, epochs)
    end = time.time()

    torch.save({
        "network": network,
        "x_data": x_data,
        "y_data": y_data,
        "train_time": end - start
    }, f"Tests/VI_mix/{filename}.pt")

def plot_VI_mix_from_pt(filename):
    """Plots the given VI mix BNN

    Args:
        filename (string): Name of the pt filename whose VI mix BNN we want to plot
    """

    network_data = torch.load(
        f"Tests/VI_mix/{filename}.pt",
        weights_only=False)

    VI_mix_network = network_data["network"]
    x_data = network_data["x_data"]
    y_data = network_data["y_data"]

    y_m, y_s = VI_mix_network.predict(x_data)

    y_pred = y_m.squeeze().detach().numpy()
    y_s = y_s.squeeze().detach().numpy()
    x_data = x_data.squeeze().detach().numpy()
    y_data = y_data.detach().numpy()

    plt.plot(x_data, y_data, 'ro', label="data")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim((-1.1, 1.1))
    plt.grid(True)
    plt.show()

def batch_train_seed_KBNN(base_json, n_seed):
    """Trains the same KBNN over n seeds and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our KBNN and its training data
        n_seed (int): number of seeds we want to train
    """
    config = open_KBNN_json(base_json).copy()

    for seed in range(1, n_seed+1):
        print("KBNN seed: ", seed)
        config["seed"] = seed
        train_KBNN_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_seed_KBNN(base_json, n_seed):
    """Takes the n trained KBNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_seed_KBNN.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our KBNN and its training data
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for seed in range(1, n_seed+1):
        network_data = torch.load(
            f"Tests/KBNN/{base_json}_seed_{seed}.pt",
            weights_only=False)

        KBNN_network = network_data["network"]
        x_data = network_data["x_data"]
        y_data = network_data["y_data"]

        start = time.time()
        y_m = torch.zeros(x_data.size(0), dtype=torch.float32)
        y_s = torch.zeros(x_data.size(0), dtype=torch.float32)

        for i in range(0, x_data.size(0)):
            y_m[i] = KBNN_network.meanOutput(x_data[i])
            y_s[i] = 2 * torch.sqrt(KBNN_network.forwardPass(torch.cat((x_data[i].unsqueeze(0), torch.ones(1)), 0))[len(KBNN_network.dimensions)-1][3])
        end = time.time()

        y_pred = y_m.squeeze().detach()
        y_s = y_s.squeeze().detach()
        x_data = x_data.squeeze().detach()
        y_data = y_data.squeeze().detach()

        RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
        # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    # MAE_mean = torch.mean(MAE_values)
    # print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_std = torch.sqrt(torch.var(RMSE_values))
    print(f"RMSE standard derivation is {RMSE_std}.")
    # MAE_std = torch.sqrt(torch.var(MAE_values))
    # print(f"MAE standard derivation is {MAE_std}.")
    NLL_std = torch.sqrt(torch.var(NLL_values))
    print(f"NLL standard derivation is {NLL_std}.")
    train_std = torch.sqrt(torch.var(train_time))
    print(f"Training time standard derivation is {train_std}.")
    predict_std = torch.sqrt(torch.var(predict_time))
    print(f"Prediction time standard derivation is {predict_std}.")

def batch_train_seed_VI(base_json, n_seed):
    """Trains the same VI BNN over n seeds and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our VI BNN and its training data
        n_seed (int): number of seeds we want to train
    """
    config = open_VI_json(base_json).copy()

    for seed in range(1, n_seed+1):
        print("VI seed: ", seed)
        config["seed"] = seed
        train_VI_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_seed_VI(base_json, n_seed):
    """Takes the n trained VI BNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_seed_VI.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our VI BNN and its training data
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for seed in range(1, n_seed+1):
        network_data = torch.load(
            f"Tests/VI/{base_json}_seed_{seed}.pt",
            weights_only=False)

        VI_network = network_data["network"]
        x_data = network_data["x_data"]
        y_data = network_data["y_data"]

        start = time.time()
        y_m, y_s = VI_network.predict(x_data)
        end = time.time()

        y_pred = y_m.squeeze().detach()
        y_s = y_s.squeeze().detach()
        x_data = x_data.squeeze().detach()
        y_data = y_data.squeeze().detach()

        RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
        # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    # MAE_mean = torch.mean(MAE_values)
    # print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_std = torch.sqrt(torch.var(RMSE_values))
    print(f"RMSE standard derivation is {RMSE_std}.")
    # MAE_std = torch.sqrt(torch.var(MAE_values))
    # print(f"MAE standard derivation is {MAE_std}.")
    NLL_std = torch.sqrt(torch.var(NLL_values))
    print(f"NLL standard derivation is {NLL_std}.")
    train_std = torch.sqrt(torch.var(train_time))
    print(f"Training time standard derivation is {train_std}.")
    predict_std = torch.sqrt(torch.var(predict_time))
    print(f"Prediction time standard derivation is {predict_std}.")

def batch_train_seed_VI_mix(base_json, n_seed):
    """Trains the same VI mix BNN over n seeds and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our VI mix BNN and its training data
        n_seed (int): number of seeds we want to train
    """
    config = open_VI_mix_json(base_json).copy()

    for seed in range(1, n_seed+1):
        print("VI mix seed: ", seed)
        config["seed"] = seed
        train_VI_mix_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_seed_VI_mix(base_json, n_seed):
    """Takes the n trained VI mix BNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_seed_VI_mix.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our VI mix BNN and its training data
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for seed in range(1, n_seed+1):
        network_data = torch.load(
            f"Tests/VI_mix/{base_json}_seed_{seed}.pt",
            weights_only=False)

        VI_mix_network = network_data["network"]
        x_data = network_data["x_data"]
        y_data = network_data["y_data"]

        start = time.time()
        y_m, y_s = VI_mix_network.predict(x_data)
        end = time.time()

        y_pred = y_m.squeeze().detach()
        y_s = y_s.squeeze().detach()
        x_data = x_data.squeeze().detach()
        y_data = y_data.squeeze().detach()

        RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
        # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    # MAE_mean = torch.mean(MAE_values)
    # print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_std = torch.sqrt(torch.var(RMSE_values))
    print(f"RMSE standard derivation is {RMSE_std}.")
    # MAE_std = torch.sqrt(torch.var(MAE_values))
    # print(f"MAE standard derivation is {MAE_std}.")
    NLL_std = torch.sqrt(torch.var(NLL_values))
    print(f"NLL standard derivation is {NLL_std}.")
    train_std = torch.sqrt(torch.var(train_time))
    print(f"Training time standard derivation is {train_std}.")
    predict_std = torch.sqrt(torch.var(predict_time))
    print(f"Prediction time standard derivation is {predict_std}.")

def batch_train_epoch_VI_mix(base_json, n_epochs, n_seed):
    """Trains the same VI mix BNN over n seeds and n_epochs (with 50 steps between) and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our VI mix BNN and its training data
        n_epochs (int): number of epchos steps we want to train, each step is 50 epochs
        n_seed (int): number of seeds we trained
    """
    config = open_VI_mix_json(base_json).copy()

    for epochs in range(1, n_epochs+1):
        for seed in range(1, n_seed+1):
            print("VI mix seed: ", seed)
            print("VI mix epochs: ", epochs*50)
            config["seed"] = seed
            config["epochs"] = epochs*50
            train_VI_mix_from_config_as(config, f"{base_json}_epochs_{epochs*50}_seed_{seed}")

def review_batch_epoch_VI_mix(base_json, n_epochs, n_seed):
    """Takes the n trained VI mix BNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_epoch_VI_mix.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our VI mix BNN and its training data
        n_epochs (int): number of epchos steps we want to train, each step is 50 epochs
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for epochs in range(1, n_epochs+1):
        for seed in range(1, n_seed+1):
            network_data = torch.load(
                f"Tests/VI_mix/{base_json}_epochs_{epochs*50}_seed_{seed}.pt",
                weights_only=False)

            VI_mix_network = network_data["network"]
            x_data = network_data["x_data"]
            y_data = network_data["y_data"]

            start = time.time()
            y_m, y_s = VI_mix_network.predict(x_data)
            end = time.time()

            y_pred = y_m.squeeze().detach()
            y_s = y_s.squeeze().detach()
            x_data = x_data.squeeze().detach()
            y_data = y_data.squeeze().detach()

            RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
            # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
            NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
            predict_time[seed - 1] = end - start
            train_time[seed - 1] = network_data["train_time"]

        print(f"Results for {50*epochs} epochs:")

        RMSE_mean = torch.mean(RMSE_values)
        print(f"RMSE mean is {RMSE_mean}.")
        # MAE_mean = torch.mean(MAE_values)
        # print(f"MAE mean is {MAE_mean}.")
        NLL_mean = torch.mean(NLL_values)
        print(f"NLL mean is {NLL_mean}.")
        train_mean = torch.mean(train_time)
        print(f"Training time mean is {train_mean}.")
        predict_mean = torch.mean(predict_time)
        print(f"Prediction time mean is {predict_mean}.")

        RMSE_std = torch.sqrt(torch.var(RMSE_values))
        print(f"RMSE standard derivation is {RMSE_std}.")
        # MAE_std = torch.sqrt(torch.var(MAE_values))
        # print(f"MAE standard derivation is {MAE_std}.")
        NLL_std = torch.sqrt(torch.var(NLL_values))
        print(f"NLL standard derivation is {NLL_std}.")
        train_std = torch.sqrt(torch.var(train_time))
        print(f"Training time standard derivation is {train_std}.")
        predict_std = torch.sqrt(torch.var(predict_time))
        print(f"Prediction time standard derivation is {predict_std}.")

def batch_train_layer_size_VI_mix(base_json, n_layer_size, n_seed):
    """Trains the same VI mix BNN over n seeds and n_layer_size (with 1 being skiped) and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our VI mix BNN and its training data
        n_layer_size (int): number of layer size steps we want to train, 1 is skiped
        n_seed (int): number of seeds we trained
    """
    config = open_VI_mix_json(base_json).copy()

    for layer_size in range(2, n_layer_size+1):
        for seed in range(1, n_seed+1):
            print("VI mix seed: ", seed)
            print("VI mix layer size: ", layer_size)
            config["seed"] = seed
            config["dimensions"] = [1, layer_size, 1, 1]
            train_VI_mix_from_config_as(config, f"{base_json}_layer_size_{layer_size}_seed_{seed}")

def review_batch_layer_size_VI_mix(base_json, n_layer_size, n_seed):
    """Takes the n trained VI mix BNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_layer_size_VI_mix.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our VI mix BNN and its training data
        n_layer_size (int): number of layer size steps we want to train, 1 is skiped
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for layer_size in range(2, n_layer_size+1):
        for seed in range(1, n_seed+1):
            network_data = torch.load(
                f"Tests/VI_mix/{base_json}_layer_size_{layer_size}_seed_{seed}.pt",
                weights_only=False)

            VI_mix_network = network_data["network"]
            x_data = network_data["x_data"]
            y_data = network_data["y_data"]

            start = time.time()
            y_m, y_s = VI_mix_network.predict(x_data)
            end = time.time()

            y_pred = y_m.squeeze().detach()
            y_s = y_s.squeeze().detach()
            x_data = x_data.squeeze().detach()
            y_data = y_data.squeeze().detach()

            RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
            # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
            NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
            predict_time[seed - 1] = end - start
            train_time[seed - 1] = network_data["train_time"]

        print(f"Results for layer size {layer_size}:")

        RMSE_mean = torch.mean(RMSE_values)
        print(f"RMSE mean is {RMSE_mean}.")
        # MAE_mean = torch.mean(MAE_values)
        # print(f"MAE mean is {MAE_mean}.")
        NLL_mean = torch.mean(NLL_values)
        print(f"NLL mean is {NLL_mean}.")
        train_mean = torch.mean(train_time)
        print(f"Training time mean is {train_mean}.")
        predict_mean = torch.mean(predict_time)
        print(f"Prediction time mean is {predict_mean}.")

        RMSE_std = torch.sqrt(torch.var(RMSE_values))
        print(f"RMSE standard derivation is {RMSE_std}.")
        # MAE_std = torch.sqrt(torch.var(MAE_values))
        # print(f"MAE standard derivation is {MAE_std}.")
        NLL_std = torch.sqrt(torch.var(NLL_values))
        print(f"NLL standard derivation is {NLL_std}.")
        train_std = torch.sqrt(torch.var(train_time))
        print(f"Training time standard derivation is {train_std}.")
        predict_std = torch.sqrt(torch.var(predict_time))
        print(f"Prediction time standard derivation is {predict_std}.")

def batch_train_mix_size_VI_mix(base_json, n_mix_size, n_seed):
    """Trains the same VI mix BNN over n seeds and n_mix_size and saves them individually.

    Args:
        base_json (string): The name of the json file whose confifuration we use to define our VI mix BNN and its training data
        n_mix_size (int): number of mix size steps we want to train
        n_seed (int): number of seeds we trained
    """
    config = open_VI_mix_json(base_json).copy()

    for mix_size in range(4, n_mix_size+1):
        for seed in range(1, n_seed+1):
            print("VI mix seed: ", seed)
            print("VI mix size: ", mix_size)
            config["seed"] = seed
            config["mix_size"] = mix_size
            train_VI_mix_from_config_as(config, f"{base_json}_mix_size_{mix_size}_seed_{seed}")

def review_batch_mix_size_VI_mix(base_json, n_mix_size, n_seed):
    """Takes the n trained VI mix BNN and outputs the mean and standard derivation of multiple important metrics.
    Args are the same as in batch_train_mix_size_VI_mix.

    Args:
        base_json (string): The name of the json file whose confifuration we used to define our VI mix BNN and its training data
        n_mix_size (int): number of mix size steps we want to train
        n_seed (int): number of seeds we trained
    """

    RMSE_values = torch.zeros(n_seed)
    # MAE_values = torch.zeros(n_seed)
    NLL_values = torch.zeros(n_seed)
    predict_time = torch.zeros(n_seed)
    train_time = torch.zeros(n_seed)

    for mix_size in range(1, n_mix_size+1):
        for seed in range(1, n_seed+1):
            network_data = torch.load(
                f"Tests/VI_mix/{base_json}_mix_size_{mix_size}_seed_{seed}.pt",
                weights_only=False)

            VI_mix_network = network_data["network"]
            x_data = network_data["x_data"]
            y_data = network_data["y_data"]

            start = time.time()
            y_m, y_s = VI_mix_network.predict(x_data)
            end = time.time()

            y_pred = y_m.squeeze().detach()
            y_s = y_s.squeeze().detach()
            x_data = x_data.squeeze().detach()
            y_data = y_data.squeeze().detach()

            RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
            # MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
            NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
            predict_time[seed - 1] = end - start
            train_time[seed - 1] = network_data["train_time"]

        print(f"Results for mix size {mix_size}:")

        RMSE_mean = torch.mean(RMSE_values)
        print(f"RMSE mean is {RMSE_mean}.")
        # MAE_mean = torch.mean(MAE_values)
        # print(f"MAE mean is {MAE_mean}.")
        NLL_mean = torch.mean(NLL_values)
        print(f"NLL mean is {NLL_mean}.")
        train_mean = torch.mean(train_time)
        print(f"Training time mean is {train_mean}.")
        predict_mean = torch.mean(predict_time)
        print(f"Prediction time mean is {predict_mean}.")

        RMSE_std = torch.sqrt(torch.var(RMSE_values))
        print(f"RMSE standard derivation is {RMSE_std}.")
        # MAE_std = torch.sqrt(torch.var(MAE_values))
        # print(f"MAE standard derivation is {MAE_std}.")
        NLL_std = torch.sqrt(torch.var(NLL_values))
        print(f"NLL standard derivation is {NLL_std}.")
        train_std = torch.sqrt(torch.var(train_time))
        print(f"Training time standard derivation is {train_std}.")
        predict_std = torch.sqrt(torch.var(predict_time))
        print(f"Prediction time standard derivation is {predict_std}.")


if __name__ == "__main__":
    print("INITIATE TESTS:")

    #generate_data("Cos_2x_data_60", test_function, 0, -3, 3, 60)

    
    # batch_train_VI_mix("Trial_60_cos_2x", 1)
    # plot_VI_mix_from_pt("Trial_60_cos_2x_seed_1")

    #review_batch_seed_VI_mix("Trial_60_cos_2x", 1)

    
    #batch_train_seed_KBNN("Trial_60", 100)
    #batch_train_seed_VI("Trial_60", 100)
    #batch_train_seed_VI_mix("Trial_60", 100)
    
    #print("KBNN:")
    #review_batch_seed_KBNN("Trial_60", 100)
    #print("VI:")
    #review_batch_seed_VI("Trial_60", 100)
    #print("VI mix:")
    #review_batch_seed_VI_mix("Trial_60", 100)


    # plot_VI_from_pt("Trial_60_seed_1")
    #plot_KBNN_from_pt("Trial_60_seed_1")
    #plot_VI_mix_from_pt("Trial_60_seed_1")

    # batch_train_layer_size_VI_mix("Trial_60", 10, 10)

    # review_batch_layer_size_VI_mix("Trial_60", 10, 10)

    batch_train_mix_size_VI_mix("Trial_60", 5, 10) # TODO

    #review_batch_epoch_VI_mix("Trial_60", 10, 10)