import src.KBNN_Class_MAIN as KBNN
import src.UncertaintyQuantificationGauss as VI
import src.UncertaintyQuantificationGaussMix as VI_mix
import src.utils as utils
from src.utils import id
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
    plt.plot(data[0], perceptron_Plot/scaling, label="KBNN prediction", color="b")
    plt.fill_between(data[0], (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.2, color="b")

    # Limit plot and show it
    plt.legend(loc='best')
    plt.ylim(y_1, y_2)
    plt.show()

    return None

def test_VI_classic():
    torch.manual_seed(1)
    dimensions = [1, 20, 1, 1]
    functions = [torch.tanh, torch.tanh, id]
    network = VI.PaperBNN(dimensions, functions)

    x_data = torch.linspace(-3, 3, steps=50)
    y_data = torch.cos(x_data)

    print(network.m)
    network.train(x_data, y_data, epochs=500)
    print(network.m)
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

def generate_data(filename, f, variance, start, end, steps):
    x_data = torch.linspace(start, end, steps=steps)
    y_data = f(x_data) + variance * torch.randn(steps)
    torch.save({
        "x_data": x_data,
        "y_data": y_data
    }, f"Test_data/{filename}.pt")

def open_KBNN_json(filename):
    with open(f"Tests/KBNN/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def open_VI_json(filename):
    with open(f"Tests/VI/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def open_VI_mix_json(filename):
    with open(f"Tests/VI_mix/{filename}.json", "r") as f:
        config = json.load(f)
    return config

def train_KBNN_from_config_as(config, filename):
    # config file has the informations for traingin
    # filname is the name of the final pt file where we save the network

    scaling = config["scaling"]
    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]
    torch.manual_seed(seed)

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
    network_data = torch.load(
        f"Tests/KBNN/{filename}.pt",
        weights_only=False)

    KBNN_network = network_data["network"]
    x_data = network_data["x_data"]
    y_data = network_data["y_data"]
    scaling = network_data["scaling"]

    # Plot data
    plt.plot(x_data, y_data/scaling, '.', label = "data", color="r")

    # Save data to plot the prediction of the BNN
    perceptron_Plot_static = torch.zeros(x_data.size(0), dtype=torch.float32)
    perceptron_Plot = torch.zeros(x_data.size(0), dtype=torch.float32)
    perceptron_Plot_var = torch.zeros(x_data.size(0), dtype=torch.float32)

    for i in range(0, x_data.size(0)):
        perceptron_Plot_static[i] = KBNN_network.staticOutput(x_data[i])
        perceptron_Plot[i] = KBNN_network.meanOutput(x_data[i])
        perceptron_Plot_var[i] = 2 * torch.sqrt(KBNN_network.forwardPass(torch.cat((x_data[i].unsqueeze(0), torch.ones(1)), 0))[len(KBNN_network.dimensions)-1][3])

    # PLot the prediction of the BNN
    plt.plot(x_data, perceptron_Plot/scaling, label="KBNN prediction", color="b")
    plt.fill_between(x_data, (perceptron_Plot-perceptron_Plot_var)/scaling, (perceptron_Plot + perceptron_Plot_var)/scaling, alpha=0.2, color="b")

    # Limit plot and show it
    plt.legend(loc='best')
    # plt.ylim(y_1, y_2)
    plt.show()

def train_VI_from_config_as(config, filename):
    # config file has the informations for traingin
    # filname is the name of the final pt file where we save the network

    dimensions = config["dimensions"]
    functions = [utils.FUNCTIONS[func] for func in config["functions"]]
    seed = config["seed"]
    epochs = config["epochs"]
    # learning_rate = config["learning_rate"]

    data = torch.load(f"Test_data/{config["data_file"]}")
    x_data = data["x_data"]
    y_data = data["y_data"]

    torch.manual_seed(seed)

    network = VI.PaperBNN(dimensions, functions)

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

    plt.plot(x_data, y_data, 'ro', label="cos(x)")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((-1, 1))
    plt.grid(True)
    plt.show()

def train_VI_mix_from_config_as(config, filename):

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

    plt.plot(x_data, y_data, 'ro', label="cos(x)")
    plt.plot(x_data, y_pred, label="Network output")
    plt.fill_between(x_data, y_pred - 2*y_s, y_pred + 2*y_s, alpha=0.5)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim((-1, 1))
    plt.grid(True)
    plt.show()

def batch_train_KBNN(base_json, n_seed):
    config = open_KBNN_json(base_json).copy()

    for seed in range(1, n_seed+1):
        config["seed"] = seed
        train_KBNN_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_KBNN(base_json, n_seed):
    RMSE_values = torch.zeros(n_seed)
    MAE_values = torch.zeros(n_seed)
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
        y_m, y_s = KBNN_network.predict(x_data)
        end = time.time()

        y_pred = y_m.squeeze().detach()
        y_s = y_s.squeeze().detach()
        x_data = x_data.squeeze().detach()
        y_data = y_data.squeeze().detach()

        RMSE_values[seed - 1] = utils.RMSE(y_data, y_pred)
        MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    MAE_mean = torch.mean(MAE_values)
    print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_var = torch.var(RMSE_values)
    print(f"RMSE variance is {RMSE_var}.")
    MAE_var = torch.var(MAE_values)
    print(f"MAE variance is {MAE_var}.")
    NLL_var = torch.var(NLL_values)
    print(f"NLL variance is {NLL_var}.")
    train_var = torch.var(train_time)
    print(f"Training time variance is {train_var}.")
    predict_var = torch.var(predict_time)
    print(f"Prediction time variance is {predict_var}.")

def batch_train_VI(base_json, n_seed):
    config = open_VI_json(base_json).copy()

    for seed in range(1, n_seed+1):
        config["seed"] = seed
        train_VI_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_VI(base_json, n_seed):
    RMSE_values = torch.zeros(n_seed)
    MAE_values = torch.zeros(n_seed)
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
        MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    MAE_mean = torch.mean(MAE_values)
    print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_var = torch.var(RMSE_values)
    print(f"RMSE variance is {RMSE_var}.")
    MAE_var = torch.var(MAE_values)
    print(f"MAE variance is {MAE_var}.")
    NLL_var = torch.var(NLL_values)
    print(f"NLL variance is {NLL_var}.")
    train_var = torch.var(train_time)
    print(f"Training time variance is {train_var}.")
    predict_var = torch.var(predict_time)
    print(f"Prediction time variance is {predict_var}.")

def batch_train_VI_mix(base_json, n_seed):
    config = open_VI_mix_json(base_json).copy()

    for seed in range(1, n_seed+1):
        config["seed"] = seed
        train_VI_mix_from_config_as(config, f"{base_json}_seed_{seed}")

def review_batch_VI_mix(base_json, n_seed):
    RMSE_values = torch.zeros(n_seed)
    MAE_values = torch.zeros(n_seed)
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
        MAE_values[seed - 1] = utils.MAE(y_data, y_pred)
        NLL_values[seed - 1] = utils.NLL(y_data, y_pred, y_s)
        predict_time[seed - 1] = end - start
        train_time[seed - 1] = network_data["train_time"]

    RMSE_mean = torch.mean(RMSE_values)
    print(f"RMSE mean is {RMSE_mean}.")
    MAE_mean = torch.mean(MAE_values)
    print(f"MAE mean is {MAE_mean}.")
    NLL_mean = torch.mean(NLL_values)
    print(f"NLL mean is {NLL_mean}.")
    train_mean = torch.mean(train_time)
    print(f"Training time mean is {train_mean}.")
    predict_mean = torch.mean(predict_time)
    print(f"Prediction time mean is {predict_mean}.")

    RMSE_var = torch.var(RMSE_values)
    print(f"RMSE variance is {RMSE_var}.")
    MAE_var = torch.var(MAE_values)
    print(f"MAE variance is {MAE_var}.")
    NLL_var = torch.var(NLL_values)
    print(f"NLL variance is {NLL_var}.")
    train_var = torch.var(train_time)
    print(f"Training time variance is {train_var}.")
    predict_var = torch.var(predict_time)
    print(f"Prediction time variance is {predict_var}.")


if __name__ == "__main__":
    print("INITIATE TESTS:")
    #batch_train_VI_mix("Trial", 3)
    #review_batch_VI_mix("Trial", 3)
    test_VI_classic()