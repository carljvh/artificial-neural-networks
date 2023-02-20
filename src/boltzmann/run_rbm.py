from restricted_boltzmann_machine import RestrictedBoltzmannMachine
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-whitegrid")
RESULT_PATH = Path(__file__).parent.resolve().joinpath("results")


def plot_kbl_divergence(n_hidden_span: list, dkl_values: list):
    plt.plot(n_hidden_span, dkl_values, marker="o")
    M_span = [1, 2, 3, 4, 8]
    dkl_theoretical = calculate_theoretical_dkl(M_span, N=3)
    plt.plot(M_span, dkl_theoretical, marker="o")

    plt.title("Kullback-Liebler divergence -- XOR dataset")
    plt.xlabel("M: number of hidden neurons")
    plt.ylabel("$\mathregular{D_{KL}}$")
    plt.legend(["Emprical values", "Theoretical values"])
    plt.savefig(RESULT_PATH.joinpath("kbl_plot.png"))
    plt.show()


def calculate_theoretical_dkl(M_span: list, N: int) -> list:
    dkl_theoretical = []
    for M in M_span:
        if M < 2 ** (N - 1) - 1:
            dkl = np.log(2) * (N - int(np.log2(M + 1)) - (M + 1) / (2 ** int(np.log2(M + 1))))
            dkl_theoretical.append(dkl)
        else:
            dkl_theoretical.append(0)

    return dkl_theoretical


def calculate_empirical_dkl(dataset: np.ndarray, boltzmann_probabilities: dict) -> int:
    dkl_sum = 0
    for i in range(dataset.shape[0]):
        boltzmann_probability = boltzmann_probabilities[str(dataset[i])]
        if np.abs(boltzmann_probability) > 0.001:
            term = 1 / 4 * (np.log(1 / 4) - np.log(boltzmann_probability))
        else:
            term = 1 / 4 * np.log(1 / 4)
        dkl_sum += term

    return dkl_sum


def parameter_sweep(dataset):
    rbm = RestrictedBoltzmannMachine()
    trial_sizes = [100, 500, 1000]
    n_timesteps = [1500, 1750, 2000]
    learning_rates = [0.005, 0.007, 0.01]
    sweep_results = np.zeros((3, 3, 3), dtype=float)

    for l, n_trials in enumerate(trial_sizes):
        for m, k in enumerate(n_timesteps):
            for n, learning_rate in enumerate(learning_rates):
                rbm.fit(
                    dataset,
                    k=k,
                    n_trials=n_trials,
                    sample_size=20,
                    n_hidden=3,
                    learning_rate=learning_rate,
                )

                boltzmann_probabilities = rbm.evaluate()
                sweep_results[l, m, n] = calculate_empirical_dkl(dataset, boltzmann_probabilities)

    f = open(RESULT_PATH.joinpath("sweep_results.txt"), "w")
    f.write(str(sweep_results))
    f.close()

    l, m, n = np.where(sweep_results == sweep_results.min())
    trial_min = trial_sizes[l[0]]
    k_min = n_timesteps[m[0]]
    lr_min = learning_rates[n[0]]

    return {"n_trials": trial_min, "k": k_min, "learning_rate": lr_min}


def hidden_neurons_sweep(dataset: np.ndarray, n_hidden_span: list, **kwargs) -> list:
    dkl_range = np.zeros((1, 4), dtype=float)
    rbm = RestrictedBoltzmannMachine()
    avg_runs = 5

    for _ in range(avg_runs):
        for i, m in enumerate(n_hidden_span):
            rbm.fit(
                dataset,
                n_hidden=m,
                n_trials=kwargs["n_trials"],
                k=kwargs["k"],
                learning_rate=kwargs["learning_rate"],
                sample_size=kwargs["sample_size"],
            )

            boltzmann_probabilities = rbm.evaluate()
            dkl_range[0, i] += calculate_empirical_dkl(dataset, boltzmann_probabilities)

    return (dkl_range / avg_runs).ravel().tolist()


def run(sweep_parameters: bool = False):
    xor_dataset = np.array([[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]])

    if sweep_parameters:
        params = parameter_sweep(xor_dataset)
        params["sample_size"] = 20

        f = open(RESULT_PATH.joinpath("best_params.txt"), "w")
        f.write(str(params))
        f.close()
    else:
        params = {"n_trials": 500, "k": 2000, "learning_rate": 0.01, "sample_size": 20}

    n_hidden_span = [1, 2, 4, 8]
    dkl_range = hidden_neurons_sweep(xor_dataset, n_hidden_span, **params)

    f = open(RESULT_PATH.joinpath("dkl_values.txt"), "w")
    f.write(str(dkl_range))
    f.close()

    plot_kbl_divergence(n_hidden_span, dkl_range)


# IF WANT TO DO PARAMETER SWEEP, SET run(TRUE)
if __name__ == "__main__":
    run(False)
