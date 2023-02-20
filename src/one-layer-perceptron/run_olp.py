import numpy as np
import pandas as pd
from pathlib import Path
from one_layer_perceptron import OneLayerPerceptron

DATA_PATH = Path(__file__).parent.resolve().joinpath("data")
RESULT_PATH = Path(__file__).parent.resolve().joinpath("results")


def load_data():
    train_df = pd.read_csv(DATA_PATH.joinpath("training_set.csv"))
    validation_df = pd.read_csv(DATA_PATH.joinpath("validation_set.csv"))

    x_train = train_df.iloc[:, [0, 1]].to_numpy()
    y_train = train_df.iloc[:, 2].to_numpy()

    x_valid = validation_df.iloc[:, [0, 1]].to_numpy()
    y_valid = validation_df.iloc[:, 2].to_numpy()

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std

    return x_train, y_train, x_valid, y_valid


def run():
    x_train, y_train, x_valid, y_valid = load_data()
    olp = OneLayerPerceptron()
    olp.fit(x_train, y_train, n_epochs=3000, n_hidden=50, learning_rate=0.005)
    C = olp.evaluate(x_valid, y_valid)

    f = open(RESULT_PATH.joinpath("results.txt"), "w")
    f.write("hidden_min: %s, learning_min: %s, C: %s" % (50, 0.005, C))
    f.close()

    params = {
        "w1.csv": olp.hidden_weights,
        "t1.csv": olp.hidden_thresholds,
        "w2.csv": olp.output_weights,
        "t2.csv": olp.output_threshold,
    }
    for filename, param in params.items():
        if param.shape != ():
            df = pd.DataFrame(data=param)
            df.to_csv(RESULT_PATH.joinpath(filename), sep=",", index=False, header=False)
        else:
            data = np.empty(shape=(1, 1))
            data[0, 0] = param
            df = pd.DataFrame(data=data)
            df.to_csv(RESULT_PATH.joinpath(filename), sep=",", index=False, header=False)


if __name__ == "__main__":
    run()
