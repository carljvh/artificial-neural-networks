import numpy as np
import pandas as pd
from pathlib import Path
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

DATA_PATH = Path(__file__).parent.resolve().joinpath("data")


def load_mnist():
    X_train = np.load(DATA_PATH.joinpath("train-images-idx3-ubyte.npy"))
    y_train = np.load(DATA_PATH.joinpath("train-labels-idx1-ubyte.npy"))

    X_test = np.load(DATA_PATH.joinpath("valid-images-idx3-ubyte.npy"))
    y_test = np.load(DATA_PATH.joinpath("valid-labels-idx1-ubyte.npy"))

    return (X_train, y_train, X_test, y_test)


def load_evaluation_data():
    return np.load(DATA_PATH.joinpath("xTest2.npy"))


def reshape(X: np.ndarray) -> np.ndarray:
    img_rows, img_cols = 28, 28
    X_new = np.zeros((X.shape[3], img_rows, img_cols, 1), dtype=X.dtype)

    for i in range(X.shape[3]):
        X_new[i, :, :, 0] = X[:, :, 0, i]
    return X_new


X_train, y_train, X_test, y_test = load_mnist()

X_train = reshape(X_train)
X_test = reshape(X_test)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

batch_size = 128
n_classes = 10
epochs = 100

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

## Define model ##
model = Sequential()

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(n_classes, activation="softmax"))


model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr=0.05),
    metrics=["accuracy"],
)

fit_info = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)
score = model.evaluate(X_test, y_test, verbose=0)

print("Test loss: {}, Test accuracy {}".format(score[0], score[1]))

x_test2 = load_evaluation_data()
x_test2 = reshape(x_test2).astype("float32") / 255
result = model.predict(x_test2)
digits = np.argmax(result, axis=1)
df = pd.DataFrame(digits)
df.to_csv(Path(__file__).parent.resolve().joinpath("classifications.csv"), index=False, header=False)
