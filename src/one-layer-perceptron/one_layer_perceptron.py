import numpy as np
from tqdm import tqdm


class OneLayerPerceptron:
    def __init__(self):
        self.n_visible = None
        self.n_hidden = None
        self.hidden_weights = None
        self.output_weights = None
        self.hidden_thresholds = None
        self.output_threshold = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 120,
        n_hidden: int = 50,
        learning_rate: float = 0.03,
    ) -> None:
        self.n_visible = X.shape[1]
        self.n_hidden = n_hidden

        hidden_weights = np.random.normal(
            0, 1 / np.sqrt(self.n_visible), size=(self.n_hidden, self.n_visible)
        )
        output_weights = np.random.normal(0, 1 / np.sqrt(self.n_hidden), size=self.n_hidden)
        hidden_thresholds = np.zeros(self.n_hidden)
        output_threshold = 0

        for _ in tqdm(range(n_epochs)):
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
            for mu in indices:
                initial_states = X[mu]

                hidden_local_field = np.dot(hidden_weights, initial_states) - hidden_thresholds
                hidden_states = np.tanh(hidden_local_field)

                output_local_field = np.dot(output_weights, hidden_states) - output_threshold
                output_state = np.tanh(output_local_field)

                output_error = (1 - np.tanh(output_local_field) ** 2) * (y[mu] - output_state)
                hidden_layer_errors = output_error * np.multiply(
                    output_weights, (1 - np.tanh(hidden_local_field) ** 2)
                )

                hidden_weights += learning_rate * np.outer(hidden_layer_errors, initial_states)
                output_weights += learning_rate * output_error * hidden_states
                hidden_thresholds -= learning_rate * hidden_layer_errors
                output_threshold -= learning_rate * output_error

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_thresholds = hidden_thresholds
        self.output_threshold = output_threshold

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        p_val = X.shape[0]
        output_states = np.zeros((1, p_val))
        for mu in range(p_val):
            initial_states = X[mu]

            hidden_states = self.forward_prop(self.hidden_weights, initial_states, self.hidden_thresholds)
            output_states[0, mu] = self.forward_prop(
                self.output_weights, hidden_states, self.output_threshold
            )

        C = 1 / (2 * p_val) * np.sum(np.abs(np.sign(output_states) - y))
        return C

    @staticmethod
    def forward_prop(weights: np.ndarray, input: np.ndarray, thresholds):
        b = np.dot(weights, input) - thresholds
        return np.tanh(b)
