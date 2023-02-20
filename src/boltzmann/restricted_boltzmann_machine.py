import numpy as np
from tqdm import tqdm


class RestrictedBoltzmannMachine:
    def __init__(self):
        self.n_visible = 3
        self.n_hidden = None
        self.weights = None
        self.visible_thresholds = None
        self.hidden_thresholds = None
        self.binary_dataset = self.generate_boolean_inputs(self.n_visible)

    def fit(
        self,
        X: np.ndarray,
        n_hidden: int = 3,
        n_trials: int = 500,
        k: int = 2000,
        learning_rate: float = 0.01,
        sample_size: int = 20,
    ):
        self.n_hidden = n_hidden

        # create patterns and state vectors
        visible_states = np.zeros((2, self.n_visible), dtype=int)
        hidden_states = np.zeros((2, self.n_hidden), dtype=int)

        # initialize weights and thresholds
        weights = np.random.normal(0, 1 / np.sqrt(self.n_visible), size=(self.n_hidden, self.n_visible))
        visible_thresholds = np.zeros((1, self.n_visible))
        hidden_thresholds = np.zeros((1, self.n_hidden))

        for _ in tqdm(range(n_trials)):
            # sample 'sample_size' number of patterns for weights and thresholds delta calculations
            rows = np.random.choice(range(X.shape[0]), size=(1, sample_size))
            sample = X[rows]

            weights_delta = np.zeros_like(weights)
            visible_thresholds_delta = np.zeros_like(visible_thresholds)
            hidden_thresholds_delta = np.zeros_like(hidden_thresholds)

            for mu in range(sample_size):
                # initialize states
                initial_states = sample[0, mu]
                visible_states[0] = initial_states.copy()

                initial_fields = np.dot(weights, visible_states[0].T) - hidden_thresholds
                hidden_states[0] = self.generate_stochastic_states(initial_fields)

                # Iterate until stochastic dynamics produces a Markov chain in steady state
                for t in range(1, k + 1):
                    visible_local_fields = np.dot(hidden_states[(t - 1) % 2], weights) - visible_thresholds
                    visible_states[t % 2] = self.generate_stochastic_states(visible_local_fields)

                    hidden_local_fields = np.dot(weights, visible_states[t % 2]) - hidden_thresholds
                    hidden_states[t % 2] = self.generate_stochastic_states(hidden_local_fields)

                # Update weights and thresholds by approximated model distribution average
                weights_delta += learning_rate * (
                    np.outer(np.tanh(initial_fields), initial_states)
                    - np.outer(np.tanh(hidden_local_fields), visible_states[k % 2])
                )
                visible_thresholds_delta -= learning_rate * (initial_states - visible_states[k % 2])
                hidden_thresholds_delta -= learning_rate * (
                    np.tanh(initial_fields) - np.tanh(hidden_local_fields)
                )

            weights += weights_delta
            visible_thresholds += visible_thresholds_delta
            hidden_thresholds += hidden_thresholds_delta

        self.weights = weights
        self.visible_thresholds = visible_thresholds
        self.hidden_thresholds = hidden_thresholds

    def evaluate(self, n_outer: int = 3000, n_inner: int = 2000):
        boltzmann_probabilities = {str(state): 0 for state in list(self.binary_dataset)}

        visible_states = np.zeros((2, self.n_visible), dtype=int)
        hidden_states = np.zeros((2, self.n_hidden), dtype=int)

        random_indices = np.random.choice(range(self.binary_dataset.shape[0]), size=n_outer, replace=True)

        for index in tqdm(random_indices):
            initial_states = self.binary_dataset[index]
            visible_states[0] = initial_states.copy()

            hidden_local_fields = np.dot(self.weights, visible_states[0].T) - self.hidden_thresholds
            hidden_states[0] = self.generate_stochastic_states(hidden_local_fields)

            for n in range(1, n_inner + 1):
                visible_local_fields = (
                    np.dot(hidden_states[(n - 1) % 2], self.weights) - self.visible_thresholds
                )
                visible_states[n % 2] = self.generate_stochastic_states(visible_local_fields)

                hidden_local_fields = np.dot(self.weights, visible_states[n % 2]) - self.hidden_thresholds
                hidden_states[n % 2] = self.generate_stochastic_states(hidden_local_fields)

                boltzmann_probabilities[str(visible_states[n % 2].ravel())] += 1 / (n_outer * n_inner)

        return boltzmann_probabilities

    @staticmethod
    def generate_stochastic_states(local_fields: np.ndarray):
        positive_probability = 1 / (1 + np.exp(-2 * local_fields))
        outcome = np.random.random(size=positive_probability.shape[1])
        states = (outcome < positive_probability) * 2 - 1

        return states

    @staticmethod
    def generate_boolean_inputs(n: int):
        m = 2**n
        boolean_inputs = np.zeros((m, n), dtype=int)
        rows = set()
        counter = 0
        while counter < m:
            temp = np.random.choice([-1, 1], size=(m, n))
            for row in temp:
                if tuple(row) not in rows:
                    rows.add(tuple(row))
                    boolean_inputs[counter] = row
                    counter += 1

        return boolean_inputs
