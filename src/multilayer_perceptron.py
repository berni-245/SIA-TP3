from typing import List
import numpy as np
import pandas
from numpy.typing import NDArray

from src.perceptron_function import PerceptronFunction

class NeuralNet:
    def __init__(self, input_count: int, hidden_layers: List[int], activation_func: PerceptronFunction):
        """
        input_count: the amount of input arguments
        hidden_layers: list of the amount of neurons per layer (including output layer)
        activation_func: activation function used in all layers
        """
        self.activation_func = activation_func
        self.weights: List[NDArray[np.float64]] = [] # NDArray can be vector or matrix, in this case matrix

        prev_neuron_count = input_count
        for neurons in hidden_layers:
            weight_matrix = np.random.uniform(-1, 1, (neurons, prev_neuron_count + 1)) # +1 for bias
            self.weights.append(weight_matrix)
            prev_neuron_count = neurons

    def calc_neuron_values(self, input_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        input_values: vector of input data (without bias)
        """
        current_values = input_values
        for weight_matrix in self.weights:
            current_values = np.insert(current_values, 0, 1.0)  # add bias
            z = np.dot(weight_matrix, current_values)
            current_values: NDArray[np.float64] = self.activation_func.func(z, beta=0.1)
        return current_values  # salida final


    def update_weights_per_data(  # TODO ESTE CÓDIGO NO ESTÁ HECHO POR MÍ PERO ESTOY CANSADO, LUEGO LO PIENSO BIEN
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1,
        beta: float = 0.1
    ):
        activations = []  # Salidas de cada capa (después de aplicar activación)
        inputs = []       # Entradas a cada capa (con bias)

        current_values = input_values

        # FORWARD PASS
        for weight_matrix in self.weights:
            current_values = np.insert(current_values, 0, 1.0)  # Add bias
            inputs.append(current_values)
            z = np.dot(weight_matrix, current_values)
            current_values = self.activation_func.func(z, beta)
            activations.append(current_values)

        # BACKWARD PASS
        deltas = []
        error = activations[-1] - expected_output
        delta = error * self.activation_func.deriv(activations[-1], beta)
        deltas.append(delta)

        # Backpropagate deltas (from output to first hidden layer)
        for i in reversed(range(len(self.weights) - 1)):
            weights_wo_bias = self.weights[i + 1][:, 1:]  # Remove bias column
            delta = (weights_wo_bias.T @ deltas[0]) * self.activation_func.deriv(activations[i], beta)
            deltas.insert(0, delta)

        # WEIGHTS UPDATE    
        for i in range(len(self.weights)):
            gradient = np.outer(deltas[i], inputs[i])
            self.weights[i] -= learning_rate * gradient



class MultiLayerPerceptron(): # TODO LUEGO DE TERMINAR EL CÓDIGO DE ARRIBA, HAY QUE INTEGRARLO CON LO DE ACÁ
    def __init__(self, neural_net: NeuralNet, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs = 1000, random_weight_initialize: bool = True) -> None:
        """
        dataset: DataFrame with cols 'x1', 'x2', ..., 'xn' and 'ev' (expected value) as final col

        learn_rate: value between 0 and 1, higher value = bigger steps, usually between 0 and 0.1
        """

        if 'ev' not in dataset.columns:
            raise ValueError("Missing 'ev' column for expected output.")

        input_cols = sorted(
            [col for col in dataset.columns if col.startswith('x')]
        )

        if len(input_cols) <= 0:
            raise ValueError("At least one input column 'x1', ..., 'xn' is required.")

        dataset.insert(0, 'x0', 1) # for the bias
        input_cols = ['x0'] + input_cols

        expected_cols = input_cols + ['ev']
        dataset = dataset[expected_cols] # sorts the dataset cols in the given order

        self.col_labels = input_cols

        
        self.dataset = dataset
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

        self.error = 1
        self.current_epoch = 0
    
    