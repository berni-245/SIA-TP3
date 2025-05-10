import pandas
import numpy as np
from numpy.typing import NDArray

from src.perceptron_function import PerceptronFunction
from src.simple_perceptron import SimplePerceptron

class UniformPerceptron(SimplePerceptron):
    def __init__(
            self,
            dataset: pandas.DataFrame,
            learn_rate: float = 0.1,
            max_epochs=1000,
            random_weight_initialize: bool = True,
            activation_func: PerceptronFunction = PerceptronFunction.HYPERBOLIC,
            beta: float = 0.1,
            min_error: float = 0.0001,
        ) -> None: 
        self.activation_func = activation_func
        self.beta = beta
        self.min_error = min_error
        super().__init__(dataset, learn_rate, max_epochs, random_weight_initialize)

    def has_next(self):
        return np.abs(self.error) > self.min_error and self.current_epoch < self.max_epochs
    
    def _calc_error_per_data(self, delta: float):
        return (delta**2) / 2

    def _activation_func(self, weighted_sum: np.float64) -> float:
        return self.activation_func.func(weighted_sum, self.beta)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs * self.activation_func.deriv(np.dot(self.weights, inputs), self.beta)
