import pandas
import numpy as np
from numpy.typing import NDArray

from src.perceptron_function import PerceptronFunction
from src.simple_perceptron import SimplePerceptron

class UniformPerceptron(SimplePerceptron):
    def __init__(self, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs=1000, func: PerceptronFunction = PerceptronFunction.HYPERBOLIC, beta: float = 0.1) -> None:
        self.func = func
        self.beta = beta
        super().__init__(dataset, learn_rate, max_epochs)

    def has_next(self):
        return np.abs(self.error) > 0.0001 and self.current_epoch < self.max_epochs
        
    def next_epoch(self) -> NDArray[np.float64]:
        return super().next_epoch()
    
    def _calc_error_per_data(self, delta: float):
        return (delta**2) / 2

    def _activation_func(self, weighted_sum: np.float64) -> float:
        return self.func.func(weighted_sum, self.beta)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs * self.func.deriv(np.dot(self.weights, inputs), self.beta)