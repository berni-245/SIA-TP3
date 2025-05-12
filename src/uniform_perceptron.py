import pandas
import numpy as np

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
            beta: float = 1,
            min_error: float = 1,
            copy_dataset = False,
        ) -> None: 
        super().__init__(dataset, learn_rate, max_epochs, random_weight_initialize, copy_dataset, min_error)
        self.activation_func = activation_func
        self.beta = beta
        if activation_func.image != None:
            Y = dataset['ev']
            min_ev = np.min(Y)
            max_ev = np.max(Y)
            (min, max) = activation_func.image
            dataset['ev'] = min + ((Y - min_ev)*(max - min))/(max_ev - min_ev)

    def has_next(self):
        return np.abs(self.error) > self.min_error and self.current_epoch < self.max_epochs
    
    def _calc_error_per_data(self, delta: float):
        return (delta**2) / 2

    def _activation_func(self, weighted_sum: np.float64) -> float:
        return self.activation_func.func(weighted_sum, self.beta)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs * self.activation_func.deriv(np.dot(self.weights, inputs), self.beta)


