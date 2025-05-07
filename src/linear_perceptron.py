import pandas
import numpy as np
from numpy.typing import NDArray

from src.simple_perceptron import SimplePerceptron

class LinearPerceptron(SimplePerceptron):
    def __init__(self, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs=1000) -> None:
        super().__init__(dataset, learn_rate, max_epochs)

    def has_next(self):
        return np.abs(self.error) > 0.0001 and self.current_epoch < self.max_epochs
        
    def next_epoch(self) -> NDArray[np.float64]:
        to_return = super().next_epoch()
        self.error /= 2
        return to_return
    
    def _calc_error_per_data(self, delta: float):
        return delta**2

    def _calc_weighted_sum(self, inputs: np.ndarray) -> np.float64:
        return np.dot(self.weights, inputs)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs