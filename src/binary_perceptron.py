from typing import Literal
import pandas
import numpy as np
from numpy.typing import NDArray

from src.simple_perceptron import SimplePerceptron

class BinaryPerceptron(SimplePerceptron):
    def __init__(self, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs=1000) -> None:
        super().__init__(dataset, learn_rate, max_epochs)

    def has_next(self):
        return self.error != 0 and self.current_epoch < self.max_epochs

    def _calc_error_per_data(self, delta: float) -> float:
        return abs(delta)

    def _calc_weighted_sum(self, inputs: np.ndarray) -> Literal[-1, 1]:
        output_raw = np.dot(self.weights, inputs)
        return 1 if output_raw >= 0 else -1

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs