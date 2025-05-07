from argparse import ArgumentTypeError
import math
from typing import List, Literal
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
        if not self.has_next():
            raise Exception("Solution was already found or max epochs were reached")
        total_err = 0
        self.current_epoch += 1
        print(f"epoch: {self.current_epoch}")
        for i, row in self.dataset.iterrows():
            inputs = row[self.col_labels].values.astype(float)
            output = self._calc_weighted_sum(inputs)
            delta = row['ev'] - output
            print(f"partial error {i}: {abs(delta)}")
            self.weights += self.learn_rate * delta * inputs
            total_err += delta**2
        
        self.error = total_err/2
        print(f"total error: {self.error}")
        return self.weights

    def _calc_weighted_sum(self, inputs: np.ndarray) -> np.float64:
        return np.dot(self.weights, inputs)

