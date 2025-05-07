from enum import Enum
import pandas
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from src.simple_perceptron import SimplePerceptron
import math

class NonLinearFunction(Enum):
    HYPERBOLIC = (
        lambda x, beta: math.tanh(beta * x),
        lambda x, beta: beta * (1 - math.tanh(beta * x) ** 2)
    )
    LOGISTICS = (
        lambda x, beta: 1 / (1 + math.exp(-beta * x)),
        lambda x, beta: (lambda fx: beta * fx * (1 - fx))
    )

    def __init__(self, func: Callable[[float, float], float], deriv_func_raw: Callable):
        self._func = func
        self._deriv_func_raw = deriv_func_raw

    def func(self, x: float, beta: float) -> float:
        return self._func(x, beta)

    def deriv(self, x: float, beta: float) -> float:
        if self is NonLinearFunction.LOGISTICS:
            fx = self._func(x, beta)
            return self._deriv_func_raw(x, beta)(fx)
        return self._deriv_func_raw(x, beta)


class NonLinearPerceptron(SimplePerceptron):
    def __init__(self, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs=1000, sigmoid_func: NonLinearFunction = NonLinearFunction.HYPERBOLIC, beta: float = 0.1) -> None:
        self.func = sigmoid_func
        self.beta = beta
        super().__init__(dataset, learn_rate, max_epochs)

    def has_next(self):
        return np.abs(self.error) > 0.0001 and self.current_epoch < self.max_epochs
        
    def next_epoch(self) -> NDArray[np.float64]:
        to_return = super().next_epoch()
        self.error /= 2
        return to_return
    
    def _calc_error_per_data(self, delta: float):
        return delta**2

    def _calc_weighted_sum(self, inputs: np.ndarray) -> float:
        return self.func.func(np.dot(self.weights, inputs), self.beta)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs * self.func.deriv(np.dot(self.weights, inputs), self.beta)