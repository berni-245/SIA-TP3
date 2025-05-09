from enum import Enum
import math
from typing import Callable


class PerceptronFunction(Enum):
    HYPERBOLIC = (
        lambda x, beta: math.tanh(beta * x),
        lambda x, beta: beta * (1 - math.tanh(beta * x) ** 2)
    )
    LOGISTICS = (
    lambda x, beta: 1 / (1 + math.exp(-2 * beta * x)),
    lambda x, beta: 2 * beta * (1 / (1 + math.exp(-2 * beta * x))) * (1 - (1 / (1 + math.exp(-2 * beta * x))))
    )
    LINEAR = (
        lambda x, beta: x,
        lambda x, beta: 1
    )

    def __init__(self, func: Callable[[float, float], float], deriv: Callable[[float, float], float]):
        self._func = func
        self._deriv = deriv

    def func(self, x: float, beta: float) -> float:
        return self._func(x, beta)

    def deriv(self, x: float, beta: float) -> float:
        return self._deriv(x, beta)