from enum import Enum
import math
from typing import Callable


class PerceptronFunction(Enum):
    HYPERBOLIC = (
        lambda x, beta: math.tanh(beta * x),
        lambda x, beta: beta * (1 - math.tanh(beta * x) ** 2)
    )
    LOGISTICS = (
        lambda x, beta: 1 / (1 + math.exp(-beta * x)),
        lambda x, beta: (lambda fx: beta * fx * (1 - fx))
    )
    LINEAR = (
        lambda x, beta: x,
        lambda x, beta: 1
    )

    def __init__(self, func: Callable[[float, float], float], deriv_func_raw: Callable):
        self._func = func
        self._deriv_func_raw = deriv_func_raw

    def func(self, x: float, beta: float) -> float:
        return self._func(x, beta)

    def deriv(self, x: float, beta: float) -> float:
        if self is PerceptronFunction.LOGISTICS:
            fx = self._func(x, beta)
            return self._deriv_func_raw(x, beta)(fx)
        return self._deriv_func_raw(x, beta)