from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Tuple, Union


class PerceptronFunction(Enum):
    HYPERBOLIC = (
        lambda x, beta: np.tanh(beta * x),
        lambda x, beta: beta * (1 - np.tanh(beta * x) ** 2),
        (-1, 1),
    )
    LOGISTICS = (
        lambda x, beta: 1 / (1 + np.exp(-2 * beta * x)),
        lambda x, beta: 2 * beta * (1 / (1 + np.exp(-2 * beta * x))) * (1 - (1 / (1 + np.exp(-2 * beta * x)))),
        (0, 1),
    )
    LINEAR = (
        lambda x, beta: x,
        lambda x, beta: 1,
        None,
    )

    def __init__(
        self,
        func: Callable[[Union[float, NDArray[np.float64]], float], Union[float, NDArray[np.float64]]],
        deriv: Callable[[Union[float, NDArray[np.float64]], float], Union[float, NDArray[np.float64]]],
        image: Tuple[float, float] | None,
    ):
        self._func = func
        self._deriv = deriv
        self.image = image

    def func(self, x: Union[float, NDArray[np.float64]], beta: float) -> Any:
        return self._func(x, beta)

    def deriv(self, x: Union[float, NDArray[np.float64]], beta: float) -> Any:
        return self._deriv(x, beta)
    
    @classmethod
    def from_string(cls, name: str):
        return cls[name.upper()]
