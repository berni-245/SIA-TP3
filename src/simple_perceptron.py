from abc import ABC, abstractmethod
from typing import Any, List
import pandas
import numpy as np
from numpy.typing import NDArray


class SimplePerceptron(ABC):
    def __init__(self, dataset: pandas.DataFrame, learn_rate: float = 0.1, max_epochs = 1000) -> None:
        """
        dataset: DataFrame with cols 'x1', 'x2', ..., 'xn' and 'ev' (expected value) as final col

        learn_rate: value between 0 and 1, higher value = bigger steps, usually between 0 and 0.1
        """

        if 'ev' not in dataset.columns:
            raise ValueError("Missing 'ev' column for expected output.")

        input_cols = sorted(
            [col for col in dataset.columns if col.startswith('x')]
        )

        if len(input_cols) <= 0:
            raise ValueError("At least one input column 'x1', ..., 'xn' is required.")

        dataset.insert(0, 'x0', 1) # for the bias
        input_cols = ['x0'] + input_cols

        expected_cols = input_cols + ['ev']
        dataset = dataset[expected_cols] # sorts the dataset cols in the given order

        self.col_labels = input_cols
        self.weights = np.random.uniform(-1, 1, len(input_cols))
        self.dataset = dataset
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

        self.error = 1
        self.current_epoch = 0

    @abstractmethod
    def has_next(self) -> bool:
        pass
        
    @abstractmethod
    def next_epoch(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _calc_weighted_sum(self, inputs: np.ndarray) -> Any:
        pass

    def try_current_epoch(self, inputs: List[float]):
        """
        inputs: an array containing the numeric parameters x1, ..., xn you want to test with this epoch
        """
        inputs.insert(0, 1) # adds the bias parameter x0
        return self._calc_weighted_sum(np.array(inputs))