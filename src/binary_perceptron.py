from argparse import ArgumentTypeError
import math
from typing import Literal
import pandas
import numpy as np
from numpy.typing import NDArray

class BinaryPerceptron():
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
            output = self.try_current_epoch(inputs)
            delta = row['ev'] - output
            print(f"partial error {i}: {abs(delta)}")
            self.weights += self.learn_rate * delta * inputs
            total_err += abs(delta)
        
        self.error = total_err
        print(f"total error: {self.error}")
        return self.weights

    def try_current_epoch(self, inputs: np.ndarray) -> Literal[-1, 1]:
        output_raw = np.dot(self.weights, inputs)
        return 1 if output_raw >= 0 else -1

