import pandas as pd
import numpy as np

from src.perceptron_function import PerceptronFunction
from src.uniform_perceptron import UniformPerceptron

data = pd.read_csv('./data/TP3-ej2-conjunto.csv')

# values = {
#     "x1": [-10.000000, -7.777778, -5.555556, -3.333333, -1.111111,
#             1.111111,  3.333333,  5.555556,  7.777778, 10.000000],
#     "ev": [ -8.235948, -7.377621, -4.576818, -1.092440,  0.756447,
#              0.133833,  4.283422,  5.404198,  7.674559, 10.410599]
# }
# data = pd.DataFrame(values)

# np.random.seed(0)
# x1 = np.linspace(-10, 10, 10)
# noise = np.random.normal(loc=0, scale=1.0, size=len(x1))
# ev = x1 + noise
# data = pd.DataFrame({'x1': x1, 'ev': ev})

perceptron = UniformPerceptron(data, 0.1, 1000, True, PerceptronFunction.HYPERBOLIC, min_error=0.01)
print(f'before before:\n{data}')

import warnings

# Convert warnings into errors within this context
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    try:
        while perceptron.has_next():
            # print(f'------- Epoc: {perceptron.current_epoch} ----------')
            perceptron.next_epoch()
            print(f'{perceptron.current_epoch} - error: {perceptron.error}')
            print(perceptron.weights)
    except Exception as e:
        print(e)

