from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.perceptron_function import PerceptronFunction
from src.multilayer_perceptron import MultiLayerPerceptron, NeuralNet, PerceptronOptimizer
from src.binary_perceptron import BinaryPerceptron

data_xor = pd.DataFrame({
    'x1': [-1,  1, -1,  1],
    'x2': [ 1, -1, -1,  1],
    'ev': [[1], [1], [-1],  [-1]]
})

neural_net: NeuralNet = NeuralNet(2, [4, 4, 2, 1], PerceptronFunction.HYPERBOLIC, PerceptronOptimizer.ADAM)

multi_layer_perceptron = MultiLayerPerceptron(neural_net, data_xor, learn_rate=0.1, min_error=0.001, max_epochs=10000)

while (multi_layer_perceptron.has_next()):
    print(f"Epoch: {multi_layer_perceptron.current_epoch}")
    multi_layer_perceptron.next_epoch()

print(multi_layer_perceptron.try_testing_set(data_xor))

print("Sleeping for 10 seconds before next test")
sleep(10)

# ---

# Leer el archivo como líneas
with open("data/TP3-ej3-digitos.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Cada número ocupa 7 líneas
num_lines_per_digit = 7
pixels_per_line = 5

# Total de dígitos
num_digits = len(lines) // num_lines_per_digit

data = []
for i in range(num_digits):
    digit_lines = lines[i * num_lines_per_digit : (i + 1) * num_lines_per_digit]
    # Aplanar los 7x5 a 1x35
    pixels = [int(ch) for line in digit_lines for ch in line.split()]
    data.append(pixels)

# Crear el DataFrame con columnas x0, x1, ..., x34
df = pd.DataFrame(data, columns=[f'x{i+1}' for i in range(35)])

expected_values = [[
    int(0==i), int(1==i), int(2==i), int(3==i), int(4==i), int(5==i), int(6==i), int(7==i), int(8==i), int(9==i)
    ] for i in range(len(df))]  
df['ev'] = expected_values


neural_net: NeuralNet = NeuralNet(35, [16, 16, 10], PerceptronFunction.LOGISTICS, PerceptronOptimizer.ADAM)

multi_layer_perceptron = MultiLayerPerceptron(neural_net, df, learn_rate=0.01, min_error=0.001, max_epochs=100000)

while (multi_layer_perceptron.has_next()):
    multi_layer_perceptron.next_epoch()
    print(f"Epoch: {multi_layer_perceptron.current_epoch} - Error: {round(multi_layer_perceptron.error, 5)}")

for output, num in zip(multi_layer_perceptron.try_testing_set(df), range(df.size)):
    print(f"Num:{num} - prediction:{[round(float(val), 2) for val in output]}")
