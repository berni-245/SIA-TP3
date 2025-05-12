from time import sleep
from typing import List, Tuple
import pandas as pd
import json
from src.perceptron_function import PerceptronFunction
from src.multilayer_perceptron import MultiLayerPerceptron, NeuralNet, PerceptronOptimizer
from sklearn.model_selection import train_test_split


# Generate xor dataframe

data_xor = pd.DataFrame({
    'x1': [-1,  1, -1,  1],
    'x2': [ 1, -1, -1,  1],
    'ev': [[1], [1], [-1],  [-1]]
})

# Generate parity or ascii_number dataframe
with open("data/TP3-ej3-digitos.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

num_lines_per_digit = 7
pixels_per_line = 5

num_digits = len(lines) // num_lines_per_digit

arr = []
for i in range(num_digits):
    digit_lines = lines[i * num_lines_per_digit : (i + 1) * num_lines_per_digit]
    # Flatten the 7x5 to 1x35
    pixels = [int(ch) for line in digit_lines for ch in line.split()]
    arr.append(pixels)

# Create dataframe with needed col names
data_digits = pd.DataFrame(arr, columns=[f'x{i+1}' for i in range(num_lines_per_digit * pixels_per_line)])

with open("configs/config_multilayer.json", "r") as f:
    config = json.load(f)

if config["problem"] == "parity":
    expected_values = [[i%2] for i in range(len(data_digits))]  
else:
    expected_values = [[
        int(0==i), int(1==i), int(2==i), int(3==i), int(4==i), int(5==i), int(6==i), int(7==i), int(8==i), int(9==i)
        ] for i in range(len(data_digits))]  

data_digits['ev'] = expected_values

data = data_xor if config["problem"] == "xor" else data_digits

print(data)
print(f"This is the dataframe used for this perceptron, it will split in {(1 - config['test_percentage'])*100}% training set and {config['test_percentage']*100}% testing set")
sleep(5)


split_result: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(data, test_size=config["test_percentage"]) # type: ignore[assignment]
train_data, test_data = split_result

layers: List[int] = config["hidden_layers"]
layers.append(len(data['ev'][0]))

neural_net: NeuralNet = NeuralNet(
    len(data.columns) - 1,
    layers, 
    PerceptronFunction.from_string(config["function"]),
    PerceptronOptimizer.from_string(config["optimizer"]),
    config["beta"],
    config["random_weight_initialize"]
)

multi_layer_perceptron = MultiLayerPerceptron(
    neural_net,
    train_data,
    config["learn_rate"],
    config["min_error"],
    config["max_epochs"]
)

while (multi_layer_perceptron.has_next()):
    multi_layer_perceptron.next_epoch()
    print(f'{multi_layer_perceptron.current_epoch} - error: {multi_layer_perceptron.error}')

print("Results with the testing set (unseen data):")
sleep(1)
for output, num in zip(multi_layer_perceptron.try_testing_set(test_data), test_data['ev']):
    print(f"Expected value:{num} - prediction:{[round(float(val), 2) for val in output]}")
print("The model can't generalize if not with a lot of data")
