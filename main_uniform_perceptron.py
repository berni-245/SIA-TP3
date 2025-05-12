from time import sleep
from typing import Tuple
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from src.perceptron_function import PerceptronFunction
from src.uniform_perceptron import UniformPerceptron

data = pd.read_csv('./data/TP3-ej2-conjunto.csv')

with open("configs/config_uniform.json", "r") as f:
    config = json.load(f)

if config["test_with_train_data"]:
    train_data = data.copy()
    test_data = data.copy()
else:
    split_result: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(data, test_size=config["test_percentage"]) # type: ignore[assignment]
    train_data, test_data = split_result

    print(data)
    print(f"This is the dataframe used for this perceptron, it will split in {(1 - config['test_percentage'])*100}% training set and {config['test_percentage']*100}% testing set")
    sleep(5)


perceptron = UniformPerceptron(
    train_data,
    config["learn_rate"],
    config["max_epochs"],
    config["random_weight_initialize"],
    PerceptronFunction.from_string(config["function"]),
    config["beta"],
    config["min_error"],
    config["batch_update"]
)


while perceptron.has_next():
    perceptron.next_epoch()
    print(f'{perceptron.current_epoch} - error: {perceptron.error}')
print("Training ended")


Y = perceptron.try_testing_set(test_data)

test_data['calculated'] = Y

print("Results with the testing set:")
sleep(1)
print(test_data)
