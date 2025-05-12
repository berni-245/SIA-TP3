from typing import List
from imageio.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as imageio
import tempfile
import os
import json
from src.binary_perceptron import BinaryPerceptron

 # AND data
and_data = pd.DataFrame({
    'x1': [-1, -1,  1,  1],
    'x2': [-1,  1, -1,  1],
    'ev': [-1,  -1,  -1, 1],
})

# XOR data
xor_data = pd.DataFrame({
    'x1': [-1, -1,  1,  1],
    'x2': [-1,  1, -1,  1],
    'ev': [-1,  1,  1, -1],
})

with open("configs/config_step.json", "r") as f:
    config = json.load(f)

data = and_data if config["test_and"] else xor_data

perceptron = BinaryPerceptron(data, config["learn_rate"], config["max_epochs"], config["random_weight_initialize"])

temp_dir = tempfile.mkdtemp()
frames = []

def plot_decision_boundary(weights, epoch, filename):
    x_vals = [-2, 2]
    if weights[2] == 0:
        y_vals = [0, 0]
    else:
        y_vals = [-(weights[0] + weights[1]*x)/weights[2] for x in x_vals]
    
    plt.figure(figsize=(4, 4))
    for _, row in data.iterrows():
        color = 'blue' if row['ev'] == 1 else 'red'
        plt.scatter(row['x1'], row['x2'], c=color)
    plt.plot(x_vals, y_vals, 'k--', label='Boundary')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f'Época {epoch}\nWeights: {weights.round(4)}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

frame_path = os.path.join(temp_dir, f"frame_000.png")
plot_decision_boundary(perceptron.weights, perceptron.current_epoch, frame_path)
frames.append(frame_path)

# Iterate and save frames
while perceptron.has_next():
    print(f'{perceptron.current_epoch} - error: {perceptron.error}')
    perceptron.next_epoch()
    frame_path = os.path.join(temp_dir, f"frame_{perceptron.current_epoch:03}.png")
    plot_decision_boundary(perceptron.weights, perceptron.current_epoch, frame_path)
    frames.append(frame_path)
print("Training ended")

# Create gif
images: List[ArrayLike] = [imageio.imread(f) for f in frames]
filename = "./perceptron.gif"
kargs = {'duration': 50}
imageio.mimsave(filename, images, 'GIF', **kargs) # type: ignore[assignment]
print(f"GIF saved as {filename}")

for f in frames:
    os.remove(f)
os.rmdir(temp_dir)
