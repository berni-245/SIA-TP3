from typing import List
from imageio.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as imageio
import tempfile
import os
from src.binary_perceptron import BinaryPerceptron

# Datos para el AND
data = pd.DataFrame({
    'x1': [-1, -1,  1,  1],
    'x2': [-1,  1, -1,  1],
    'ev': [-1,  1,  1, -1],
})

# Inicializamos el perceptrón
perceptron = BinaryPerceptron(data, 0.001, 100, False)

# Directorio temporal para guardar los frames
temp_dir = tempfile.mkdtemp()
frames = []

# Función para graficar la frontera de decisión
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

# Primera imagen
frame_path = os.path.join(temp_dir, f"frame_000.png")
plot_decision_boundary(perceptron.weights, perceptron.current_epoch, frame_path)
frames.append(frame_path)

# Iterar y guardar frames
while perceptron.has_next():
    perceptron.next_epoch()
    frame_path = os.path.join(temp_dir, f"frame_{perceptron.current_epoch:03}.png")
    plot_decision_boundary(perceptron.weights, perceptron.current_epoch, frame_path)
    frames.append(frame_path)

# Crear GIF
images: List[ArrayLike] = [imageio.imread(f) for f in frames]
filename = "./results/perceptron.gif"
kargs = {'duration': 50}
imageio.mimsave(filename, images, 'GIF', **kargs)
print(f"GIF saved as {filename}")

# Limpieza opcional del directorio temporal
for f in frames:
    os.remove(f)
os.rmdir(temp_dir)
