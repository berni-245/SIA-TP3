from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
from src.binary_perceptron import BinaryPerceptron

# Datos para el AND
data_and = pd.DataFrame({
    'x1': [-1,  1, -1,  1],
    'x2': [ 1, -1, -1,  1],
    'ev': [-1, -1, -1,  1]
})

# Inicializamos el perceptrón
perceptron = BinaryPerceptron(data_and)

# Función para graficar la frontera de decisión
def plot_decision_boundary(weights, epoch):
    # Limites del plano
    x_vals = [-2, 2]
    if weights[2] == 0:
        y_vals = [0, 0]
    else:
        y_vals = [-(weights[0] + weights[1]*x)/weights[2] for x in x_vals]
    
    plt.clf()
    # Dibuja los puntos
    for _, row in data_and.iterrows():
        color = 'blue' if row['ev'] == 1 else 'red'
        plt.scatter(row['x1'], row['x2'], c=color, label=f"ev={row['ev']}")
    # Línea de decisión
    plt.plot(x_vals, y_vals, 'k--', label='Boundary')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f'Época {epoch} - Weights: {weights.round(2)}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.pause(0.5)

# Mostrar en vivo
plt.ion()
plot_decision_boundary(perceptron.weights, perceptron.current_epoch)
sleep(3)
while perceptron.has_next():
    perceptron.next_epoch()
    plot_decision_boundary(perceptron.weights, perceptron.current_epoch)
    sleep(3)
print("End!")

plt.ioff()
plt.show()
