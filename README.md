# TP3 SIA - Perceptrón Simple y Multicapa

## 👋 Introducción

Trabajo práctico para la materia de Sistemas de Inteligencia Artificial en el ITBA. Se buscó implementar un perceptrón escalón, perceptrón lineal, perceptrón no lineal y perceptrón multicapa genérico que, al hacer un main que genere un dataframe con las columnas "x1", "x2", ... "xn", "ev" y se lo pases a estos, se pueda categorizar resultados utilizando el modelo de la neurona. En los mains de este proyecto se agregan algunos problemas de ejemplo (tomados del enunciado de más abajo) y la posibilidad de alterar los hiperparámetros en la configuración previa a correr los mains.

Se cuenta con 3 mains:
- `main_step_perceptron.py` (perceptrón escalón)
- `main_uniform_perceptron.py` (integra perceptrón lineal y perceptrón no lineal en uno)
- `main_multilayer_perceptron.py`(perceptrón multicapa)

Este fue el [Enunciado](docs/Enunciado%20TP3.pdf)

### ❗ Requisitos

- Python3 (La aplicación se probó en la versión de Python 3.11.*)
- pip3
- [pipenv](https://pypi.org/project/pipenv)

### 💻 Instalación

En caso de no tener python, descargarlo desde la [página oficial](https://www.python.org/downloads/release/python-3119/)

Utilizando pip (o pip3 en mac/linux) instalar la dependencia de **pipenv**:

```sh
pip install pipenv
```

Parado en la carpeta del proyecto ejecutar:

```sh
pipenv install
```

para instalar las dependencias necesarias en el ambiente virtual.

## 🛠️ Configuración
En este proyecto se cuenta con 3 archivos de configuración en la carpeta `configs`, cada uno para su main correspondiente. 

Si querés configurar el main `main_step_perceptron.py`, tenés el archivo [`config_step.json`](configs/config_step.json) que cuenta con los hiperparámetros:
- `learn_rate`: [float] la tasa de aprendizaje para la actualización de pesos
- `max_epochs`: [int] la cantidad de épocas máxima antes de dejar de iterar
- `random_weight_initialize`: [bool] True para empezar con pesos aleatorios entre -1 y 1, False para empezar con pesos en 0
- `test_and`: [bool] True para probar el problema And, False para el problema XOR

Si querés configurar el main `main_uniform_perceptron.py`, tenés el archivo [`config_uniform.json`](configs/config_uniform.json) que cuenta con los hiperparámetros:
- `test_percentage`: [float entre 0-1] para elegir el porcentaje tomado del dataset para ser testing set
- `test_with_train_data`: [bool] True para no separar en train dataset y test dataset, y simplemente entrenar y luego testear con el mismo dataset completo (útil para asegurarse que funciona el entrenamiento), false en el caso contrario
- `learn_rate`: [float] la tasa de aprendizaje para la actualización de pesos
- `max_epochs`: [int] la cantidad de épocas máxima antes de dejar de iterar
- `random_weight_initialize`: [bool] True para empezar con pesos aleatorios entre -1 y 1, False para empezar con pesos en 0
- `function`: [str entre "LINEAR", "LOGISTICS", "HYPERBOLIC"] la función de activación a usar, los expected values se normalizarán para este main acorde a sus imágenes
- `beta`: [float] el beta en caso de usar la función LOGISTICS/HYPERBOLIC, se recomienda usar 1.0
- `min_error`: [float] el error mínimo para dejar de iterar
- `batch_update`: [int] para este main solo se agrega la posibilidad de modificar los pesos cada cierto número de datos, el valor ingresado será ese número

Si querés configurar el main `main_multilayer_perceptron.py`, tenés el archivo [`config_multilayer.json`](configs/config_multilayer.json) que cuenta con los hiperparámetros:
- `problem`: [str entre "XOR", "PARITY", "DIGITS"] para elegir el problema a resolver
- `hidden_layers`: [List[int]] arreglo con la cantidad de neuronas para cada capa oculta, la cantidad de elementos en el arreglo será la cantidad de capas ocultas
- `test_percentage`: [float entre 0-1] para elegir el porcentaje tomado del dataset para ser testing set
- `test_with_train_data`: [bool] True para no separar en train dataset y test dataset, y simplemente entrenar y luego testear con el mismo dataset completo (útil para asegurarse que funciona el entrenamiento). false en el caso contrario
- `learn_rate`: [float] la tasa de aprendizaje para la actualización de pesos
- `max_epochs`: [int] la cantidad de épocas máxima antes de dejar de iterar
- `random_weight_initialize`: [bool] True para empezar con pesos aleatorios entre -1 y 1, False para empezar con pesos en 0
- `function`: [str entre "LINEAR", "LOGISTICS", "HYPERBOLIC"] la función de activación a usar
- `beta`: [float] el beta en caso de usar la función LOGISTICS/HYPERBOLIC, se recomienda usar 1.0
- `min_error`: [float] el error mínimo para dejar de iterar
- `optimizer`: [str entre "GRADIENT_DESCENT", "MOMENTUM" y "ADAM"] para elegir el optimizar a utilizar con el que se va a actualizar los pesos, esto está solo para este main

## 🏃 Ejecución

Para probar la aplicación, correr:
```shell
pipenv run python <main deseado>
```
Siendo \<main deseado> una de estas opciones:
- `main_step_perceptron.py`
- `main_uniform_perceptron.py` 
- `main_multilayer_perceptron.py`

Se imprimirá en stdout el error y época actual. En el caso del `main_step_perceptron.py`, se generará un GIF para ver como se separa linealmente los datos (por esta generación de GIF, este perceptrón va un poco más lento que los otros).

Para abrir el google Colab dónde se realizaron las pruebas ir al siguiente [link](https://colab.research.google.com/drive/1g6m9iIVd1q4IE0GQ9GoGSwTa-zh8L5bb?usp=sharing)
