import types
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from variants import *


class PredictNeuralNetwork:

    def __init__(self,
                 learning_rate: float,
                 true_function: Callable[["PredictNeuralNetwork", float], float],
                 error_function: Callable[["PredictNeuralNetwork", int], float] = just_x):
        """
        Инициализируем веса
        """
        self.weights: List[float] = [0.0, 0.0, 0.0, 0.0]

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate
        self.get_expected_output = types.MethodType(true_function, self)
        self.error_function = types.MethodType(error_function, self)

    def get_net(self, x1: float, x2: float, x3: float, x4: float) -> float:
        return x1 * self.weights[0] + \
               x2 * self.weights[1] + \
               x3 * self.weights[2] + \
               x4 * self.weights[3]

    def change_weights(self,
                       input_train_data: List[float],
                       error: float) -> None:
        for i, _ in enumerate(self.weights):
            self.weights[i] += self.learning_rate * self.error_function(error) * input_train_data[i]

    def epoch(self,
              input_data: List[float],
              x: float) -> (float, float):
        current_output = self.predict(input_data)
        error = self.get_expected_output(x) - current_output
        self.change_weights(input_data, error)
        return current_output, error ** 2

    def predict(self,
                input_data: List[float]) -> float:
        return self.get_net(*input_data)


if __name__ == "__main__":
    a = 0
    b = 3

    true_x = np.linspace(a, b + b - a, 200)
    true_y = np.sin(true_x) ** 2

    plt.plot(true_x, true_y)
    nn = PredictNeuralNetwork(learning_rate=0.1, true_function=f)

    for i in range(4, len(true_x[:len(true_x) // 2])):
        nn.epoch(true_x[i - 4: i], true_y[i])

    nn_x = true_x
    nn_y = list(true_y)[:len(true_y) // 2]

    test_part = true_x[(len(true_x) // 2) - 4:]

    for i in range(4, len(test_part)):
        nn_y.append(nn.predict(test_part[i - 4: i]))

    plt.plot(nn_x, nn_y)
