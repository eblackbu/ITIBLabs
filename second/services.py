import types
from typing import Callable, Optional

import numpy as np

from variants import *


class NeuralNetwork:

    def __init__(self,
                 learning_rate: float,
                 activation_function: Callable[["NeuralNetwork", float], float],
                 error_function: Callable[["NeuralNetwork", int], float] = just_x,
                 c: float = 0.0,
                 d: float = 0.0):
        """
        Инициализируем веса
        Переданные функции активации и ее производная должны первым параметром принимать параметр self,
        чтобы превратить их в метод.
        """
        self.weights: List[float] = [c, d]  # c, d

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate
        self.activation_function = types.MethodType(activation_function, self)
        self.error_function = types.MethodType(error_function, self)

        self.current_nn_output: Optional[List[float]] = None

    def get_net(self, x: float) -> float:
        return self.weights[0] * x + self.weights[1]

    def _get_output_data(self,
                         input_data: List[float]) -> List[float]:
        return [self.activation_function(self.get_net(x)) for x in input_data]

    def change_weights(self,
                       input_train_data: List[float],
                       errors: List[float]) -> None:
        self.weights[0] += sum(self.learning_rate *
                               self.error_function(errors[j]) *
                               input_train_data[j]
                               for j in range(len(input_train_data)))
        self.weights[1] += sum(self.learning_rate *
                               self.error_function(errors[j])
                               for j in range(len(input_train_data)))

    def epoch(self,
              input_data: List[float],
              expected_output_data: List[float]) -> float:
        self.current_nn_output = self._get_output_data(input_data)
        errors = [expected_output_data[i] - self.current_nn_output[i] for i in range(len(self.current_nn_output))]
        self.change_weights([x for x in input_data], errors)
        return sum([x ** 2 for x in errors])

    def predict(self,
                input_data: List[float]) -> List[float]:
        return [self.c * x + self.d for x in input_data]

    @property
    def c(self):
        return self.weights[0]

    @property
    def d(self):
        return self.weights[1]


if __name__ == '__main__':
    data = []
    epochs = 10
    epoch_output = []
    a = -1
    b = 3
    c = 3.0
    d = 1.0
    N = 16
    A = 3

    x_points = np.random.uniform(low=a, high=b, size=N)
    rA = np.random.normal(scale=A, size=x_points.size)
    y_points = c * x_points ** 2 + d + rA
    _c, _d = mnk(list(zip([x ** 2 for x in x_points], y_points)))

    true_x = np.linspace(a, b)
    true_y = c * true_x ** 2 + d

    mnk_x = np.linspace(a, b)
    mnk_y = _c * mnk_x ** 2 + _d

    nn = NeuralNetwork(learning_rate=0.0005, activation_function=af)

    for i in range(epochs):

        nn.epoch(input_data=[x ** 2 for x in x_points], expected_output_data=list(y_points))
        x = np.linspace(a, b, 100)
        y = c * x ** 2 + d
