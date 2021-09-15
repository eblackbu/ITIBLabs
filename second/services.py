import types
from copy import copy
from itertools import combinations
from typing import Callable, Optional, List
from variants import *


class NeuralNetwork:

    def __init__(self,
                 learning_rate: float,
                 activation_function: Callable[["NeuralNetwork", int], int],
                 derivative_function: Callable[["NeuralNetwork", int], int],
                 error_function: Callable[["NeuralNetwork", int], float] = just_x):
        """
        Инициализируем веса
        Переданная функция активации должна первым параметром принимать параметр self, чтобы превратить ее в метод.
        """
        self.weights: List[float] = [0.0, 0.0]  # c, d

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate
        self.activation_function = types.MethodType(activation_function, self)
        self.derivative_function = types.MethodType(derivative_function, self)
        self.error_function = types.MethodType(error_function, self)


if __name__ == '__main__':
    n = 16
    x_start = -1
    x_end = 3
    y_start = 0
    y_end = 4
