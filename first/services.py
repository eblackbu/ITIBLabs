import types
from typing import Callable, Optional
from variants import *


def hamming_distance(a: List[int], b: List[int]) -> int:
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])


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
        self.weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.current_error_count: Optional[int] = None
        self.current_nn_output: Optional[List[int]] = None
        self._net = None

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate
        self.activation_function = types.MethodType(activation_function, self)
        self.derivative_function = types.MethodType(derivative_function, self)
        self.error_function = types.MethodType(error_function, self)

    def get_net(self, x1: int, x2: int, x3: int, x4: int, *args):
        return x1 * self.weights[1] + \
               x2 * self.weights[2] + \
               x3 * self.weights[3] + \
               x4 * self.weights[4] + \
               1 * self.weights[0]

    def _get_output_data(self,
                         input_data: List[List[int]]) -> List[int]:
        return [self.activation_function(self.get_net(*combination)) for combination in input_data]

    def change_weights(self,
                       input_train_data: List[List[int]],
                       errors: List[int]) -> None:
        for i, _ in enumerate(self.weights):
            self.weights[i] += sum(self.learning_rate *
                                   self.error_function(errors[j]) *
                                   self.derivative_function(self.get_net(*input_train_data[j])) *
                                   input_train_data[j][i]
                                   for j in range(len(input_train_data)))

    def epoch(self,
              input_data: List[List[int]],
              expected_output_data: List[int]) -> int:
        """
        :return: Возвращает количество ошибок на данном этапе
        """
        self.current_nn_output = self._get_output_data(input_data)
        self.current_error_count = hamming_distance(self.current_nn_output, expected_output_data)
        if not self.current_error_count:
            return self.current_error_count
        errors = [expected_output_data[i] - self.current_nn_output[i] for i in range(len(self.current_nn_output))]
        self.change_weights([[1] + x for x in input_data], errors)
        return self.current_error_count


if __name__ == '__main__':
    nn = NeuralNetwork(learning_rate=0.1, activation_function=first_af, derivative_function=first_df)
    input_data = FULL_TRAIN_DATA
    expected_output_data = set_expected_output_data()

    epoch_number = 0
    count_errors = 1
    while count_errors:
        epoch_number += 1
        print(f'Current state: w1 = {nn.weights[1]:.4} w2 = {nn.weights[2]:.4} w3 = {nn.weights[3]:.4} '
              f'w4 = {nn.weights[4]:.4} w0 = {nn.weights[0]:.4}')
        count_errors = nn.epoch(input_data, expected_output_data)
        print(f'k = {epoch_number}')
        print(f'E = {nn.current_error_count}')
        print(f'Y = {nn.current_nn_output}', end='\n\n')

    nn = NeuralNetwork(learning_rate=0.45, activation_function=third_af, derivative_function=third_df)
    input_data = FULL_TRAIN_DATA
    expected_output_data = set_expected_output_data()

    epoch_number = 0
    count_errors = 1
    while count_errors:
        epoch_number += 1
        print(f'Current state: w1 = {nn.weights[1]:.4} w2 = {nn.weights[2]:.4} w3 = {nn.weights[3]:.4} '
              f'w4 = {nn.weights[4]:.4} w0 = {nn.weights[0]:.4}')
        count_errors = nn.epoch(input_data, expected_output_data)
        print(f'k = {epoch_number}')
        print(f'E = {nn.current_error_count}')
        print(f'Y = {nn.current_nn_output}', end='\n\n')

