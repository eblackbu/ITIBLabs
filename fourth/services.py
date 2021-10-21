import types
from copy import copy
from itertools import combinations
from typing import Callable, Optional
from variants import *

FULL_TRAIN_DATA = [list(map(int, bin(i)[2:].zfill(4))) for i in range(2 ** 4)]
FULL_EXPECTED_OUTPUT_DATA = set_expected_output_data(FULL_TRAIN_DATA)


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
        self.weights: List[float] = [0.0, 0.0, 0.0, 0.0]
        self.rbf: List[List[int]] = [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1]]
        self.current_error_count: Optional[int] = None
        self.current_nn_output: Optional[List[int]] = None
        self.current_partial_error_count: Optional[int] = None
        self.current_partial_nn_output: Optional[List[int]] = None

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate
        self.activation_function = types.MethodType(activation_function, self)
        self.derivative_function = types.MethodType(derivative_function, self)
        self.error_function = types.MethodType(error_function, self)

    def get_fi(self, input_data: List[int]) -> List[float]:
        return [math.exp(-sum([math.pow(input_data[i] - self.rbf[j][i], 2) for i in range(len(input_data))]))
                for j in range(len(self.rbf))]

    def get_net(self, x1: int, x2: int, x3: int, *args) -> float:
        fi_1, fi_2, fi_3 = self.get_fi([x1, x2, x3])
        return fi_1 * self.weights[1] + \
               fi_2 * self.weights[2] + \
               fi_3 * self.weights[3] + \
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
                                   self.get_fi(input_train_data[j])
                                   for j in range(len(input_train_data)))

    def epoch(self,
              input_data: List[List[int]]) -> int:
        """
        :return: Возвращает количество ошибок на данном этапе
        """
        self.current_nn_output = self._get_output_data(input_data)
        self.current_error_count = hamming_distance(self.current_nn_output, FULL_EXPECTED_OUTPUT_DATA)
        if not self.current_error_count:
            return self.current_error_count
        errors = [FULL_EXPECTED_OUTPUT_DATA[i] - self.current_nn_output[i] for i in range(len(self.current_nn_output))]
        self.change_weights([[1] + x for x in input_data], errors)
        return self.current_error_count

    def partial_epoch(self,
                      partial_input_data: List[List[int]],
                      partial_expected_output_data: List[int]) -> int:
        self.current_partial_nn_output = self._get_output_data(partial_input_data)
        self.current_partial_error_count = hamming_distance(self.current_partial_nn_output,
                                                            partial_expected_output_data)
        self.current_nn_output = self._get_output_data(FULL_TRAIN_DATA)
        self.current_error_count = hamming_distance(self.current_nn_output, FULL_EXPECTED_OUTPUT_DATA)
        if not self.current_error_count:
            return self.current_error_count
        errors = [partial_expected_output_data[i] - self.current_partial_nn_output[i]
                  for i in range(len(self.current_partial_nn_output))]
        self.change_weights([[1] + x for x in partial_input_data], errors)
        return self.current_error_count

    def reset(self):
        """
        Ресетает веса на начало
        """
        self.weights = [0.0, 0.0, 0.0, 0.0]


def check_combination(nn: NeuralNetwork, partial_input_data: List[List[int]]) -> (int, bool):
    """
    Проверяет входящую неполную выборку для обучения.
    Возвращает количество эпох для обучения + True, если обучение успешно, False в противном случае
    """
    partial_expected_output_data = set_expected_output_data(partial_input_data)
    epoch_number = 0
    count_errors = 1
    while count_errors:
        epoch_number += 1
        count_errors = nn.partial_epoch(partial_input_data, partial_expected_output_data)
        if nn.current_partial_error_count == 0 and nn.current_error_count != 0:
            return epoch_number, False
    return epoch_number, True


def get_min_sample(nn: NeuralNetwork) -> List[int]:
    result_sample = copy(FULL_TRAIN_DATA)
    epochs = 100

    for i in range(1, len(FULL_TRAIN_DATA)):
        is_found = False
        for comb in combinations(FULL_TRAIN_DATA, len(FULL_TRAIN_DATA) - i):
            tested_sample = list(comb)
            nn.reset()
            tmp_epochs, result = check_combination(nn, comb)
            if result and (len(result_sample) > len(tested_sample)
                           or len(result_sample) == len(tested_sample) and tmp_epochs < epochs):
                result_sample = list(comb)
                is_found = True
                break
        if not is_found:
            break
    return result_sample
