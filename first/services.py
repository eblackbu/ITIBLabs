from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Callable


FULL_TRAIN_DATA = [list(map(int, bin(i)[2:].zfill(4))) for i in range(2 ** 4)]


def hamming_distance(a: List[int], b: List[int]):
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])


class NeuralNetwork:

    def __init__(self, learning_rate: float):
        """
        Инициализируем веса
        """
        self.w1: float = 0.0
        self.w2: float = 0.0
        self.w3: float = 0.0
        self.w4: float = 0.0

        # Параметр w5 не меняется
        self.w5: float = 1.0

        # параметр learning_rate должен иметь значение от 0 не включительно до 1 включительно
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError('Поставьте параметр learning_rate от 0 до 1 (необходимо по сабжекту)')
        self.learning_rate: float = learning_rate

    def _get_value(self, x1: int, x2: int, x3: int, x4: int) -> int:
        out = x1 * self.w1 + x2 * self.w2 + x3 * self.w3 + x4 * self.w4 + self.w5
        return 1 if out > 0.5 else 0

    def _get_output_data(self,
                         input_data: List[List[int]]) -> List[int]:
        return [self._get_value(*combination) for combination in input_data]

    def change_weights(self,
                       sigma_function: Callable[[int], float],
                       sigma: int) -> None:
        raise NotImplementedError

    def epoch(self,
              input_data: List[List[int]],
              expected_output_data: List[int],
              sigma_function: Callable[[int], float]) -> bool:
        """
        :return: True, если вывод совпадает с expected_output_data (нет ошибок). В противном случае False
        """
        nn_output_data = self._get_output_data(input_data)
        sigma = hamming_distance(nn_output_data, expected_output_data)
        if not sigma:
            return True
        self.change_weights(sigma_function, sigma)
        return False
