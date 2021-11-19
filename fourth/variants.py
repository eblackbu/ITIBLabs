from typing import TYPE_CHECKING, List
import math


if TYPE_CHECKING:
    from services import NeuralNetwork


def f(combination: List[int]) -> int:
    # 8 Вариант
    return int((combination[0] | combination[1] | combination[2]) & combination[3])


def set_expected_output_data(input_data: List[List[int]]) -> List[int]:
    data = []
    for combination in input_data:
        data.append(f(combination))
    return data


def first_af(self: "NeuralNetwork", net: int):
    return 1 if net > 0 else 0


def first_df(self: "NeuralNetwork", net: int):
    return 1


def second_af(self: "NeuralNetwork", net: int):
    return 1 if 0.5 * (net / (1 + abs(net)) + 1) > 0.5 else 0


def second_df(self: "NeuralNetwork", net: int):
    return 1 / (2 * (1 + pow(1 + abs(net), 2)))


def third_af(self: "NeuralNetwork", net: int):
    return 1 if 1 / (1 + math.exp(-net)) > 0.5 else 0


def third_df(self: "NeuralNetwork", net: int):
    return 1 / (1 + math.exp(-net)) * (1 - 1 / (1 + math.exp(-net)))


def fourth_af(self: "NeuralNetwork", net: int):
    return 1 if 0.5 * (np.tanh(net) + 1) > 0.5 else 0


def fourth_df(self: "NeuralNetwork", net: int):
    return (1 - pow(0.5 * (np.tanh(net) + 1), 2)) / 2


def just_x(self: "NeuralNetwork", x: int) -> float:
    return float(x)


def first_df(self: "NeuralNetwork", net: int):
    return 1


def second_af(self: "NeuralNetwork", net: int):
    raise NotImplementedError


def second_df(self: "NeuralNetwork", net: int):
    raise NotImplementedError


def third_af(self: "NeuralNetwork", net: int):
    return 1 if 1 / (1 + math.exp(-net)) > 0.5 else 0


def third_df(self: "NeuralNetwork", net: int):
    return 1 / (1 + math.exp(-net)) * (1 - 1 / (1 + math.exp(-net)))


def fourth_af(self: "NeuralNetwork", net: int):
    raise NotImplementedError


def fourth_df(self: "NeuralNetwork", net: int):
    raise NotImplementedError


def just_x(self: "NeuralNetwork", x: int) -> float:
    return float(x)
