from typing import TYPE_CHECKING, List
import math


if TYPE_CHECKING:
    from services import NeuralNetwork


def set_expected_output_data(input_data: List[List[int]]) -> List[int]:
    data = []
    for combination in input_data:
        # 8 Вариант
        data.append((combination[0] | combination[1] | combination[2]) & combination[3])
    return data


def first_af(self: "NeuralNetwork", net: int):
    return 1 if net > 0.5 else 0


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
