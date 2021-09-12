from typing import TYPE_CHECKING, List
import math


if TYPE_CHECKING:
    from services import NeuralNetwork


FULL_TRAIN_DATA = [list(map(int, bin(i)[2:].zfill(4))) for i in range(2 ** 4)]


def set_expected_output_data() -> List[int]:
    data = []
    for combination in FULL_TRAIN_DATA:
        # 8 Вариант
        data.append((combination[0] | combination[1] | combination[2]) & combination[3])
    return data


def first_af(self: "NeuralNetwork", net: int):
    return 1 if net > 0 else 0


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
