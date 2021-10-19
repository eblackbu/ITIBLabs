from typing import List, TYPE_CHECKING

import math

if TYPE_CHECKING:
    from services import PredictNeuralNetwork


def f(self: "PredictNeuralNetwork", x: float) -> float:
    """
    Функция 8 варианта
    """
    return math.sin(x) ** 2


def just_x(self: "PredictNeuralNetwork", x: float) -> float:
    return float(x)
