import math
import random
from dataclasses import dataclass
from typing import List, Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from services import NeuralNetwork


@dataclass
class Point:
    x: float
    y: float


def mnk(data: List[Iterable[List[float]]]) -> (float, float):
    """
    Возвращает значения c, d, расчитанные по методу наименьших квадратов
    Принимает в себя список точек
    """
    points = [Point(x, y) for x, y in data]
    n = len(points)
    c = (n * sum((point.x * point.y for point in points)) - sum((point.x for point in points)) * sum((point.y for point in points))) \
        / (n * (sum((math.pow(point.x, 2) for point in points))) - math.pow(sum((point.x for point in points)), 2))
    d = (sum((point.y for point in points)) - c * sum((point.x for point in points))) / n
    return c, d


def af(self: "NeuralNetwork", x: float) -> float:
    return self.weights[0] * x + self.weights[1]


def df(self: "NeuralNetwork", net: float) -> float:
    return 1.0


def just_x(self: "NeuralNetwork", x: int) -> float:
    return float(x)
