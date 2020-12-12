import numpy as np


def sigma(x: float) -> float:
    return 1 / (1 + np.exp(x))
