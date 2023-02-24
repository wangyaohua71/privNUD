import numpy as np
import math


def binomial(eps, number, number1):
    p = 0.5
    q = float(1 / (math.exp(eps) + 1))
    number0 = number - number1
    k1 = np.random.binomial(number1, p)
    k2 = np.random.binomial(number0, q)
    k = k1 + k2
    value = (k - number * q) / (p - q)
    return value



