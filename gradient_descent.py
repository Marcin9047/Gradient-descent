from autograd import grad
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

import time


def euclidean(xin: list) -> float:
    p = np.array(xin)
    return np.sqrt(np.dot(p.T, p))


def f(x: list, alfa: int) -> float:
    n = len(x)
    alf = alfa
    res = 0
    for i in range(1, n + 1):
        exp = (i - 1) / (n - 1)
        res += pow(alf, exp) * pow(x[i - 1], 2)
    return res


@dataclass
class Points:
    points: list


@dataclass
class Learning_params:
    lerning_rate: float
    tmax: int
    error: float


@dataclass
class Solver_Params:
    _points_cls: Points
    _params_cls: Learning_params

    def start(self):
        return self._points_cls.points

    def learning_rate(self):
        return self._params_cls.lerning_rate

    def tmax(self):
        return self._params_cls.tmax

    def error(self):
        return self._params_cls.error


class Solver_Results:
    def __init__(self, fx: list, i: int, x: float):
        self._fx = fx
        self._imax = i
        self._x_fin = x

    def f_array(self) -> list:
        return self._fx

    def i_max(self) -> int:
        return self._imax

    def final_x(self) -> float:
        return self._x_fin


def gradient_descent(f, params: Solver_Params) -> Solver_Results:
    x = params.start()
    beta = params.learning_rate()
    t = params.tmax()
    error = params.error()
    f_out = []
    for i in range(t):
        f_out.append(f(x))
        grad1 = grad(f)
        euc = euclidean(grad1(x))
        if euc < error:
            break
        else:
            x -= np.multiply(grad1(x), beta)
    results = Solver_Results(f_out, i, x)
    return results


# TESTOWANIE


def test_case(f, x_0, Learning_params):
    solver_params = Solver_Params(x_0, Learning_params)
    # start = time.time()
    res = gradient_descent(f, solver_params)
    label1 = f"Learning rate = {solver_params.learning_rate()}"
    print(res.f_array()[0])
    plt.plot([*range(res.i_max() + 1)], res.f_array(), label=label1)


def find_best(f, x_0, min, max, imax, error):
    size = max - min
    step = size / 100
    params = []
    results = []
    time1 = []
    for i in range(100):
        params.append(min + (i * step))
        params1 = Learning_params(min + (i * step), imax, error)
        solver_params = Solver_Params(x_0, params1)
        start = time.time()
        res = gradient_descent(f, solver_params)
        stop = time.time()
        time1.append(stop - start)
        results.append(res.i_max())
    plt.plot(params, time1)


# f1 = partial(f, alfa=1)
# x_0 = Points([3.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.7, 2.1])
# imax = 50
# error = 1e-14

# params = Learning_params(1, imax, error)

# params2 = Learning_params(0.02, imax, error)

# # params3 = Learning_params(0.0095, imax, error)

# # params4 = Learning_params(0.0099, imax, error)

# test_case(f1, x_0, params)
# test_case(f1, x_0, params2)
# test_case(f100, x_0, params3)
# test_case(f100, x_0, params4)
# find_best(f100, x_0, 0.0092, 0.01)

# plt.title("Gradient descent -> alfa = 100")

# plt.xlabel("Learning rate")
# plt.ylabel("Time")

# plt.ylim(0, 100000000)
# plt.yticks(np.arange(0, 1000, 100))
# plt.xticks(np.arange(0, 50, 5))
# plt.yscale("log")
# plt.legend()
# plt.grid()
# plt.show()
