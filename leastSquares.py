import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def my_leastsq(x, y, n):
    k = len(x)
    X = np.arange(k * n).reshape([k, n])
    for i in range(k):  # k row
        for j in range(n):  # n colums
            var = x[i] ** j
            X[i][n-j-1] = var

    X = np.mat(X)
    Y = np.mat([y]).T
    xtxi = (X.T * X).I
    ret = xtxi * X.T * Y
    return ret.T


def real_func(x):
    return np.sin(x)


def residual_func(p, x, y):
    f = np.poly1d(p)
    return f(x) - y


def fit(x, y, index, M=0):
    p_init = np.random.rand(M + 1)
    p_lsm = my_leastsq(x, y, M + 1)
    p_lsm = np.array(p_lsm)
    # p_lsm = leastsq(residual_func, p_init, args=(x, y))
    plt.subplot(141 + index)
    # plt.xlabel('x0 AXIS', fontsize=14)
    # plt.ylabel('x1 AXIS', fontsize=14)
    plt.title('LeastSquares', fontsize=24)
    plt.plot(x, real_func(x), color='r', label='real curve')
    plt.plot(x, np.poly1d(p_lsm[0])(x), color='y', label='fit curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    return p_lsm


plt.figure(figsize=(18, 10))
x0_with_noise = np.linspace(0, 2 * np.pi, 30)
x1_with_noise = real_func(x0_with_noise) + np.random.normal(0, 0.1, 30)
# plt.scatter(x0_with_noise, x1_with_noise, label='noise')
index = 0
for i in [0, 3, 6, 9]:
    fit(x0_with_noise, x1_with_noise, index, i)
    index += 1

plt.show()
