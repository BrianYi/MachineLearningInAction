import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def Gram(x):
    x = np.mat(x)
    ret = x * x.T
    return ret


def sign(gi, y, a, b):
    ret = a * y @ gi + b
    return ret


def fit(x, y):
    g = Gram(x)
    n = len(y)
    a = np.array(np.zeros(n))
    b = 0
    while True:
        flag = True
        for i in range(n):
            gi = np.array(g[i, :])[0]
            if y[i] * sign(gi, y, a, b) <= 0:
                a[i] = a[i] + 1
                b = b + y[i]
                flag = False
            if not flag:
                break
        if flag:
            break
    w = a * y @ x
    return w, b


def auto_norm(X):
    """特征归一化(或特征缩放)

    Arguments:
        X {array} -- 数据集

    Returns:
        array -- 返回归一化后的数据集
    """
    X = np.array(X)
    n = len(X[0])
    minVals = X.min(0)
    maxVals = X.max(0)
    newVals = (X-minVals)/(maxVals-minVals)
    return newVals


def main():
    raw_x, raw_y = load_iris(return_X_y=True)
    x = auto_norm(raw_x[:100, :2])
    y = raw_y[:100] * 2 - 1
    w, b = fit(x, y)
    x0_min = x[:100, 0].min()
    x0_max = x[:100, 0].max()
    x1_min = -(w[0] * x0_min + b) / w[1]
    x1_max = -(w[0] * x0_max + b) / w[1]
    plt.title('perceptron dual by hand')
    plt.scatter(x[:50, 0], x[:50, 1], label='-1')
    plt.scatter(x[-50:, 0], x[-50:, 1], label='1', marker='x')
    plt.plot([x0_min, x0_max], [x1_min, x1_max],
             label='fitting curve', color='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
