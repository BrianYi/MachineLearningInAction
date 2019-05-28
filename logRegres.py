import matplotlib.pyplot as plt
import numpy as np
import math


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        # 去空白字符,并划分
        lineArr = line.strip().split()
        # X=(x0,x1,x2)=(1.0,x1,x2), x0默认1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 加入类别标签
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """梯度上升算法

    Arguments:
        dataMatIn {list or mat} -- 2维矩阵,每行代表一个训练样本,列代表特征
        classLabels {list or array,mat} -- 类标签,行向量

    Returns:
        weights {mat} -- 回归系数
    """
    dataMatrix = np.mat(dataMatIn)
    # 为了便于矩阵操作,向量都转换为列向量
    labelMat = np.mat(classLabels).T
    # 获取数据集的维度(样本数,特征数)
    m, n = np.shape(dataMatrix)
    # 设置向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 列向量nx1
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # mxn * nx1 = mx1, h为mx1的列向量
        h = sigmoid(dataMatrix@weights)
        # 获取差值
        error = (labelMat - h)
        # 按差值方向调整回归系数(使用全批量梯度下降法)
        weights = weights + alpha * dataMatrix.T * error
    return weights


def plotBestFit(wei):
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord0 = []
    ycord0 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord0.append(dataArr[i, 1])
            ycord0.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    #b = np.poly1d([val[0] for val in weights.tolist()])
    #y = b(x)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    

    dataMat, labelMat = loadDataSet()
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    weights = gradAscent(dataMat[:], labelMat[:])
    plotBestFit(weights)

    # # Sigmoid function
    # x = np.linspace(-10, 10, 201)
    # y = 1/(1+math.e**(-x))
    # plt.plot(x, y)
    # plt.show()
