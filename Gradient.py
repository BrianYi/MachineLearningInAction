import numpy as np
import matplotlib.pyplot as plt


def fx(inX):
    """f(x)=(x-1)(x-2)(x-3)=x^3-6x^2+11x-6

    Arguments:
        inX {array} -- x坐标点数组

    Returns:
        [array] -- 返回f(x)计算结果
    """
    ones = np.ones(np.shape(inX))
    return inX**3-6*inX**2+11*inX-6*ones


def deriv_fx(inX):
    """f'(x)=3x^2-12x+11

    Arguments:
        inX {array} -- x坐标点数组

    Returns:
        [array] -- 返回f'(x)计算结果
    """
    ones = np.ones(np.shape(inX))
    return 3*inX**2-12*inX+11*ones


def GradientAscent(startX, endX, alpha=0.1):
    """梯度上升

    Arguments:
        startX {float} -- 起始位置
        endX {float} -- 终止位置

    Keyword Arguments:
        alpha {float} -- 步长 (default: {0.1})

    Returns:
        outX,outY[array] -- 返回梯度上升的路径点
    """
    outX = []
    xOld = startX-1
    xNew = startX
    epsilon = 0.000001
    while xNew-xOld > epsilon and xNew < endX:
        xOld = xNew
        outX.append(xOld)
        xNew = xOld+alpha*deriv_fx(xOld)
    outX = np.array(outX)
    outY = fx(outX)
    return outX, outY


# def GradientDescent(startX, endX, alpha=0.1):
#     """梯度下降

#     Arguments:
#         startX {float} -- 起始位置
#         endX {float} -- 终止位置

#     Keyword Arguments:
#         alpha {float} -- 步长 (default: {0.1})

#     Returns:
#         outX,outY[array] -- 返回梯度下降的路径点
#     """
#     outX = []
#     xOld = startX-1
#     xNew = startX
#     epsilon = 0.000001
#     while xNew-xOld > epsilon and xNew < endX:
#         xOld = xNew
#         outX.append(xOld)
#         xNew = xOld-alpha*deriv_fx(xOld)
#     outX = np.array(outX)
#     outY = fx(outX)
#     return outX, outY



if __name__ == "__main__":
    # 从[1.0,3.0)平均间隔取100个点,获取x坐标
    x = np.linspace(1.0, 3.0, 100)
    # 计算曲线的y坐标
    y = fx(x)

    # 梯度上升放在第一个图
    plt.subplot(121)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, label='curve')
    newX, newY = GradientAscent(1.0, 3.0, 0.1)
    plt.scatter(newX, newY, color='r', label='step')
    plt.title('Gradient Ascent')
    plt.legend()

    # 梯度下降放在第二个图
    plt.subplot(122)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, label='curve')
    newX, newY = GradientDescent(1.5, 3.0, 0.1)
    plt.scatter(newX, newY, color='b', label='step')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()
