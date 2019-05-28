import numpy as np
import matplotlib.pyplot as plot


def fx(inX):
    """f(x)=(x-1)(x-2)(x-3)=x^3-6x^2+11x-6

    Arguments:
        inX {array or value} -- [description]

    Returns:
        [type] -- [description]
    """
    ones = np.ones(np.shape(inX))
    return inX**3-6*inX**2+11*inX-6*ones


def deriv_fx(inX):
    """f'(x)=3x^2-12x+11

    Arguments:
        inX {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    ones=np.ones(np.shape(inX))
    return 3*inX**2-12*inX+11*ones


def GradientAscent(startX,endX,alpha=0.1):
    """梯度上升

    Arguments:
        inX {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    outX = []
    xOld = startX
    xNew = 0.0
    epsilon = 0.000001
    while xNew-xOld > epsilon and xNew < endX:
        xOld = xNew
        outX.append(xOld)
        xNew = xOld+alpha*deriv_fx(xOld)
    outX = np.array(outX)
    outY = fx(outX)
    return outX, outY


def GradientDescent(startX,endX,alpha=0.1):
    """梯度下降

    Arguments:
        inX {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    outX = []
    alpha = 0.1
    xOld = startX
    xNew = 0.0
    epsilon = 0.000001
    while xNew-xOld > epsilon and xNew < endX:
        xOld = xNew
        outX.append(xOld)
        xNew = xOld-alpha*deriv_fx(xOld)
    outX = np.array(outX)
    outY = fx(outX)
    return outX, outY


if __name__ == "__main___":
    print('asddd')
    x = np.linspace(-0.0, 4.0, 200)
    y = fx(x)

    plt.subplots(121)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, label='curve')
    newX, newY = GradientAscent(0.0,4.0)
    plt.scatter(newX, newY, color='r', label='step')
    plt.title('Gradient Ascent')
    plt.legend()

    plt.subplot(122)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, label='curve')
    newX, newY = GradientDescent(0.0,4.0)
    plt.scatter(newW, newY, color='r', label='step')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()
