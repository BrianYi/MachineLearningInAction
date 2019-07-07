import numpy as np
from sklearn.datasets import load_iris


def loadSimpData():
    datMat = np.mat([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类,分类标准是threshIneq=['lt','gt'],如果为'lt',那么≤阈值的为-1,>阈值的为+1
    :param dataMatrix:数据集
    :param dimen:维度
    :param threshVal:阈值
    :param threshIneq:分类标准['lt','gt']
    :return: 返回分类后的结果
    '''
    # 进行划分,划分标准是threshIneq=['lt','gt'],如果为'lt',那么≤阈值的为-1,>阈值的为+1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    找到数据集上最佳的单层决策树(即从哪个维度,阈值多少进行分类能获得最小的错误率)
    :param dataArr:数据集
    :param classLabels:类别标签
    :param D:权重向量
    :return:最佳单层决策树的相关信息(维度,阈值,分类标准),最小错误率,最好的分类结果
    '''
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                # 计算新的阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 根据该阈值进行的类别划分的值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 计算错误率
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = (D.T * errArr).flat[0]
                print("split: dim {0}, thresh {1:.2f}, thresh inequal: {2}, the weighted error is {3:.3f}".format(i,
                                                                                                                  threshVal,
                                                                                                                  inequal,
                                                                                                                  weightedError))
                # 获取最小错误率所对应的最佳的单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # (1)初始的权重D_1
    D = np.mat(np.ones((m, 1)) / m)
    # 类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # (2)-a 得到基本分类器G_m(x):根据权重向量D(错分的数据权重较高,正确的数据权重较低),找到最佳的单层决策树
        # (2)-b 计算G_m(x)在数据集上的分类误差率
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:{0}".format(D.T))
        # (2)-c 计算G_m(x)的系数
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 记录最佳决策树的信息
        bestStump['alpha'] = alpha
        # 将最佳决策树加入到单层决策树数组
        weakClassArr.append(bestStump)
        print("classEst: {0}".format(classEst.T))
        # (2)-d 更新训练数据集的权值分布D_{m+1}
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # (3) 构建基本分类器的线性组合:计算累积类别估计值
        aggClassEst += alpha * classEst
        print("aggClassEst: {0}".format(aggClassEst.T))
        # 根据累积类别估计值来计算错误率
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classEst), np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: {0}\n".format(errorRate))
        # 无错,则直接退出
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

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


def loadDataSet_iris():
    """数据集生成

    Returns:
        array -- 数据集
        array -- 标签集
    """
    dataMat, labelMat = load_iris(return_X_y=True)
    dataMat, labelMat = dataMat[:100,:2], labelMat[:100]
    return dataMat, labelMat



# 读取数据集,标签集
dataMat, labelMat = loadDataSet_iris()
tmp=[]
for var in labelMat:
    if var==0:
        tmp.append(-1)
    else:
        tmp.append(1)
labelMat=tmp

m = len(dataMat)
# 特征归一化(特征缩放)
dataMat[:, :] = auto_norm(dataMat[:, :])
# 所有数据的特征增加一列x0为1
dataMat = np.column_stack((np.ones(m), dataMat))
# 交叉验证:将数据打乱
rndidx = np.arange(m)
np.random.shuffle(rndidx)
shuffledX = []
shuffledy = []
for i in range(m):
    shuffledX.append(dataMat[rndidx[i]])
    shuffledy.append(labelMat[rndidx[i]])
dataMat, labelMat = np.array(shuffledX), np.array(shuffledy)
X, y = np.array(dataMat), np.array(labelMat)
mTrain = int(0.75*m)
mTest = m-mTrain
# 获取前mTrain个数据做训练数据,用于训练模型
Xtrain, ytrain = np.array(dataMat[:mTrain]), np.array(labelMat[:mTrain])

# 获取后mTest个数据做测试数据,用于测试预测准确率
Xtest, ytest = np.array(dataMat[-mTest:]), np.array(labelMat[-mTest:])

classifieryArray = adaBoostTrainDS(Xtrain, ytrain)
ypredict = adaClassify(Xtest, classifieryArray)
errors=(ypredict!=np.mat(ytest).T)
errorRate=errors.sum()/len(ytest)
print(errorRate)

