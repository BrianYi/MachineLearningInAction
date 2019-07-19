import numpy as np
from os import listdir

# K近邻类
class NearestNeighbor:
    def __init__(self):
        pass

    # 训练
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.Xtr = X
        self.ytr = y

    # 预测
    def predict(self, X):
        X = np.array(X)
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            # 当前点的各个维度值与所有点的各个维度值进行相减取平方和,找出最小值(即距离最近)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred

# 文件内容转矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 图片转vector
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = int(lineStr[j])
    return returnVect

# 获取训练数据集
trainFiles = listdir('trainingDigits')
lenTrainFiles = len(trainFiles)
trainDataSet = np.zeros((lenTrainFiles, 1024))
trainLabels = []
for i in range(lenTrainFiles):
    fileFullNameStr = trainFiles[i]
    fileNameStr = fileFullNameStr.split('.')[0]
    classNumber = int(fileNameStr.split('_')[0])
    trainLabels.append(classNumber)
    trainDataSet[i,:] = img2vector('trainingDigits/{}'.format(trainFiles[i]))

# 获取测试数据集
testFiles = listdir('testDigits')
lenTestFiles = len(testFiles)
testDataSet = np.zeros((lenTestFiles, 1024))
testLabels = []
for i in range(lenTestFiles):
    fileFullNameStr = testFiles[i]
    fileNameStr = fileFullNameStr.split('.')[0]
    classNumber = int(fileNameStr.split('_')[0])
    testLabels.append(classNumber)
    testDataSet[i, :] = img2vector('testDigits/{}'.format(fileFullNameStr))

# k近邻预测
near = NearestNeighbor()
near.train(trainDataSet, trainLabels)
y = near.predict(testDataSet)
errorCount = np.sum(y != testLabels)
print("error rate:{0:f}".format(errorCount*1.0/lenTestFiles))
