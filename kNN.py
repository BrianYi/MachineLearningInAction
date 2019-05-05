'''
    第二章:k-近邻法
    1.收集数据
    2.准备输入数据
    3.分析输入数据
    4.训练算法
    5.测试算法
    6.使用算法
'''

import numpy as np
import operator as op
import matplotlib.pyplot as plt
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


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


def autoNorm(dataSet):
    maxVals, minVals = dataSet.max(0), dataSet.min(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    erroCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {0:}, the real answer is: {1:}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            erroCount += 1.0
    print("the total error rate is: {}".format(erroCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, datingDataMat, datingLabels, 3)
    print("You will probably like this person: {}".format(resultList[classifierResult - 1]))


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = int(lineStr[j])
    return returnVect


def handWritingClassTest():
    trainFiles = listdir('trainingDigits')
    mTrain = len(trainFiles)
    trainDataSet = np.zeros((mTrain, 1024))
    trainLabels = []
    for i in range(mTrain):
        fileFullNameStr = trainFiles[i]
        fileNameStr = fileFullNameStr.split('.')[0]
        classNumber = int(fileNameStr.split('_')[0])
        trainLabels.append(classNumber)
        trainDataSet[i, :] = img2vector('trainingDigits/{}'.format(fileFullNameStr))
    testFiles = listdir('testDigits')
    mTest = len(testFiles)
    testDataSet = np.zeros((mTest, 1024))
    testLabels = []
    predLabels = []
    errorCount = 0
    for i in range(mTest):
        fileFullNameStr = testFiles[i]
        fileNameStr = fileFullNameStr.split('.')[0]
        classNumber = int(fileNameStr.split('_')[0])
        testLabels.append(classNumber)
        testDataSet[i, :] = img2vector('testDigits/{}'.format(fileFullNameStr))
        predLabels.append(classify0(testDataSet[i, :], trainDataSet, trainLabels, 3));
        if (predLabels[i] != testLabels[i]):
            errorCount += 1
        print('the classifier came back with {0:}, the real one is {1:}'.format(predLabels[i], testLabels[i]))
    print('the total number of errors is: {0:}'.format(errorCount))
    print('the total error rate is: {0:}'.format(errorCount / float(mTest)))

file2matrix('datingTestSet.txt')
# handWritingClassTest()
# datingClassTest()
# classifyPerson()
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# normMat, ranges, minVals = autoNorm(datingDataMat)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], s=15.0*np.array(datingLabels), c=15.0*np.array(datingLabels))
# plt.show()
