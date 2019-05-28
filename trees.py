'''
    决策树的实现
    5/5
'''
import numpy as np
from math import log

def calcShannonEnt(dataSet):
    '''
    计算熵值
    :param dataSet: 数据集
    :return: 熵值
    '''
    dataSetSize = len(dataSet)
    classCounts = {}
    for item in dataSet:
        currentClass = item[-1]
        if currentClass not in classCounts.keys():
            classCounts[currentClass] = 0
        classCounts[currentClass] += 1
    ent = 0.0
    for currentClass in classCounts.keys():
        prob = float(classCounts[currentClass]) / dataSetSize
        ent += -prob * log(prob, 2)
    return ent


# def createDataSet():
#     '''
#     数据集创建
#     :return: 返回数据集
#     '''
#     fr = open('loanDataSet.txt',encoding='utf-8')
#     dataSet=[]
#     headLine = fr.readline()
#     headLine = headLine.strip()
#     labels = headLine.split(',')
#     content = fr.readlines()
#     for oneLine in content:
#         oneLine = oneLine.strip()
#         dataSet.append(oneLine.split(','))
#     return dataSet, labels

def createDataSet():
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 第几特征
    :param value: 特征值
    :return: 按指定特征划分的数据集
    '''
    dataSetCopy = []
    for item in dataSet:
        if item[axis] == value:
            reducedItemVec = item[:axis]
            reducedItemVec.extend(item[axis + 1:])
            dataSetCopy.append(reducedItemVec)
    return dataSetCopy


def chooseBestFeatureToSplit(dataSet):
    '''
    根据信息增益的大小,返回最好的数据划分方式
    :param dataSet:
    :return: 特征
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, axis=i, value=val)
            prob = float(len(subDataSet)) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    多数表决
    :param classList: 类型列表
    :return: 类型中占数最多的类型值
    '''
    classCount = {}
    for item in classList:
        if item not in classCount:
            classCount[item] = 0
        classCount[item] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[0], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    featLabels = labels[:]
    '''
    创建字典决策树
    :param dataSet: 数据集
    :param labels: 数据标签
    :return: 返回字典决策树
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 如果所有类别相同,则停止递归,返回当前类别值
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果使用完了所有特征,仍然不能将数据划分成仅包含唯一类别的分组,则用多数表决的方法来返回出现次数最多的类别作为返回值
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 增益最大的特征
    bestFeatLabel = featLabels[bestFeat]  # 获取特征标签
    myTree = {bestFeatLabel: {}}  # 加入字典中
    del (featLabels[bestFeat])  # 从标签列表删除该标签(因为已经选取了该标签作为划分标准,以后不会再用到)
    featValues = [example[bestFeat] for example in dataSet]  # 获取该特征下所有特征值
    uniqueVals = set(featValues)  # 让每个特征值唯一
    for value in uniqueVals:  # 根据标签值,来进行分类
        subLabels = featLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree


def classify(decisionTree, featLabels, testVec):
    firstStr = list(decisionTree.keys())[0]    # 获取当前用于划分数据的特征标签
    secondDict = decisionTree[firstStr]    # 获取该特征标签下的不同特征值分支
    featIndex = featLabels.index(firstStr)  # 获取特征标签对应的特征索引
    for key in secondDict.keys():   # 遍历特征标签下的所有特征值
        if testVec[featIndex] == key:   # 若特征值相同则进入下一个分支
            if type(secondDict[key]).__name__ == 'dict':    # 当前下一个分支是否是子树
                classLabel = classify(secondDict[key], featLabels, testVec) # 进入将特征向量放入,在子树中继续递归决策(选择分支)
            else: classLabel = secondDict[key]  # 当前下一个分支是叶子节点,则确定了类别,可以直接返回了
    return classLabel

dataSet, featLabels = createDataSet()
myTree = createTree(dataSet, featLabels)
print(myTree)
# age,work,house,loan,class
classLabel = classify(myTree, featLabels, [1,1])
print(classLabel)
