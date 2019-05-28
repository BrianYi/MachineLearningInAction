import numpy as np
from math import log

# 熵表示混乱程度
def calcEntropy(dataSet):
    dataSetSize = len(dataSet)
    labelCount = {}
    for i in range(dataSetSize):
        label = dataSet[i][-1]
        if label not in labelCount:
            labelCount[label] = 0
        labelCount[label] += 1
    ent = 0.0
    for label in labelCount.values():
        prob = float(labelCount[label])/dataSetSize
        ent += -prob * log(prob,2)
    return ent

# 读取数据集
def createDataSet():
    fr = open('loanDataSet.txt',encoding='utf-8')
    featureNames = []
    headLine = fr.readline()
    headLine = headLine.strip()
    featureNames = headLine.split(',')

    contentByLines = fr.readlines()
    numOfLines = len(contentByLines)
    dataSet = []
    for oneLine in contentByLines:
        oneLine = oneLine.strip()
        oneItem = oneLine.split(',')
        dataSet.append(oneItem)
    return dataSet, featureNames
