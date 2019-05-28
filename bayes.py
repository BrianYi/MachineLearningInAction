import numpy as np
import math

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],        # 正常(0)
                   ['maybe', 'not', 'take', 'him', 'to',
                       'dog', 'park', 'stupid'],    # 侮辱(1)
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],       # 正常(0)
                   ['stop', 'posting', 'stupid', 'worthless',
                    'garbage'],          # 侮辱(1)
                   ['mr', 'licks', 'ate', 'my', 'steak',
                    'how', 'to', 'stop', 'him'],  # 正常(0)
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]        # 侮辱(1)
    classVec = [0, 1, 0, 1, 0, 1]    # 1代表侮辱性文字,0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    '''
    :param dataSet: 数据集
    :return vocabList: 返回词汇表
    '''
    vocabSet = set([])  # 词汇集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 将词汇并入词汇集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return returnVec: 文档向量
    '''
    returnVec = [0]*len(vocabList)  # 创建文档向量
    for word in inputSet:   # 遍历文档,如果词汇表中的词在文档中出现则设为1,否则为0(默认)
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 由于朴素贝叶斯默认每个特征都同等重要,因此设置为1而不是累加
        else:
            print("the word: {0} is not in my Vocabulary!".format(word))
    return returnVec
# P(Y|X)={P(X|Y)*P(Y)}/P(X)
# Y类情况下,X的概率
# Y类的概率
# X的概率

# 此处计算P(X=x|Y=c_k),P(Y=c_k)的概率
# 利用向量的方法,一次性就把所有属于类c_k的各个特征的特征值占比一次就给计算完成
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 获取训练文档数量
    numWords = len(trainMatrix[0])  # 获取特征个数
    # p(Y=1)=所有为1(侮辱性)文档数量/总文档数量
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 计算出P(Y=1)
    p0Num = np.ones(numWords)  # 用于统计Y=0(正常文档)对应的各个特征下特征值的数量总和,默认为全是1的向量而不是0,是为了避免后面累乘时为0
    p1Num = np.ones(numWords)  # 用于统计Y=1(侮辱性文档)对应的各个特征下特征值的数量总和
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):   # 分别算出在Y=0的情况下W每个特征的值的个数和总个数,Y=1的情况下W每个特征的值的个数和总个数
        if trainCategory[i] == 1:   # 为侮辱性文档
            p1Num += trainMatrix[i]  # 
            p1Denom += sum(trainMatrix[i])  # 
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # np.log是为了防止数值太小而损失精度
    p1Vect = np.log(p1Num/p1Denom)  # 计算出P(X=x_i|Y=1),i=1,2,...,32,属于类1的各个特征的特征值数量占比
    p0Vect = np.log(p0Num/p0Denom)  # 计算出P(X=x_i|Y=0),i=1,2,...,32,属于类0的各个特征的特征值数量占比
    return p0Vect, p1Vect, pAbusive # 返回P(X=x_i|Y=0),P(X=x_i|Y=1),P(Y=1)


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOfPosts, listClasses = loadDataSet() # 6篇文章,6个类别即T=(x_1,y_1),(x_2,y_2),\cdots,(x_6,y_6)
    myVocabList = createVocabList(listOfPosts)  # 获取所有单词构成词汇表
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc)) # 将文档转换为文档向量,并放入训练集
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))    # 训练朴素贝叶斯模型
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList,inputSet):
        #词袋向量
        returnVec=[0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                #某词每出现一次，次数加1
                returnVec[vocabList.index(word)]+=1
        return returnVec


def textParse(bigString):
    """解析邮件为单词列表
    
    Arguments:
        bigString {str} -- 一封邮件内容
    """
    # 导入正则表达式
    import re
    # 使用正则表达式
    listOfTokens = re.split(r'\W+', bigString)
    # 将所有长度>2的单词小写并构成一个单词列表
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    """
    对贝叶斯垃圾邮件分类器进行自动化处理
    选取40封邮件做训练数据集,10封邮件做测试数据集,计算分类器的错误率
    """
    import random
    docList = []; classList = []; fullText = []
    # 遍历垃圾邮件,选取50封邮件(垃圾邮件,正常邮件各25封),穿插进行文本提取,标记类别
    for i in range(1,26):
        # 解析垃圾邮件为单词列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        # 加入文档列表集中  
        docList.append(wordList) 
        # 扩展到全文中   
        fullText.extend(wordList)   
        # 标记类别:垃圾邮件
        classList.append(1)
        # 解析正常邮件为单词列表
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        # 加入文档列表集中
        docList.append(wordList)
        # 扩展到全文中
        fullText.extend(wordList)
        # 标记类别:正常邮件
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 随机生成50个数的列表
    trainingSet = list(range(50)); testSet = []
    # 随机选取10个邮件作为测试数据集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    # 选取剩下的40个邮件作为训练数据集
    for docIndex in trainingSet:
        # 邮件转为词向量并追加到训练数据集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 已知的邮件类别追加到训练数据集
        trainClasses.append(classList[docIndex])
    # 获取p(w|c=0),p(w|c=1),p(c=1)
    # p(c|w)=p(w|c)*p(c)/p(w),由于p(w)不变,可以直接比较p(w|c)*p(c)/p(w)
    p0V,p1V,pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        # 邮件转为词向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 计算判错类别的个数
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('classification error ', docList[docIndex])
    # 显示分类器的错误率
    print('the error rate is: ', float(errorCount)/len(testSet))


if __name__ == '__main__':
    # listOfPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOfPosts)
    # print(myVocabList)
    # trainMat = []
    # for postinDoc in listOfPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)
    spamTest()
