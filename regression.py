import numpy as np
import matplotlib.pyplot as plt
from random import random

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        diffMat=diffMat[:,1]
        weights[j,j]=np.exp(diffMat.T*diffMat/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if np.linalg.det(xTx)==0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws =xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom)==0.0:
        print('This matrix is singualr, cannot do inverse')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr).T
    yMat=np.mat(yArr)
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def stageWise(xArr,yArr,eps=0.1,numIt=200):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMat[:,1]=regularize(xMat[:,1])
    m,n=np.shape(xMat)
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        #print(ws.T)
        lowestError=np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)  # calc mean then subtract it off
    inVar = np.var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        np.random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9:
                trainX.append(xArr[indexList[j]].A1)
                trainY.append(yArr[indexList[j]].A1)
            else:
                testX.append(xArr[indexList[j]].A1)
                testY.append(yArr[indexList[j]].A1)
        wMat = ridgeTest(np.array(trainX)[:,1],trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print(errorMat[i,k])
    meanErrors = np.mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))

# 局部加权线性回归
# xArr,yArr=loadDataSet('ex0.txt')
# xArr,yArr=np.mat(xArr),np.mat(yArr)
# ws=standRegres(xArr,yArr)
# plt.scatter(xArr[:,1].A,yArr.A,s=10)
# xCopy=xArr.copy()
# xCopy.sort(0)
# yHat=lwlrTest(xCopy,xArr,yArr,k=0.01)
# plt.plot(xCopy[:,1],yHat,color='r')
# plt.show()

# 岭回归
# abX,abY=loadDataSet('abalone.txt')
# ridgeWeights=ridgeTest(abX,abY)
# plt.plot(ridgeWeights)
# plt.show()

# 岭回归
# xArr,yArr=loadDataSet('ex0.txt')
# xArr,yArr=np.mat(xArr),np.mat(yArr)
# ws=ridgeTest(xArr[:,1],yArr)
# plt.scatter(xArr[:,1].A,yArr.A,s=10)
# xCopy=xArr.copy()[:,1]
# xCopy.sort(0)
# yHat=ws*xCopy
# plt.plot(xCopy[:,1],yHat,color='r')
# plt.show()


xArr,yArr=loadDataSet('ex0.txt')
xArr,yArr=np.mat(xArr),np.mat(yArr)
crossValidation(xArr,yArr.T)
# plt.scatter(xArr[:,1].A,yArr.A,s=10)
# xCopy=xArr.copy()
# xCopy.sort(0)
#
# ws=stageWise(xArr,yArr,0.1,200)
# yHat=np.multiply(xCopy.A,ws)
# yHat=yHat.sum(1)
# plt.plot(xCopy[:,1],yHat,color='r')
# plt.show()
