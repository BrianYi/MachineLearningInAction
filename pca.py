import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return np.mat(dataMat)

def pca(dataMat,topNfeat=9999999):
    # 取平均值
    meanVals=np.mean(dataMat,axis=0)
    # 去平均值
    meanRemoved=dataMat-meanVals
    # meanRemoved=[x_1,...,x_n],求Cov(meanRemoved,meanRemoved)协方差
    covMat=np.cov(meanRemoved,rowvar=False)
    # 求出协方差矩阵的特征值与特征向量
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    # 将特征值排序
    eigValInd=np.argsort(eigVals)
    # 选择最大特征值对应的特征向量,将其构成矩阵
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    # 降维
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat

dataMat = loadDataSet('testSet3.txt')
lowDMat,reconMat=pca(dataMat,1)
plt.scatter(dataMat[:,0].A1, dataMat[:,1].A1,marker='o',s=50)
plt.scatter(reconMat[:,0].A1,reconMat[:,1].A1,marker='o',s=10,c='red')
plt.show()
