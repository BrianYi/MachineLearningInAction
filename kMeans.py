import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 欧式距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

# 为给定数据集构建一个包含k个随机质心的集合
# 随机质心必须要在整个数据集的边界之内
def randCent(dataSet,k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    # 遍历每个特征的值,并随机生成质心,质心应在每个特征值的范围内,不能越界
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

# k-均值聚类
# 初始化k个随机质心(代表k个簇),然后对于每个样本点找到与其最近的质心,则划分为该簇
# 重新计算质心,然后再次找最近点并划分簇,不断重复该过程,直到样本点的簇分配结果不再改变为止
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m=np.shape(dataSet)[0]
    # 存放[类别,欧式距离^2]
    clusterAssment=np.mat(np.zeros((m,2)))
    # 随机生成k个质心
    centroids=createCent(dataSet,k)
    # 用于标记样本点的簇分配结果是否改变了
    clusterChanged=True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist=np.inf; minIndex=-1
            for j in range(k):
                # 计算样本点到每个质心的距离
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                # 选择离样本点距离最近簇作为该样本点的簇
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            # 判断本次样本点的簇分配结果是否与之前不同
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # 记录样本点的[簇,欧式距离^2]
            clusterAssment[i,:]=minIndex,minDist**2
        # 更新质心,即拿簇中所有点的横,纵坐标取平均值赋值给质心
        for cent in range(k):
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=np.mean(ptsInClust,axis=0) # axis=0时,求列平均值
    return centroids,clusterAssment

# 二分K-均值分类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    # 存放[类别,欧式距离^2(误差)]
    clusterAssment = np.mat(np.zeros((m,2)))
    # 初始化一个质心
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 初始化质心列表(放入一个质心)
    centList =[centroid0]
    # 计算初始误差(这里为欧式距离^2)
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    # 当簇的个数小于k时,不断将簇进行划分(K-均值聚类)
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            # 获取当前属于i簇的样本点
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            # 将第i簇进行K-均值聚类(k=2)
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 一分为二之后的总误差
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            # 其余数据的总误差
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            #print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 一分为二的总误差+其余数据的总误差 < lowestSSE
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 最好的划分中心
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 最好的划分重新编号,划分为0的簇编号为原来的编号,划分为1的簇编号为新的编号
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #print('the bestCentToSplit is: ',bestCentToSplit)
        #print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心,原来编号的质心更新
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        # 划分后,增加一个新的质心
        centList.append(bestNewCents[1,:].tolist()[0])
        # 新的簇分配结果更新
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    # 返回质心列表,簇分配结果
    return np.mat(centList), clusterAssment


# 标记列表
markers=['o','v','s','p','P','*','x','D']
# 颜色列表
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
datMat=np.mat(loadDataSet('testSet2.txt'))
k=5
myCentroids,clustAssing=biKmeans(datMat,k)

for classLabel in set(clustAssing[:,0].A1):
    selectSamples=(clustAssing[:,0].A1.T==classLabel)
    plt.scatter(datMat[selectSamples,0].A1,datMat[selectSamples,1].A1,marker=markers[int(classLabel)],c=colors[int(classLabel)], label=int(classLabel))
    plt.scatter(myCentroids[int(classLabel),0],myCentroids[int(classLabel),1],marker='$'+str(int(classLabel))+'$',c=colors[int(classLabel)],s=100)

plt.legend()
plt.show()