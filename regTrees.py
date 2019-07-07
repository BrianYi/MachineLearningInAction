import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #将数据转float类型
        dataMat.append(fltLine)
    return dataMat

# 按照第i个特征的值将数据集划分为2类
def binSplitDataSet(dataSet, feature, value):
    # 选取所有样本中第i个特征值>value的,构成一个样本矩阵
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    # 选取所有样本中第i个特征值<=value的,构成一个样本矩阵
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

# 建立叶节点
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

# 计算总方差,方差越小则分类越好(用来衡量错误率,可以理解为类似于决策树中的信息增益)
def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

# 创建二叉树
# 分叉结点[特征索引,特征值], 叶子结点[特征索引,均值]
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val
    # 字典存储子树
    retTree = {}
    # 存储特征索引
    retTree['spInd']=feat
    # 存储特征值
    retTree['spVal']=val
    # 根据特征索引与特征值进行数据集划分
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    # 递归创建左子树
    retTree['left']=createTree(lSet,leafType,errType,ops)
    # 递归创建右子树
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

# 找到最佳切分方式,即选择最好的特征进行划分
# 1.如果特征值都相等,则直接创建叶节点
# 2.如果切分数据集后效果提升不大,则直接创建叶节点
# 3.如果两个切分后的子集中,有一个大小小于用户定义的tolN,那么直接创建叶节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # tolS是容许的误差下降值,tolN是划分的最少样本数
    tolS=ops[0]; tolN=ops[1]
    # 停止条件,如果第i个特征的所有值相等,则停止,生成叶节点
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None, leafType(dataSet)
    m,n=np.shape(dataSet)
    # 当前数据集的误差
    S=errType(dataSet)
    bestS=np.inf; bestIndex=0; bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].A1):
            mat0,mat1=binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果有一个少于划分的最少样本数tolN,则继续进行选择
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            # 选取误差值最小的,存储[特征索引,特征值,误差值]
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 停止条件,当前误差与划分后的误差的差值较小(<tolS),即误差减少不大则停止,生成叶节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据[特征索引,特征值]切分数据集
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则停止切分,生成叶节点
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

myDat = loadDataSet('ex0.txt')
myMat = np.mat(myDat)[:,1:]
tree = createTree(myMat)
print(tree)
plt.scatter(myMat[:,0].A,myMat[:,1].A)
plt.show()

