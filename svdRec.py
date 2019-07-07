import numpy as np

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

# 欧几里得距离(0~1)
def ecludSim(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))

# 皮尔逊相关系数(0~1)
def pearsSim(inA,inB):
    if len(inA) < 3: return 1.0
    t=np.corrcoef(inA,inB,rowvar=0)
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]

# 余弦相似度(0~1)
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

myMat = np.mat(loadExData())
ecludSimRes=ecludSim(myMat[:,0],myMat[:,4])
ecludSimRes=ecludSim(myMat[:,0],myMat[:,0])
ecludSimRes=cosSim(myMat[:,0],myMat[:,4])
ecludSimRes=cosSim(myMat[:,0],myMat[:,0])
ecludSimRes=pearsSim(myMat[:,0],myMat[:,4])
ecludSimRes=pearsSim(myMat[:,0],myMat[:,0])
