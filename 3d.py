import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.linspace(-3,3,5)
y = np.linspace(-3,3,5)

x,y=np.meshgrid(x,y)

h=[[0,0,1,2,2],[0,-2,-2,1,5],[4,2,6,8,1],[3,-3,-3,0,5],[1,-5,-2,0,3]]

plt.contour(x,y,h,level=10,alpha=0.6)

C = plt.contour(x,y,h,10,colors='black',linewidth=0.5)

plt.clabel(C, inline=True, fontsize=10)
plt.show()