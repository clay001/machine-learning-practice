import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
import random
#先随机生成2000个均匀分布在两个半圆上的点数据
sample_num = 2000
rad = 10
thk = 5
sep = 5
now = 1
data = pd.DataFrame(index = np.arange(1,sample_num+1), columns=['x','y','label'])
while(now <= sample_num):
    x = random.uniform(-20,40)
    y = random.uniform(-30,20)
    if (y>=0 and (np.square(x)+np.square(y) >= np.square(rad)) and (np.square(x)+np.square(y) <=np.square(rad+thk))):
        data['x'][now] = x
        data['y'][now] = y
        data['label'][now] = 1
        now += 1
    elif ((y<= -1*sep) and (np.square(x-rad-0.5*thk)+np.square(y+sep) >= np.square(rad)) and (np.square(x-rad-0.5*thk)+np.square(y+sep) <= np.square(rad+thk))):
        data['x'][now] = x
        data['y'][now] = y
        data['label'][now] = -1
        now += 1

#按照题目的意思，上半圆为+1标红，下半圆为-1标蓝
for i in range(sample_num):
    if data.loc[i+1,'label'] == 1:
        plt.scatter(data.loc[i+1,'x'], data.loc[i+1,'y'], color='red',marker='o')
    if data.loc[i+1,'label'] == -1:
        plt.scatter(data.loc[i+1,'x'], data.loc[i+1,'y'], color='blue', marker='o')

#分割出数据和标签
X = data.iloc[:,:2].values
y = data.iloc[:,2].values
X = np.hstack((np.ones((X.shape[0],1)), X))
w = np.random.randn(3,1)  #初始化参数
m = len(y)
X = np.array(X,dtype = 'float')
y = np.array(y,dtype = 'float')

#normal equation
w = np.dot(np.dot(inv(np.matrix(np.dot(X.T,X))),X.T),y)
w = np.array(w)

#取两个点画出超平面所在的直线（hypothesis function）
feature1 = -20
feature2 = -1 * w[0][1]/ w[0][2] * feature1 - (w[0][0]/w[0][2])
feature3 = 20
feature4 = -1 * w[0][1]/ w[0][2] * feature3 - (w[0][0]/w[0][2])
plt.plot([feature1,feature3], [feature2,feature4],'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression performance')
plt.show()
