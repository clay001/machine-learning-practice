import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
import random
#先随机生成2000个均匀分布在两个半圆上的点数据
sample_num = 2000
rad = 10
thk = 5
sep = -5
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
#画图展示
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Generate Data')
#plt.show()

#分割出数据和标签
#加上x和y的平方项 （不行，可能要立方项）
X = data.iloc[:,:2].values
expand1 = (np.power(data.iloc[:,0].values,2)).reshape((2000,1))
expand2 = (np.power(data.iloc[:,1].values,2)).reshape((2000,1))
X = np.hstack((X,expand1))
X = np.hstack((X,expand2))
y = data.iloc[:,2].values
X = np.hstack((np.ones((X.shape[0],1)), X))
w = np.zeros((5,1))
X = np.array(X,dtype = 'float')
y = np.array(y,dtype = 'float')

#normal equation
w = np.dot(np.dot(inv(np.matrix(np.dot(X.T,X))),X.T),y)
w = np.array(w)

#计算打印in sample 的准确值
score = np.dot(X, w.T)
y_pred = np.ones_like(y)  #初始化预测结果
negative = np.where(score < 0)[0]
y_pred[negative] = -1
mistake = len(np.where(y != y_pred)[0])
acc =  (1-mistake/sample_num)*100
print(acc)

#取密集的点画出曲线（hypothesis function）（需要解方程，列公式）
kkk = np.linspace(-20,30,10000)
for i in kkk:
    feature1 = i
    #feature2 =

    feature3 = i+0.001
    #feature4 =
    #plt.plot([feature1,feature3], [feature2,feature4],'g')

#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('transform performance')
#plt.show()