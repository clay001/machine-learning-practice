import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
#先随机生成2000个均匀分布在两个半圆上的点数据
sample_num = 2000
rad = 10
thk = 5
sep = np.linspace(0.2,5,25)
ctime = []
#分不同的sep循环
for i in range(len(sep)):
    data = pd.DataFrame(index = np.arange(1,sample_num+1), columns=['x','y','label'])
    now = 1
    while(now <= sample_num):
        x = random.uniform(-20,40)
        y = random.uniform(-30,20)
        if (y>=0 and (np.square(x)+np.square(y) >=      np.square(rad)) and (np.square(x)+np.square(y) <=np.square(rad+thk))):
            data['x'][now] = x
            data['y'][now] = y
            data['label'][now] = 1
            now += 1
        elif ((y<= -1*sep[i]) and (np.square(x-rad-0.5*thk)+np.square(y+sep[i]) >= np.square(rad)) and (np.square(x-rad-0.5*thk)+np.square(y+sep[i]) <= np.square(rad+thk))):
            data['x'][now] = x
            data['y'][now] = y
            data['label'][now] = -1
            now += 1

    #分割出数据和标签
    X = data.iloc[:,:2].values
    y = data.iloc[:,2].values
    X = np.hstack((np.ones((X.shape[0],1)), X))
    w = np.zeros((3,1))
    #开始迭代更新
    alpha = 1
    history = [] #记录每次迭代分类错误的个数
    iter = 1
    while(1):
        score = np.dot(X, w)
        y_pred = np.ones_like(y)  #初始化预测结果
        negative = np.where(score < 0)[0]
        y_pred[negative] = -1
        mistake = len(np.where(y != y_pred)[0])
        history.append(mistake) #记录每次迭代的错误个数
        if mistake == 0:
            ctime.append(iter)
            break
        else:
            #pick = np.random.choice(mistake)
            index = np.where(y != y_pred)[0][0]
            w = w + (alpha * y[index] * X[index, :]).reshape((3,1))
            iter += 1

plt.plot(sep,ctime,'y')
plt.xlabel('sep value')
plt.ylabel('converge time')
plt.title('the convergence time for each sep')
plt.show()