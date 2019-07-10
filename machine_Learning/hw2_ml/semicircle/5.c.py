import numpy as np
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

#画图展示
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Generate Data')
#plt.show()

#分割出数据和标签
X = data.iloc[:,:2].values
y = data.iloc[:,2].values
X = np.hstack((np.ones((X.shape[0],1)), X))
w = np.zeros((3,1))
m = len(y)
#开始迭代更新
iteration = 10
alpha = 1
history = [] #记录每次迭代分类错误的个数

#pocket algorithm
for i in range(iteration):
    score = np.dot(X, w)
    y_pred = np.ones_like(y)  #初始化预测结果
    negative = np.where(score < 0)[0]
    y_pred[negative] = -1
    mistake = len(np.where(y != y_pred)[0])
    if mistake == 0:
        break
    else:
        pick = np.random.choice(mistake) #在迭代的时候随机选择点进行尝试更新
        index = np.where(y != y_pred)[0][pick]
        w_temp = w+ (alpha * y[index] * X[index, :]).reshape((3,1))
        score = np.dot(X, w_temp)
        y_pred = np.ones_like(y)  # 初始化
        negative = np.where(score < 0)[0]
        y_pred[negative] = -1
        mistake_temp = len(np.where(y != y_pred)[0])
        if mistake_temp < mistake: #判断新的更新是否会向好的方向发展，如果是则更新w，如果不是则保留原w
            w = w_temp
            history.append(mistake_temp)
        else:
            history.append(mistake)
print(history)

#画in sample error
history = np.dot(history,1/m)
length = len(history)
time = []
for i in range(length):
    time.append(i)
plt.plot(time,history,'y')
plt.xlabel('iteration time')
plt.ylabel('in sample error')
plt.title('in sample error versus iteration time')
plt.show()

# 按照题目的意思，上半圆为+1标红，下半圆为-1标蓝
for i in range(sample_num):
    if data.loc[i + 1, 'label'] == 1:
        plt.scatter(data.loc[i + 1, 'x'], data.loc[i + 1, 'y'], color='red', marker='o')
    if data.loc[i + 1, 'label'] == -1:
        plt.scatter(data.loc[i + 1, 'x'], data.loc[i + 1, 'y'], color='blue', marker='o')

#取两个点画出超平面所在的直线（hypothesis function）
feature1 = -20
feature2 = -1 * w[1]/ w[2] * feature1 - (w[0]/w[2])
feature3 = 20
feature4 = -1 * w[1]/ w[2] * feature3 - (w[0]/w[2])
plt.plot([feature1,feature3], [feature2,feature4],'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pocket performance')
plt.show()
