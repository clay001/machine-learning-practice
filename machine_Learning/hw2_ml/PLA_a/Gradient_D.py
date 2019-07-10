import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取训练数据
data = pd.read_csv('./train_data.csv', header=None)
X = data.iloc[1:,:].values
X = X.astype(float)
label = pd.read_csv('./train_label.csv', header=None)
y = label.iloc[1:,:].values
y = y.astype(float)
m = len(y)  #记录数据点的个数

#把原始标签里的0改成-1
for i in range(m):
    if y[i] == 0:
        y[i] = -1

#数据归一化
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X = (X - mu) / sigma
X = np.hstack((np.ones((X.shape[0],1)), X))
w = np.random.randn(3,1)  #初始化参数

#开始梯度下降
epsilon = 0.0001  #迭代阈值
alpha = 0.01     #学习率
iter = 10000
history = []
error_last = 0
for i in range(iter):
    score = np.dot(X, w)
    y_pred = np.ones_like(y)  #初始化预测结果
    negative = np.where(score < 0)[0]
    y_pred[negative] = -1
    error = 0
    for i in range(m):
        new_error = max((1-np.dot(y_pred[i],np.dot(X[i],w))), 0)
        error += new_error
    history.append(error)
    if  abs(error_last - error) < epsilon:  #如果迭代差小于阈值则跳出
        break
    else:
        w += alpha*np.dot(X.T,y)/m
        error_last = error
   #否则负梯度更新参数值，继续迭代

#画in sample error
history = np.dot(history,1/m)
length = len(history)
time = []
for i in range(length):
    time.append(i)
plt.plot(time,history,'y')
plt.xlabel('iteration time')
plt.ylabel('error')
plt.title('error versus iteration time')
plt.show()

#读取测试数据
test = pd.read_csv('./test_data.csv', header=None)
x = test.iloc[1:,:].values
x = x.astype(float)
tlabel = pd.read_csv('./test_label.csv', header=None)
ty = tlabel.iloc[1:,:].values
ty = ty.astype(float)
test_number = len(ty)

#取两个点画出超平面所在的直线（hypothesis function）
feature1 = -20
feature2 = -1 * w[1]/ w[2] * feature1 - (w[0]/w[2])
feature3 = 20
feature4 = -1 * w[1]/ w[2] * feature3 - (w[0]/w[2])
plt.plot([feature1,feature3], [feature2,feature4],'g')

#画出测试点表现情况
for i in range(test_number):
    if ty[i] == 1:
        plt.scatter(x[i, 0], x[i, 1], color='blue', marker='o', label='Positive')
    if ty[i] == 0:
        plt.scatter(x[i, 0], x[i, 1], color='red', marker='x', label='Negative')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Performance on test data')
plt.show()


