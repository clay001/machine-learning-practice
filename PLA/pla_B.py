import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取训练数据
data = pd.read_csv('/home/wx/Desktop/hw2_ml/PLA_b/train_data.csv', header=None)
X = data.iloc[1:,:].values
X = X.astype(float)
label = pd.read_csv('/home/wx/Desktop/hw2_ml/PLA_b/train_label.csv', header=None)
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
w = np.random.randn(3,1)
#开始迭代更新
iteration = 100
alpha = 1
history = [] #记录每次迭代分类错误的个数
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

#读取测试数据
test = pd.read_csv('/home/wx/Desktop/hw2_ml/PLA_b/test_data.csv', header=None)
x = test.iloc[1:,:].values
x = x.astype(float)
tlabel = pd.read_csv('/home/wx/Desktop/hw2_ml/PLA_b/test_label.csv', header=None)
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


