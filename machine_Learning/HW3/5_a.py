import numpy as np  ##科学计算库
import matplotlib.pyplot as plt  ##绘图库

#读取数据
A = np.loadtxt('./A.txt').reshape(80,3)
b = np.loadtxt('./b.txt').reshape(80,1)
x = np.zeros((3,1))  #初始化参数

def grad(A,b,x):
    #开始梯度下降
    epsilon = 0.0001  #迭代阈值
    alpha = 0.00003     #学习率
    iter = 10000
    history = []
    error_last = 0
    for i in range(iter):
        dis = np.dot(A, x) - b
        error = 0
        for i in range(80):
            error += np.square(dis[i])
        error = np.sqrt(error)
        history.append(error)
        if  abs(error_last - error) < epsilon:            #如果迭代差小于阈值则跳出
            break
        else:
            x -= alpha*np.dot(A.T,dis)
            error_last = error
    #否则负梯度更新参数值，继续迭代
    #print(history)
    return history

#画iteration
history = grad(A,b,x)  #得到梯度下降的误差信息
length = len(history)
time = []
for i in range(length):
    time.append(i)
plt.plot(time,history,'y')
plt.xlabel('iteration time')
plt.ylabel('error')
plt.title('error versus iteration time')
plt.show()

#第二问
condition_number_list = [] #分别建立列表存储
convergence_rate_list = []
for i in range(50):
    my_A = np.random.rand(80,3) #随机生成新矩阵
    my_b = np.random.rand(80,1)
    value, _ = np.linalg.eig(np.dot(my_A.T,my_A))
    condition_number = max(value)/min(value)
    x = np.zeros((3,1))
    convergence_number = len(grad(my_A,my_b,x))
    convergence_rate = 1/convergence_number
    condition_number_list.append(condition_number)
    convergence_rate_list.append(convergence_rate)

plt.scatter(condition_number_list,convergence_rate_list)
plt.xlabel('condition_number')
plt.ylabel('convergence_rate')
plt.title('convergence_rate versus condition_number')
plt.ylim([])
plt.show()


