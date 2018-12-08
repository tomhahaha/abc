import random
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math

def createData(num, dim):  # 数据组数与数据维度
    data = []
    for i in range(num):
        pair = []
        for j in range(dim):
            pair.append(random.random())
        data.append(pair)
    return data
def initOutputLayer(m, n):  # m为竞争层节点数目；n为每一个节点的维度
    layers = []
    random.seed()
    for i in range(m):
        unit = []  # 每一个节点
        for j in range(n):
            unit.append(round(random.random(),2))
        layers.append(unit)
    return layers
def adjustWeight(w, x, alpha):  # w为要调整的权值向量；x为输入向量；alpha为学习率
    if len(w)!=len(x):
        raise("w,x维度应该相等！")
    w_new = []
    for i in range(len(w)):
        w_new.append(w[i] + alpha*(x[i] - w[i]))
    return w_new
def normalization(v):  # v为向量
    norm = LA.norm(v, 2)  # 计算2范数
    v_new = []
    for i in range(len(v)):
        v_new.append(round(v[i]/norm,2))  # 保留2位小数
    return v_new

def normalizationVList(X):
    X_new = []
    for x in X:
        X_new.append(normalization(x))
    return X_new

def calSimilarity(x, y):  # 计算x,y两个向量的相似度
    if len(x)!=len(y):
        raise("维度不一致！")
    c = 0
    for i in range(len(x)):
        c += pow((x[i] - y[i]), 2)
    return  math.sqrt(c)

def getWinner(x, layers):  # 找到layers里面与x最相似的节点
    # x = normalization(x)
    # layers = normalizationVList(layers)
    min_value = 100000  # 存储最短距离
    min_index = -1  # 存储跟x最相似节点的竞争层节点index
    for i in range(len(layers)):
        v = calSimilarity(x, layers[i])
        if v < min_value:
            min_value = v
            min_index = i
    return min_index  # 返回获胜节点index

# 参数设置
train_times = 1000  # 训练次数
data_dim = 2 # 数据维度
train_num = 160
test_num = 40
learn_rate = 0.5  # 学习参数

# 生成数据
random.seed()
# 生成训练数据
train_X = createData(train_num, data_dim)
# 生成测试数据
test_X = createData(test_num, data_dim)
# print(test_X)

# 初始化m个类
m = 15  # m个类别
layers = initOutputLayer(m, data_dim)
print("Original layers:", layers)

# 开始迭代训练
while train_times > 0:
    for i in range(train_num):
        # 权值归一化
        layers_norm = normalizationVList(layers)
        # 计算某一个x输入的竞争层胜利节点
        winner_index = getWinner(train_X[i], layers_norm)
        # 修正权值
        layers[winner_index] = adjustWeight(layers[winner_index], train_X[i], learn_rate)
    train_times -= 1
print("After train layers:", layers)

# 测试
for i in range(test_num):
    # 权值归一化
    layers_norm = normalizationVList(layers)
    # 计算某一个x输入的竞争层胜利节点
    winner_index = getWinner(test_X[i], layers_norm)
    # 画图
    color = "ro"
    if winner_index == 0:
        color = "ro"
    elif winner_index == 1:
        color = "bo"
    elif winner_index == 2:
        color = "yo"
    else:
        color='c'
    plt.plot(test_X[i][0], test_X[i][1], color)
plt.legend()
plt.show()
