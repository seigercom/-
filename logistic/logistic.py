import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston,load_wine
from sklearn.metrics import accuracy_score,mean_squared_error
from dataset.mnist.load_mnist import load_mnist_data

def logger(func):
    def wrapper(*args, **kwargs):
        print("%s is running\n" % func.__name__)
        time1 = time.time()
        ret =  func(*args, **kwargs)
        time2 = time.time()
        print('using %.3f seconds\n' %(time2-time1))
        return ret
    return wrapper
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (d, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (m,d),m为样本数
    Y -- 真实标签，shape： (m,1)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    #获取样本数m：
    m = X.shape[0]

    # 前向传播 ：
    A = sigmoid(np.matmul(X,w)+b)    #调用前面写的sigmoid函数
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m

    # 反向传播：
    dZ = A-Y
    dw = (np.matmul(X.T,dZ))/m
    db = (np.sum(dZ))/m

    #返回值：
    grads = {"dw": dw,
             "db": db}

    return grads, cost
@logger
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    #定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []
    #进行迭代：
    for i in range(num_iterations):
        # 用propagate计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        # 用上面得到的梯度来更新参数：
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # 每100次迭代，保存一个cost看看：
        if i % 100 == 0:
            costs.append(cost)

        # 这个可以不在意，我们可以每100次把cost打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
@logger
def predict(w,b,X):
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))

    A = sigmoid(np.dot(X,w)+b)
    for i in range(m):
        if A[i,0]>0.5:
            Y_prediction[i,0] = 1
        else:
            Y_prediction[i,0] = 0

    return Y_prediction


if __name__=='__main__':
    sub_num = 10000
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0][:sub_num], train_data[1]
    # train_features = get_hog_features(train_imgs)
    test_imgs, test_labels = test_data[0], test_data[1]
    # test_features = get_hog_features(test_imgs)
    train_labels_ = np.array([int(x > 0) for x in train_labels[:sub_num]])
    test_labels_ = np.array([int(x > 0) for x in test_labels])
    w,b = initialize_with_zeros(train_imgs.shape[1])
    param, grads,costs = optimize(w,b,train_imgs,train_labels_,20,0.001)
    pred = predict(param['w'],param['b'],test_imgs)
    score = accuracy_score(pred,test_labels_)
    print('Accuracy is ',score)
