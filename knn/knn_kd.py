from dataset.mnist.load_mnist import load_mnist_data
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree

def knn_pred(train,test,labels,k):
    '''
    :param train: 训练样本
    :param test: 测试样本
    :param labels: 训练标签
    :param k: 近邻个数
    :return: 预测标签
    '''
    pred_labels = []
    tree = KDTree(train)
    for i in range(len(test)):
        print(i)
        _, ind = tree.query(test[i].reshape(1,-1), k)
        label = [0] * 10
        for j in ind[0]:
            label[int(labels[j])] += 1
        pred_labels.append(label.index(max(label)))
    return pred_labels



if __name__=='__main__':
    time1 = time.time()
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0], train_data[1]
    test_imgs, test_labels = test_data[0], test_data[1]
    time2 = time.time()
    print('loading data ', time2 - time1, ' seconds\n')
    # knn not need training

    test_predict = knn_pred(train_imgs,test_imgs[:100],train_labels,k=10)
    time4 = time.time()
    print('predicting cost ', time4 - time2, ' second\n')

    score = accuracy_score(test_labels[:100], test_predict)
    print("The accruacy socre is ", score)