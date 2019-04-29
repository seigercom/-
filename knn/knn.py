from dataset.mnist.load_mnist import load_mnist_data
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score
import heapq
def knn_pred(train,test,labels,k):
    '''
    :param train: 训练样本
    :param test: 测试样本
    :param labels: 训练标签
    :return: 测试标签
    '''
    test_labels = []

    for i in range(len(test)):
        print(i)
        x = np.repeat(test[i],len(train)).reshape(-1,len(train)).T
        distance = list(np.sqrt(np.sum(np.square(x-train),axis=1)))
        index = map(distance.index,heapq.nsmallest(k,distance))
        label = [0]*10
        for j in index:
            label[int(labels[j])]+=1
        test_labels.append(label.index(max(label)))
    return test_labels

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