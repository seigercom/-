from dataset.mnist.load_mnist import load_mnist_data
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score


num_class = 10
feature_len = 784

def to_binary(img):
    img[img>0.5]=1
    img[img<=0.5]=0
    return img

def train(train,train_labels):
    prior_prob = np.zeros(num_class)
    conditional_prob = np.zeros((num_class,feature_len,2))
    for i in range(len(train_labels)):
        img = to_binary(train[i])
        label = int(train_labels[i])
        prior_prob[label]+=1
        for j in range(feature_len):
            conditional_prob[label][j][int(img[j])]+=1
    for i in range(num_class):
        for j in range(feature_len):
            pix_0 = conditional_prob[i][j][0]
            pix_1 = conditional_prob[i][j][1]

            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0) / float(pix_0 + pix_1)) * 1000000 + 1
            probalility_1 = (float(pix_1) / float(pix_0 + pix_1)) * 1000000 + 1

            conditional_prob[i][j][0] = probalility_0
            conditional_prob[i][j][1] = probalility_1
    return prior_prob,conditional_prob

def calculate_probability(img,label):
    probability = int(prior_prob[label])

    for i in range(len(img)):
        probability *= int(conditional_prob[label][i][int(img[i])])

    return probability

def Predict(testset,prior_probability,conditional_probability):
    predict = []

    for img in testset:

        # 图像二值化
        img = to_binary(img)

        max_label = 0
        max_probability = calculate_probability(img,0)

        for j in range(1,10):
            probability = calculate_probability(img,j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict)

if __name__=='__main__':
    time1 = time.time()
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0], train_data[1]
    test_imgs, test_labels = test_data[0], test_data[1]
    time2 = time.time()
    print('loading data ', time2 - time1, ' seconds\n')
    prior_prob, conditional_prob = train(train_imgs,train_labels)
    time3 = time.time()
    print('training ',time3-time2,' seconds\n')
    test_pred = Predict(test_imgs,prior_prob,conditional_prob)
    time4 = time.time()
    print('predicting ',time4-time3,' seconds\n')
    score = accuracy_score(test_labels,test_pred)
    print('accuracy score is ',score)


