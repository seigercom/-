from dataset.mnist.load_mnist import load_mnist_data
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score

def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features


class Perceptron(object):
    def __init__(self):
        self.lr = 0.0001
        self.max_itr = 5000

    def _predict(self,x):
        '''
        :param x: sample
        :return: label
        '''
        wx = np.dot(self.w,x)
        return int(wx>0)

    def train(self,features,labels):
        self.w = [0.]*(len(features[0])+1)
        itr = 0
        correct_count = 0
        max_num = len(labels)
        while itr<self.max_itr:
            #使用随机梯度训练
            index = np.random.randint(0,len(labels)-1)
            # 这里注意不要改变原来数组
            x = list(features[index])
            x.append(1.)
            if labels[index]>0:
                y=1.
            else:
                y=-1.
            wx = np.dot(self.w,x)

            if wx*y<0:
                correct_count+=1
                if correct_count>=max_num:
                    break
                continue
            for i in range(len(self.w)):
                self.w[i] += self.lr*(y*x[i])
            itr+=1

    def predict(self,features):
        labels = []
        for f in features:
            x = list(f)
            x.append(1.)
            labels.append(self._predict(x))
        return labels

if __name__=='__main__':
    time1 = time.time()
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0], train_data[1]
    # train_features = get_hog_features(train_imgs)
    test_imgs, test_labels = test_data[0], test_data[1]
    # test_features = get_hog_features(test_imgs)
    test_labels_ = np.array([int(x > 0) for x in test_labels])
    time2 = time.time()
    print('loading data ',time2-time1,' seconds\n')
    p = Perceptron()
    p.train(train_imgs,train_labels)
    time3 = time.time()
    print('training data ', time3 - time2, ' seconds\n')
    test_predict = p.predict(test_imgs)
    time4 = time.time()
    print('predicting cost ', time4 - time3, ' second\n')

    score = accuracy_score(test_labels_, test_predict)
    print("The accruacy socre is ", score)

# 不用HOG提取特征0.902
# 使用后0.902