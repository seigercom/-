# coding=utf-8
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
class BasicClassifier(object):
    def __init__(self,features,labels,w):
        '''
        :param v: 阈值
        :param features: 特征
        :param labels: 标签
        '''
        self.v = list(set(features))
        self.features = features
        self.labels = labels
        self.w = w
    def _train_less_v(self):
        v = -1
        error_score = 1000000
        for i in self.v:
            score = 0
            for j in range(len(self.labels)):
                val = -1
                if self.features[j]<i:
                    val=1
                if val*self.labels[j]<0:
                    score+=self.w[j]
            if score<error_score:
                error_score = score
                v = i
        return v, error_score
    def _train_more_v(self):
        v = -1
        error_score = 1000000
        for i in self.v:
            score = 0
            for j in range(len(self.labels)):
                val = -1
                if self.features[j]>i:
                    val=1
                if val*self.labels[j]<0:
                    score+=1
            if score<error_score:
                error_score = score
                v = i
        return v, error_score
    def train(self):
        less_ind,less_score = self._train_less_v()
        more_ind,more_score = self._train_more_v()
        if less_score>more_score:
            self.flag='more'
            self.v = more_ind
            return more_score
        else:
            self.flag = 'less'
            self.v = less_ind
            return less_score
    def predict(self,X):
        if self.flag=='less':
            return 1 if X<self.v else -1
        else:
            return 1 if X>self.v else -1


class AdaBoosting:
    def __init__(self):
        pass
    def _init_parameters(self,features,labels,max_c=5):
        self.n = max_c
        self.X = features
        self.y = labels
        self.fea_dim = len(features[0])
        self.size = len(features)
        self.w = [1.0/self.size]*self.size
        self.alpha = []
        self.classifier = []
    def _w(self,index,classifier,i):
        return self.w[i]*np.exp(-self.alpha[-1]*self.y[i]*classifier.predict(self.X[i][index]))
    def _Z(self,index,classifier):
        z=0
        for i in range(self.size):
            z+=self._w(index,classifier,i)
        return z
    def train(self,features,labels):
        self._init_parameters(features,labels)
        for num in range(self.n):
            best_c =(10000,None,None) #误差，特征，分类器
            for i in range(self.fea_dim):
                features = self.X[:,i]
                classifier = BasicClassifier(features,self.y,self.w)
                error_score = classifier.train()
                if error_score<best_c[0]:
                    best_c = (error_score,i,classifier)
            em = best_c[0]
            if em==0:
                self.alpha.append(100)
            else:
                self.alpha.append(0.5*math.log((1-em)/em))
            self.classifier.append(best_c[1:])
            Z = self._Z(best_c[1],best_c[2])
            for i in range(self.size):
                self.w[i]=self._w(best_c[1],best_c[2],i)/Z
    def _predict(self,feature):
        result=0
        for i in range(self.n):
            ind = self.classifier[i][0]
            classifier = self.classifier[i][1]
            result+=self.alpha[i]*classifier.predict(feature[ind])
        if result>0:
            return 1
        else:
            return -1
    def predict(self,features):
        results=[]
        for feature in features:
            results.append(self._predict(feature))
        return results
if __name__=='__main__':
    wine = load_wine()
    wine.target[np.where(wine.target==2)]=1
    wine.target[np.where(wine.target == 0)] = -1
    train_x, test_x, train_y, test_y = train_test_split(wine.data, wine.target, test_size=0.3)
    ada = AdaBoosting()
    ada.train(train_x,train_y)
    test_pred = ada.predict(test_x)
    score = accuracy_score(test_y,test_pred)
    print(score)