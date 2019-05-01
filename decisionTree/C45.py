from dataset.mnist.load_mnist import load_mnist_data
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score
import logging
import threading
num_class = 10
feature_len = 784



def to_binary(img):
    img[img>0.3]=1
    img[img<=0.3]=0
    return img

def cal_entropy(train_labels):
    ent = 0
    for i in range(num_class):
        pi = train_labels[train_labels==i].shape[0]/(train_labels.shape[0])
        ent += -pi*np.log(pi+1e-6)  # add 1e-6 to avoid log(0)
    return ent

def cal_con_entropy(x,y):
    fea = set(x)
    c_ent = 0
    for f in fea:
        y_sub = y[x==f]
        ent = cal_entropy(y_sub)
        c_ent += y_sub.shape[0]/y.shape[0]*ent
    return c_ent

class Tree(object):
    def __init__(self,node_type,Class = None, feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        if self.node_type == 'leaf':
            return self.Class

        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def train_decisionTree(train,labels,features,epsilon):
    global num_class
    # define types of tree
    LEAF = 'leaf'
    OTHER = 'other'

    label_set = set(labels)
    # 1.judge if all samples belong to the same class
    if len(label_set)==1:
        return Tree(LEAF,Class=label_set.pop())
    # 2.judge if features is empty
    label,label_c = np.unique(labels,return_counts=True)
    max_label = label[np.argsort(label_c)[-1]]
    if len(features)==0:
        return Tree(LEAF,Class=max_label)
    # 3.cal gain index
    max_fea = 0
    max_gain_index = 0
    for fea in features:
        sub_set = np.array(train[:,fea].flat)
        H = cal_entropy(labels)
        gain_index = (H-cal_con_entropy(sub_set,labels))/H
        if max_gain_index<gain_index:
            max_fea=fea
            max_gain_index = gain_index
    # 4.judge whether gain less than epsilon
    if max_gain_index<=epsilon:
        return Tree(LEAF, Class=max_label)
    # 5.construct sub tree
    sub_features = list(filter(lambda x:x!=max_fea,features))
    tree = Tree(OTHER,feature=max_fea)
    sub_set = np.array(train[:, max_fea].flat)
    feature_value_list = set(sub_set)
    for k in feature_value_list:
        sub_ind = np.where(sub_set==k)
        sub_train = train[sub_ind]
        sub_labels = labels[sub_ind]
        sub_tree = train_decisionTree(sub_train, sub_labels, sub_features, epsilon)
        tree.add_tree(k,sub_tree)
    return tree

def predict(test_set,tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)



if __name__=='__main__':
    time1 = time.time()
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0], train_data[1]
    test_imgs, test_labels = test_data[0], test_data[1]
    time2 = time.time()
    print('loading data ',time2-time1,' seconds\n')
    train_features = to_binary(train_imgs)
    test_features = to_binary(test_imgs)
    tree = train_decisionTree(train_features, train_labels, list(range(784)), 0.01)
    time3 = time.time()
    print('training decision tree ', time3 - time2, ' seconds\n')
    test_predict = predict(test_features, tree)
    score = accuracy_score(test_labels, test_predict)
    time4 = time.time()
    print('predicting test data ', time4 - time3, ' seconds\n')
    print('Accuracy ',score)

