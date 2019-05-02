import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston,load_wine
import time
from sklearn.metrics import accuracy_score,mean_squared_error

def logger(func):
    def wrapper(*args, **kwargs):
        print("%s is running\n" % func.__name__)
        time1 = time.time()
        ret =  func(*args, **kwargs)
        time2 = time.time()
        print('using %.3f seconds\n' %(time2-time1))
        return ret
    return wrapper

class Tree(object):
    def __init__(self,type,c=None,feature=None,split=None):
        self.type = type
        self.c = c
        self.feature = feature
        self.split = split
        self.left = None
        self.right = None
    def pred(self,features):
        if self.type == 'LEAF':
            return self.c
        fea = features[self.feature]
        if fea<=self.split:
            tree = self.left
        else:
            tree = self.right
        return tree.pred(features)



def cal_mse(x,y):
    min_mse = np.iinfo(np.int32).max
    best_split = 0
    for i in set(x):
        index1 = np.where(x <= i)
        index2 = np.where(x >i)
        sub_y1 = y[index1]
        c_y1 = np.mean(sub_y1)
        sub_y2 = y[index2]
        c_y2 = np.mean(sub_y2)
        mse = np.sum(np.square(sub_y1-np.repeat(c_y1,len(index1))))+\
            np.sum(np.square(sub_y2 - np.repeat(c_y2, len(index2))))
        if min_mse > mse:
            best_split = i
            min_mse = mse
    return min_mse, best_split

def cal_gini(x,y):
    min_gini = np.iinfo(np.int32).max
    best_split = 0
    for i in set(x):
        index1 = np.where(x<=i)
        index2 = np.where(x>i)
        sub_y1 = y[index1]
        _,c_y1 = np.unique(sub_y1,return_counts=True)
        sub_y2 = y[index2]
        _,c_y2 = np.unique(sub_y2,return_counts=True)
        gini = index1[0].shape[0]/len(y)*(1-np.sum(np.square(c_y1/index1[0].shape[0]))) +\
               index2[0].shape[0]/len(y)*(1-np.sum(np.square(c_y2/index2[0].shape[0])))
        if min_gini>gini:
            best_split = i
            min_gini = gini
    return min_gini,best_split

def train_Cart(train_x,train_y,features,flag,s_threshold,l_threshold):
    # # sometimes train_x is empty, I don't konw how to deal with it
    # if len(train_x)==0:
    #     return Tree('LEAF',c=0)

    if flag == 'D':
        # judge whether number of samples is less than s_threshold
        if train_x.shape[0] <= s_threshold or not features:
            all_c,cc = np.unique(train_y,return_counts=True)
            lc = all_c[np.argsort(cc)[-1]]
            return Tree('LEAF',c=lc)
        best_f = features[0]
        best_split = train_x[0][0]
        min_los = np.iinfo(np.int32).max
        for f in features:
            tmp_los, tmp_split = cal_gini(train_x[:, f], train_y)
            if min_los > tmp_los:
                best_f = f
                best_split = tmp_split
                min_los = tmp_los
        # judge whether loss is less than l_threshold
        if min_los<l_threshold:
            all_c, cc = np.unique(train_y, return_counts=True)
            lc = all_c[np.argsort(cc)[-1]]
            return Tree('LEAF', c=lc)
        # sub_features = list(filter(lambda x: x != best_f, features))
        l_x = train_x[train_x[:,best_f]<=best_split]
        l_y = train_y[train_x[:,best_f]<=best_split]
        r_x = train_x[train_x[:, best_f] > best_split]
        r_y = train_y[train_x[:, best_f] > best_split]
        if l_y.shape[0]<=0 or r_y.shape[0]<=0:
            all_c, cc = np.unique(train_y, return_counts=True)
            lc = all_c[np.argsort(cc)[-1]]
            return Tree('LEAF', c=lc)
        else:
            tree = Tree('OTHER', feature=best_f, split=best_split)
            tree.left = train_Cart(l_x, l_y, features, flag, s_threshold, l_threshold)
            tree.right = train_Cart(r_x, r_y, features, flag, s_threshold, l_threshold)
            return tree

    elif flag=='R':
        if train_x.shape[0] <= s_threshold or not features:
            return Tree('LEAF',c=np.mean(train_y))
        best_f = features[0]
        best_split = train_x[0][0]
        min_los = np.iinfo(np.int32).max
        for f in features:
            tmp_los, tmp_split = cal_mse(train_x[:, f], train_y)
            if min_los > tmp_los:
                best_f = f
                best_split = tmp_split
                min_los = tmp_los
        # judge whether loss is less than l_threshold
        if min_los < l_threshold:
            all_c, cc = np.unique(train_y, return_counts=True)
            lc = all_c[np.argsort(cc)[-1]]
            return Tree('LEAF', c=lc)
        # sub_features = list(filter(lambda x: x != best_f, features))
        l_x = train_x[train_x[:, best_f] <= best_split]
        l_y = train_y[train_x[:, best_f] <= best_split]
        r_x = train_x[train_x[:, best_f] > best_split]
        r_y = train_y[train_x[:, best_f] > best_split]
        if l_y.shape[0]<=0 or r_y.shape[0]<=0:
            return Tree('LEAF',c=np.mean(train_y))
        else:
            tree = Tree('OTHER', feature=best_f, split=best_split)
            tree.left = train_Cart(l_x, l_y, features, flag, s_threshold, l_threshold)
            tree.right = train_Cart(r_x, r_y, features, flag, s_threshold, l_threshold)
            return tree
    else:
        raise Exception('flag should be D or R')

@logger
def train(train_set,train_label,features, flag='D',s_threshold=5,l_threshold=0.0001):
    return train_Cart(train_set,train_label,features,flag,s_threshold,l_threshold)

@logger
def predict(test_set,tree):
    result = []
    for features in test_set:
        tmp_predict = tree.pred(features)
        result.append(tmp_predict)
    return np.array(result)


if __name__ =='__main__':
    boston = load_boston()
    wine = load_wine()
    train_x,test_x,train_y,test_y = train_test_split(boston.data,boston.target,test_size=0.3)
    # train_x, test_x, train_y, test_y = train_test_split(wine.data, wine.target, test_size=0.3)
    features = list(range(train_x.shape[1]))
    d_tree = train(train_x,train_y,features,flag='R')
    pred_y = predict(test_x,d_tree)
    score = mean_squared_error(test_y,pred_y)
    # print('Accuracy is %.3f' %score)
    print('MSE is %.3f' % score)
