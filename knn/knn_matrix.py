from dataset.mnist.load_mnist import load_mnist_data
import numpy as np
import time
if __name__=='__main__':

    test_subset_size = 10000
    train_subset_size = 60000
    train_data, test_data = load_mnist_data()
    train_imgs, train_labels = train_data[0][:train_subset_size], train_data[1][:train_subset_size]
    test_imgs, test_labels = test_data[0][:test_subset_size], test_data[1][:test_subset_size]

    train_n = train_imgs.shape[0]
    test_n = test_imgs.shape[0]

    time1 = time.time()
    train_norm = np.sum(train_imgs * train_imgs, axis=1)
    train_norm = np.reshape(train_norm, (train_n, 1))
    test_norm = np.sum(test_imgs * test_imgs, axis=1)
    test_norm = np.reshape(test_norm, (test_n, 1))

    dist = np.repeat(train_norm, test_n, axis=1) + np.repeat(test_norm, train_n, axis=1).T - \
    2 * np.matmul(train_imgs, test_imgs.T)
    for knn in range(1, 20, 2):
    #knn = 10
        sorted = np.argsort(dist, axis=0)
        pred = train_labels[sorted[:knn, :]]
        time2 = time.time()
        pred_labels = []
        for idx in range(test_n):
            u, u_c = np.unique(pred[:, idx], return_counts=True)
            pred_labels.append(u[np.argmax(u_c)])
        print ("%d, Accuracy:%.3f" % (knn, np.mean(pred_labels==test_labels)))
        print(time2-time1)