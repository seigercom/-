# coding:utf-8
'''
MNIST下载数据集为http://yann.lecun.com/exdb/mnist
该代码参考https://blog.csdn.net/jiede1/article/details/77099326
'''
import numpy as np
import matplotlib.pyplot as plt
import struct

train_images_file = '../dataset/mnist/train-images-idx3-ubyte'
train_labels_file = '../dataset/mnist/train-labels-idx1-ubyte'
test_images_file = '../dataset/mnist/t10k-images-idx3-ubyte'
test_labels_file = '../dataset/mnist/t10k-labels-idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    images/=255.0
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_data(img_path = train_images_file, label_path = train_labels_file):
    '''
    :param img_path: 图片路径
    :param label_path: 标签路径
    :return: (N,width*height)图片，(N,)标签
    '''
    return decode_idx3_ubyte(img_path), decode_idx1_ubyte(label_path)

def load_test_data(img_path = test_images_file, label_path = test_labels_file):
    '''
    :param img_path: 图片路径
    :param label_path: 标签路径
    :return: (N,width*height)图片，(N,)标签
    '''
    return decode_idx3_ubyte(img_path), decode_idx1_ubyte(label_path)
def load_mnist_data():
    '''
    :return: 训练集和测试集
    '''
    return load_train_data(),load_test_data()
# if __name__=='__main__':
#     train_imgs,train_labels = load_train_data()
#     test_imgs, test_labels = load_test_data()
#     for i in range(10):
#         print(train_labels[i])
#         plt.imshow(train_imgs[i].reshape(28,28), cmap='gray')
#         plt.show()