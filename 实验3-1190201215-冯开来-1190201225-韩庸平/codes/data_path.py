import os
import math


def get_data():
    root = 'train/'
    path = os.listdir(root)
    path.sort()
    images_set = []         # 记录每个文件夹中图片的路径
    # 遍历所有文件夹
    for line in path:
        child_path = root + line
        image_set = []      # 记录一个文件夹中图片的路径
        images = os.listdir(child_path)
        # 遍历一个文件夹中的所有图片，获得相应的数目和路径
        for image in images:
            image_set.append(child_path + '/' + image)
        images_set.append(image_set)
    # 将每个文件夹中的图片划分出训练集、开发集和测试集
    train_data = []     # 训练集的路径
    labels_index = []    # 所有类别的label
    for img in images_set:
        train_data.extend(img)
    # 获得所有类别的label
    for data_path in train_data:
        label = data_path[6:-14]
        if label not in labels_index:
            labels_index.append(label)
    return train_data, labels_index


def get_test():
    root = 'test/'
    path = os.listdir(root)
    path.sort()
    test_set = []  # 记录一个文件夹中图片的路径
    for line in path:
        test_set.append(root + line)
    return test_set
