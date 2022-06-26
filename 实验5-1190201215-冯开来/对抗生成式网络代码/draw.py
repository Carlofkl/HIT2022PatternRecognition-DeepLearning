from matplotlib import pyplot as plt
import mat4py
import numpy as np
import torch

data = mat4py.loadmat("./dataset/points.mat")['xx']
data = np.array(data)


def draw_background(D, x_min, x_max, y_min, y_max):
    i = x_min
    background = []
    color = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            background.append([i, j])
            j += 0.01
        background.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        background.append([i, j])
        j += 0.01
        background.append([i, y_max])
    background.append([x_max, y_max])
    result = D(torch.Tensor(background).to("cuda:0"))
    for i in range(len(result)):
        if result[i] < 0.5:
            color.append('w')
        else:
            color.append('k')
    # print(result)
    background = np.array(background)
    plt.scatter(background[:, 0], background[:, 1], c=color)


def draw_scatter(D, xy, epoch, model):
    x = xy[:, 0]
    y = xy[:, 1]
    draw_background(D, -0.5, 2.2, -0.2, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-0.2, 1)
    plt.scatter(data[:, 0], data[:, 1], c='b', s=10)
    plt.scatter(x, y, c='r', s=10)
    plt.savefig("./result/" + model + '/epoch-' + str(epoch) + '.jpg')