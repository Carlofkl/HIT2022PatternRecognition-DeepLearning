from __future__ import division

import torch
import numpy as np
import cv2


# 该文件中包含各种辅助函数的代码，为了完成模型的加载以及图像的检测

def letterbox_image(img, inp_dim):
    """
    填充图像，用不变化的比例
    Args:
        img: 输入的图像
        inp_dim: 变化的比例

    Returns:
        返回填充之后结果
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


def unique(tensor):
    """
    去重操作
    Args:
        tensor: 待处理张量，是一个图片中的预测结果

    Returns:  去重之后的张量，即一个类别值保留一个

    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)  # 对np.array()进行去重操作
    unique_tensor = torch.from_numpy(unique_np)  # 将np.array()转换成张量

    tensor_res = tensor.new(unique_tensor.shape)  # 创建一个新的张量
    tensor_res.copy_(unique_tensor)  # 拷贝
    return tensor_res


def load_classes(name_file):
    """
    Args:
        name_file: 文件名，包含了可以检测的类别名称

    Returns:
        将每个类的索引映射到类名字符串的字典
    """
    fp = open(name_file, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def bbox_iou(box1, box2):
    """
    返回两个边界框的重叠之处所占比例
    Args:
        box1: 边界框1
        box2: 边界框2

    Returns:
        返回重叠面积占真实边界框box2面积的比例
    """
    # 得到边界的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 求出两个矩形相交的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 求解交叉区域
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # 合并区域
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    该函数将特征映射prediction转换成一个二维张量，二维张量的每一行表示一个边界框的属性
    Args:
        prediction: 待转换的特征映射
        inp_dim: 输入维度大小
        anchors: 先验框
        num_classes: 预测的类的总数
        CUDA: 选用CPU或者GPU，当CUDA是True的时候用GPU，False的时候使用CPU

    Returns: prediction对应的二维张量

    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # 对对象进行sigmoid函数处理
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 加上中心的偏移量
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    # 使用CUDA进行测试
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    # 将高度宽度变换到对数空间
    anchors = torch.FloatTensor(anchors)
    # 使用CUDA
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride  # 将特征映射的大小调整为和图像大小一致
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    对特征映射prediction做客观评分阈值处理和非极大值抑制，获得真实的预测结果
    Args:
        prediction: 特征映射
        confidence: 分数阈值
        num_classes: 类别总数
        nms_conf: IOU阈值

    Returns: 真实预测结果

    """
    # 挑选出得分大于阈值的边界框
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # 将边界框使用四个角的坐标进行描述，方便进行IOU的计算
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]
    batch_size = prediction.size(0)
    write = False  # 表示是否对output进行了初始化
    # 每个边界框有85个属性值，其中的80个是分数，循环中的处理将分数全部清除，但是保留分数最大值的索引以及最大分数值
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # NMS方法进行处理
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # 对象置信度小于阈值的都要去除
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # 获取图像中检测到的类别
        img_classes = unique(image_pred_[:, -1])
        # 非极大值抑制操作
        for cls in img_classes:

            # 对一个特定的类的检测结果
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # 对检测进行结果排序，得到最大可能性
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            # 检测到的数量
            idx = image_pred_class.size(0)
            # 对每个检测内容循环处理
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # 检测大于阈值的交叉比例
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # 移除部分项
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)
            seq = batch_ind, image_pred_class
            # 检测是否初始化
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


def prep_image(img, inp_dim):
    """
    将输入神经网络的图像进行预处理，将其转换成一个张量
    Args:
        img: 输入图像
        inp_dim: 维度

    Returns: 处理好的图片

    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
