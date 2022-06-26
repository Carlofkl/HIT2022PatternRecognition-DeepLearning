from __future__ import division
from torch.autograd import Variable
from utils import *
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

images = 'imgs'
batch_size = 1
confidence = 0.5
# NMS方法阈值选择
nms_thesh = 0.4
# 选择CUDA进行测试和检测
device = torch.cuda.is_available()
# coco数据集可以识别的种类是80
num_classes = 80
classes = load_classes('data/coco.names')
# 加载神经网络，初始化参数
model = Darknet('cfg/yolov3.cfg')
model.load_weights('cfg/yolov3.weights')
model.net_info['height'] = '416'
inp_dim = int(model.net_info['height'])


def write_a(x, filename, results):
    """
    处理图像文件，并且将实验结果写回det文件夹中
    Args:
        x: 输入的文件
        filename: 文件名称
        results: 需要进行处理图像的整体

    Returns:
        处理之后的图像进行写回
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = '{0}'.format(classes[cls])
    # 勾勒矩形轮廓
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, 2)
    # 写入类别
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 2)
    # 将处理的结果进行写回
    cv2.imwrite(filename[int(x[0])], img)
    return img


# 使用CUDA进行训练
if device:
    model.cuda()

# 将模型变成测试，不再进行训练改变参数
model.eval()

# 对测试图像的检测阶段
im_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
# 建立检测之后的图像文件夹
if not os.path.exists('det'):
    os.makedirs('det')
# 加载测试图像
loaded_ims = [cv2.imread(x) for x in im_list]
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(im_list))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(im_list) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]

write = 0
# 设置CUDA的使用
if device:
    im_dim_list = im_dim_list.cuda()

for i, batch in enumerate(im_batches):
    # 加载图像，使用CUDA
    if device:
        batch = batch.cuda()
    with torch.no_grad():
        # 进行模型测试，得到预测值
        prediction = model(Variable(batch), device)
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
    if type(prediction) == int:
        for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
            im_id = i * batch_size + im_num
        continue
    prediction[:, 0] += i * batch_size
    # 输出初始化
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
    # 设置CUDA的使用
    if device:
        torch.cuda.synchronize()

try:
    # 输出内容
    output
except NameError:
    print('No detections were made')
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)
output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

# 从已有模型加载
colors = pkl.load(open('pallete', 'rb'))

# 处理得到处理文件的名字
det_names = pd.Series(im_list).apply(lambda x: './{}/det_{}'.format('det', x.split('/')[-1]))
# 将文件名进行简化，存储在目标文件夹中
for i in range(len(det_names)):
    # 将所有\进行替换，然后将:也进行替换
    det_names[i] = det_names[i].replace('\\', '_')
    det_names[i] = det_names[i].replace(':', '')
    # 利用split操作得到一个简化名字
    simple_name = det_names[i].split('_')
    det_names[i] = './det/' + simple_name[len(simple_name) - 1]
# 对图像进行探测处理
list(map(lambda x: write_a(x, det_names, loaded_ims), output))
list(map(cv2.imwrite, det_names, loaded_ims))

# 清空缓存，实验探测处理完成
torch.cuda.empty_cache()
