from __future__ import division
from torch import nn
from utils import *


class EmptyLayer(nn.Module):
    """
    仅仅是一个空模块，在shortcut层和route层使用可以大大简化代码编写
    """

    def __init__(self):
        """
        初始化函数，进行初始化处理
        """
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """
    用于检测层，是一个单独的类
    """

    def __init__(self, anchors):
        """
        对层进行初始化处理
        Args:
            anchors: 对anchors进行定义初始化
        """
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfg_file):
    """
    读取配置文件，解析配置文件，并将每个模块存储为字典的list
    Args:
        cfg_file: 配置文件路径

    Returns:
        表示模块参数的字典构成的 list
    """

    file = open(cfg_file, 'r')
    # 将配置文件的行表示成一个list，后续处理将在该list上进行
    lines = file.read().split('\n')
    # 去除空行，注释，两侧的blank
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        # 表示一个新模块的开始
        if line[0] == '[':
            # 如果block不是空，则表示其存储了签一个模块的值
            if len(block) != 0:
                # 将前一个模块加进blocks
                blocks.append(block)
                # 重新初始化block
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    """
    根据blocks指示的模块参数进行模块的构建
    Args:
        blocks: 需要建立模型的所有block参数字典列表

    Returns:
        通过这个函数构建好神经网络的模型
    """
    net_info = blocks[0]
    # 获取关于输入和预处理的信息，在迭代blocks之前，用该变量存储关于网络的信息
    # 该方法返回这个list，这个类类似一个包含nn.Module元素的正常的list
    # 但是将nn.ModuleList看做nn.Module的成员添加时，nn.Modulelist中成员的参数会被当做nn.Module的参数
    module_list = nn.ModuleList()
    # 定义一个卷积层的时候，必须定义卷积核的维度。尽管卷积核的宽度和高度在cfg文件中设置，卷积核的深度就是前一层
    # 的filters的个数，因此我们需要记录卷积层所在的层的filters的数量。使用prev_filters变量来完成该功能。
    # 将prev_filters初始化为3，表示RGB三个通道
    prev_filters = 3
    # Route之后如果有卷积层，则卷积层需要使用Route带来的特征，因此需要保存所有层的filters的数量
    output_filters = []

    # 迭代blocks中的所有模块，并且为每个模块创建nn.Module；在迭代过程中，要根据block的类型创建响应的模块，并添加到module_list中
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # 如果block是一个卷积层
        if x['type'] == 'convolutional':
            # 获得关于这层的信息
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            # 是否有padding填充处理
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # 添加BN层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # 检查激活函数
            # YOLO中可能是线性函数或者是Leaky ReLU
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)
        # 如果是Route层
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # 路由层的起点
            start = int(x['layers'][0])
            # 如果已经存在就停止
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            # 构建route层
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
                # 这一部分对应跳过连接，就是一个empty层
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        # Yolo层，也就是用于探测的层
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            # 定义探测检测层
            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)
        # 如果是上采样层
        # 使用Bilinear2dUpsampling
        elif x['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{}'.format(index), upsample)

        module_list.append(module)  # 最后将模块添加到module_list中
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
    """
    为探测器构造的类
    """

    def __init__(self, cfg_file):
        """
        初始化函数，对网络进行初始化
        Args:
            cfg_file: 读取配置文件进行初始化
        """
        super(Darknet, self).__init__()
        # 读取网络中的每一个块
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """
        该函数实现两个功能：一是计算输出；二是将探测器输出的特征图转换成能够更加容易被处理的形式
        （例如对特征图做转换令不同尺度的特征图能够彼此拼接，否则如果他们维度不同，是无法进行拼接的）
        Args:
            x: 输入
            CUDA: 如果该参数是True，则使用GPU；如果是False，则使用CPU

        Returns:
            返回前向检测的运算结果
        """
        # 从blocks[1]开始迭代，因为blocks[0]是一个网络，不是前向传播的模块
        modules = self.blocks[1:]
        # 对路由层的输出进行缓存，供后续层使用（键值是层的索引，值是特征图）
        outputs = {}

        # 表示我们是否遇到了第一个detection；值为0表示collector还没有初始化；是1表示已经初始化了，则可以
        # 直接将特征图和它相连
        write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])
            # 卷积层或者上采样层
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            # route层
            elif module_type == 'route':
                # 处理路由层连接一个层的特征图或者两个层的特征图的情况
                layers = module['layers']
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            # shortcut层，即一个empty层
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
            # YOLO的输出时一个卷积特征映射，包含沿特征映射深度的边界框属性，神经元预测的边界框的属性被一个一个地堆叠在一起
            # 所以，如果想要获得（5，6）位置处的神经元预测的第二个边界，需要使用map[5, 6, (5+c) : 2*(5+c)]来索引
            # 这种形式对于输出处理非常不方便，如通过对象置信阈值，向中心添加网格偏移量，应用锚定等
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # 提取输入的维度
                inp_dim = int(self.net_info['height'])

                # 提取识别和检测的类别数量
                num_classes = int(module['classes'])

                # 进行张量运算
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                #  若还没初始化，则需要进行初始化
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections

    def load_weights(self, weight_file):
        """
        加载模型的权重
        Args:
            weight_file: 权重的配置文件

        Returns:
            配置权重之后就是权重配置的网络
        """
        fp = open(weight_file, 'rb')

        # 首先提取5个头部文件的信息
        # 1. 获取主版本号
        # 2. 获取次版本号
        # 3. 获得子版本号的数量
        # 4,5. 训练过程中被模型识别的图像
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # 提取权重信息
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            # 卷积层则提取这些参数信息，否则进行忽略
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    # 获得BN层的参数信息
                    num_bn_biases = bn.bias.numel()

                    # 加载参数，并对参数进行处理
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 将加载的weight作为模型的weight
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # 将数据载入模型中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    # 将weights信息加载进来
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # 根据模型的设置，对参数进行reshape
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 最终将数据进行复制
                    conv.bias.data.copy_(conv_biases)

                # 为卷积层加载模型参数和信息
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
