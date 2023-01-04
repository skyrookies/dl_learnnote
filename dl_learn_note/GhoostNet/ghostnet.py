# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""

# Chinese code notes by WangJY

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# __all__ = ['ghost_net']

# 该函数可以通过设置divisor，使得通道数可以被divisor（或其倍数）整除
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# hard-Sigmoid函数时Sigmoid激活函数的分段线性近似。从公示和曲线上来看，其更易计算，因此会提高训练的效率
# 使用ReLU6而不是ReLU的原因是：主要是为了在移动端float16的低精度的时候，也能有很好的数值分辨率，
# 如果对ReLu的输出值不加限制，那么输出范围就是0到正无穷，而低精度的float16无法精确描述其数值，带来精度损失。
# inplace：是否直接对原始值进行修改
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

# SE模块：学习了channel之间的相关性，筛选出了针对通道的注意力，并融合到特征图中
# SE模块的目的：增强重要的特征，削弱不重要的特征
# SE模块源自Squeeze-and-Excitation Networks（SENet）
# SENet论文：https://arxiv.org/abs/1709.01507
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        # 这一步操作是确保reduced_chs可以被8整除
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # 使用自适应平均全局池化，输出：B*C*1*1
        x_se = self.avg_pool(x)
        # 降维，减少计算量，输出：B*reduced_chs*1*1
        x_se = self.conv_reduce(x_se)
        # ReLU一下
        x_se = self.act1(x_se)
        # 恢复成原始输入通道数，输出：B*in_chs*1*1
        x_se = self.conv_expand(x_se)
        # 经过gate_fn函数，并融合到原始输入特征图中
        x = x * self.gate_fn(x_se)
        return x


# 主要功能是960*1*1->1280*1*1
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

# GhostModel主要作用是生成特征图
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup

        # math.ceil(x)->返回大于等于x的一个整数
        # init_channels：原始固有特征图数量
        # new_channels：廉价操作生成的特征图数量
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            # inp：输入通道数；
            # init_channels：输出通道数
            # kernel_size：卷积核大小
            # stride：卷积步长
            # kernel_size//2：填充
            # bias：偏置值，如果卷积后需要进行BN操作，则应当设置bias值为False。
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),

            # 对特征图进行归一化处理，使其满足标准正态分布，需要提供上一步操作后的图像通道数作为参数
            nn.BatchNorm2d(init_channels),

            # 使用ReLU作为激活函数，inplace=True表示结果将替换原有输入
            # if语句：relu控制是否进行ReLU
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            # init_channels：输入通道数
            # new_channels：输出通道数
            # dw_size：卷积核大小
            # 1：stride，步长大小
            # dw_size//2：padding，填充
            # groups=init_channels，分组数等于输入通道数，此卷积演化为深度卷积
            # bias：偏置值
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # out[:, :self.oup, :, :]保证输出张量的通道数与设定的输出通道数一致
        return out[:, :self.oup, :, :]


# 瓶颈层
# 相关结构，详情可见：论文中Figure3
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        # 利用GhostModel生成特征图
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution（深度可分离卷积）
        # 只要stride=2时执行
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        # SE模块
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        # 根据论文，第二个GhostModel后不接ReLU
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        # 此操作与ResNet中的残差模块相似，主要作用帮助训练
        # 有关残差模块，可见ResNet论文：https://arxiv.org/abs/1512.03385
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            # 该处操作主要用于规整特征图
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution（深度可分离卷积）
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    # width对应论文中Width Multiplier小节中的α
    def __init__(self, cfgs, num_classes=3, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

# GhostNet中Ghost-bneck相关参数
def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # K：卷积核大小
        # t：第一个GhostModule输出通道数大小
        # c：瓶颈层最终输出通道数
        # SE：是否使用SE注意力模块
        # s：瓶颈层的步长
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == '__main__':
    model = ghostnet()
    model.eval()
    print(model)
    input = torch.randn(32, 3, 320, 256)
    y = model(input)
    print(y.size())

