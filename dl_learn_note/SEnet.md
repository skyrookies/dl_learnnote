# SEnet

What is a SEnet?

Squeeze-and- Excitation（挤压和激发）模块

SE模块首先对卷积得到的特征图进行Squeeze操作，得到channel级的全局特征，然后对全局特征进行Excitation操作，学习各个channel间的关系，也得到不同channel的权重，最后乘以原来的特征图得到最终特征
本质上，SE模块是在channel维度上做attention或者gating操作，这种注意力机制让模型可以更加关注信息量最大的channel特征，而抑制那些不重要的channel特征。

### 背景

最后一届ImageNet冠军 CVPR2017文章 
重点是senet思路简单，容易扩展到已有网络结构中
其创新点在于关注channel之间的关系

### 作用

机制让模型可以更加关注信息量最大的channel特征，而抑制那些不重要的channel特征

### 实现

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 创建一个自适应平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False)
          # 输入的特征数为channel，输出的特征数为channel // reduction
            nn.ReLU(inplace=True)
            nn.Linear(channel // reduction, channel, bias=False) 
          # 输入的特征数为channel // reduction，输出的特征数为channel
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取张量x的形状
        b, c, _, _ = x.size()
        # 进行平均池化并将结果展开成二维张量
        y = self.avg_pool(x).view(b, c)
        # 使用全连接层对y进行线性变换并激活
        y = self.fc(y).view(b, c, 1, 1)
        # 将y和x对应元素相乘，实现对通道注意力机制在特征图上的应用
        x = torch.mul(x, y)
        return x

```



### 小结

SE模块主要为了提升模型对channel特征的敏感性，这个模块是轻量级的，而且可以应用在现有的网络结构中，只需要增加较少的计算量就可以带来性能的提升

### 注意点

在初次阅读这个代码时，没注意到channel//reduction 干嘛用，经过仔细阅读和提问chatGPT
我发现参数`channel // reduction`表示全连接层输出的特征数。
我们之所以使用整数除法运算符，是因为这个参数必须是整数，而不是浮点数

而被除数reduction=16也表示在SE操作的示意图中channel先除16进行Squeeze，而后再恢复的过程

## 参考文档：

知乎：[﻿最后一届ImageNet冠军模型：SENet - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/65459972)

论文原文：[[1709.01507\] Squeeze-and-Excitation Networks (arxiv.org)](https://arxiv.org/abs/1709.01507)

