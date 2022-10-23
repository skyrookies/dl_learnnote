# ghostnet论文阅读_10_25

知识蒸馏Knowledge Distillation

包含模型压缩、迁移学习与多教师信息融合



FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。

ResNet是一种残差网络，残差：观测值与估计值之间的差
ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度
ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习
https://www.zhihu.com/question/64494691/answer/786270699

## 实验结论

实验结果表明，所提出的Ghost模块能够在保持相似识别性能的同时降低通用卷积层的计算成本，并且GhostNet可以超越MobileNetV3等先进的高效深度模型，在移动设备上进行快速推断。

## 实验核心（自己觉得）

鉴于主流CNN计算出的中间特征图中存在大量的冗余，提出减少用于生成它们的卷积核所需的资源，使用少数原始特征图通过一些廉价转换生成输出特征图。

## ghostnet构成

基于Ghost bottleneck。GhostNet主要由一堆Ghost bottleneck组成，其中Ghost bottleneck以Ghost模块为构建基础。第一层是具有16个卷积核的标准卷积层，然后是一系列Ghost bottleneck，通道逐渐增加。这些Ghost bottleneck根据其输入特征图的大小分为不同的阶段。除了每个阶段的最后一个Ghost bottleneck是stride = 2，其他所有Ghost bottleneck都以stride = 1进行应用。最后，利用全局平均池和卷积层将特征图转换为1280维特征向量以进行最终分类。SE模块也用在了某些Ghost bottleneck中的残留层，如表1中所示。与MobileNetV3相比，这里用ReLU换掉了Hard-swish激活函数。

> ghost bottleneck 

## ImageNet分类数据集中的实验

结果中我们可以看到，通常较大的FLOPs会在这些小型网络中带来更高的准确性，这表明了它们的有效性。而GhostNet在各种计算复杂度级别上始终优于其他竞争对手，主要是因为GhostNet在利用计算资源生成特征图方面效率更高