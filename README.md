# 学习笔记
本仓库由LJTJ创建，LJTJ和WJY共同维护。
该仓库主要用于记录学习所得，内容为一家之言，仅供参考！
***
# 更新日志
## 2022-12-21
由W添加dl\_learnnote\dl\_learn\_note\RepVGG，内容包含RepVGG论文笔记、模型代码等，详情如下：
- code
  - ~~best.pth：Inter Image Classification使用模型RepVGG\_A1最佳训练权重，on train 94%，on val 89%。~~
  - model\_visualization.py：RepVGG\_A1模型可视化。
  - my\_train.py：模型训练脚本。
  - prediction.py：推理脚本。
  - repvgg.py：网络模型。
  - se\_block.py：部分RepVGG网络模型需使用SE模块。
- RepVGG.pdf：RepVGG论文笔记。
- RepVGG.pptx：论文汇报PPT。
- repvgg\_A1\_deploy.onnx.png：推理阶段模型可视化图。
- repvgg\_A1\_train.onnx.png：训练阶段模型可视化图。
## 2022-12-30
由W添加dl\_learnnote\dl\_learn\_note\ConvNeXt，内容包含ConvNeXt论文笔记、模型代码等，详情如下：
- code
  - convnext.py：网络模型。
- ConvNeXt.pdf：论文笔记。
- ConvNeXt.pptx：论文汇报PPT。
## 2023-01-4
LJTJ添加ghostnet相关代码，该代码为W注释的旧版本代码，同步仓库方便多端学习
## 2023-01-6
学习完SEnet 笔记同步（是否记录过于琐碎 以后还是一段时间的工作结束后一次性上传？ 但是多端同步就是为了方便平时记录 怕有什么忘了的）
## 2023-01-08
由W添加dl\_learnnote\dl\_learn\_note\trick\DataBlance&Augmentation，内容包含数据平衡与增强代码和方法介绍。详情如下：
- code
  - Deal.py：执行脚本。
  - ImageOperate.py：数据平衡和增强方法类，包含图像处理和方法定义。
- README.md：相关文档。
- 示例图片：文档中的图片。
## 2023-01-13
由W添加dl\_learnnote\dl\_learn\_note\YOLO series\YOLOv7\backbone，主要内容包含YOLOv7使用的backbone-ELAN的相关介绍。
- ELAN.pdf：论文笔记。
- ELAN.pptx：论文汇报PPT。

更新：添加了ConvNeXt模型的预训练权重，详情见本仓库下的ConvNeXt文件夹。

## 2023-01-15
GhostNet代码阅读完毕，在W写过的注释上完善。添加代码阅读笔记，标记我关注的几点内容，以后可能用的上。
## 2023-01-18
由W添加dl\_learnnote\dl\_learn\_note\trick\TricksForImageClassification，主要介绍CNN在分类任务上的训练技巧，论文提及了多种方法具有参考价值，详情如下：
- 论文笔记。
- 论文汇报PPT。

## 2023-01-21

由W添加dl\_learn\_note\dl\_learn\_note\MobileNetV1，主要包含了MobileNetV1相关内容，详情如下：

- MobileNetV1.pdf：论文笔记。
- MobileNetV1.pptx：论文汇报PPT。
- mobilenetv1.py：使用PyTorch构建的MobileNetV1网络模型。

如何使用我们提供的代码，下面给出示例：

```python
from mobilenet import *

# 提供了三种不同宽度MobileNetV1模型
net = MobileNetV1_100()

# 可以自定义网络宽度，以0.25为例
net = MobilNet(cfgs=cfgs, ratio=0.25, **kwargs)
```

## 2023-01-28

由W添加timm\，主要介绍了timm库的使用：

- README.md：《PyTorch 图像分类模型（timm）：实用指南》的Markdown文档。
- PyTorch 图像分类模型（timm）：实用指南.pdf：由Markdown文档导出的PDF文件。

## 2023-01-29

由W添加dl\_learn\_note\dl\_learn\_note\ShuffleNetV1，主要包含了ShuffleNetV1相关内容，详情如下：

- ShuffleNetV1.pdf：论文笔记。
- ShuffleNetV1.pptx：论文汇报PPT。
- shufflenetv1.py：基于PyTorch构建的ShuffleNetV1模型。

如何使用我们所提供的代码，下面给出示例：

```python
from shufflenetv1 import *

# 我们提供了网络宽度为0.5、1.0、1.5和2.0，以及分组数为3和8
net = ShuffleNet_050_g3()
```

注意：如果您想自定义网络宽度（例如0.75）或者增加新的分组（例如4），您可能需要修改代码，并在修改时注意各个卷积层的输入和输出通道数，这将导致某些通道数不符合缩放比例。

## 2023-04-07

LJ 添加翻译文档，pytorch.OPTIM官方文档中文翻译，torch2.0

## 2023-04-25

W创建Vision Transformer（长期更新）

**note** : 不再提供复现代码与训练脚本，请自信查阅官方代码仓库！仅提供论文部分注释和PPT。

已有内容：
- ViT
- DeiT
