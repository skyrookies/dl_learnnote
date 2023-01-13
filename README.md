# 学习笔记
本仓库由LJTJ创建，LJTJ和WJY共同维护。
该仓库主要用于记录学习所得，内容为一家之言，仅供参考！
***
# 更新日志
## 2022-12-21
由W添加dl\_learnnote\dl\_learn\_note\RepVGG，内容包含RepVGG论文笔记、模型代码等，详情如下：
- code
  - best.pth：Inter Image Classification使用模型RepVGG\_A1最佳训练权重，on train 94%，on val 89%。
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







