# Ghostnet代码阅读1.13

​	Ghostnet的核心是用一些线形操作代替普通的卷积，生成更多的特征图，其网络结构也参考了mobileNetV2和mobileNetV3，也使用了SE模块（通道注意力机制），深度可分离卷积，shortcut操作。这些操作在如今见怪不怪，但是代码阅读中，参数在单独一个模块使用**在定义时放入参数信息第一次见，可以方便修改网络参数。

​		在代码中发现实现所谓线形操作的代码就是深度卷积（Depth-wise convolution）再加上大量训练的tricks，当然，ghostnet是在2020年提出的。以今天的视角来看有不足是正常的。而使用**放入参数的操作是第一眼看的误解，代码中是表示在函数调用时传递的关键字参数。表示所有传入函数的关键字参数。网格参数的传递是单独定义了一个二位列表。

### 使用**传参数，使用列表统一传参

```python
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
```

### 设置标志位，在类似结构中简化代码

在该模块中，会用到使用relu和不使用relu但是总体结构一样的前后调用，在构造函数时加上inplace标志位

```python
nn.ReLU(inplace=True) if relu else nn.Sequential(),
```

### 规整通道数

这个函数，"_make_divisible"，用于确保给定值 "v" 能被指定的除数整除。它通过首先将最小值设置为除数，然后取最小值和将除数的一半加到 "v" 上之后的结果的最大值，然后除以除数并向下取整到最接近的除数倍数。然后它检查新值是否小于原始值的90％，如果是，它会在新值上加上除数。该函数的目的是确保模型中的所有层的通道数都能被8整除。这是从TensorFlow存储库的MobileNet实现中获取的。

```python
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
```

