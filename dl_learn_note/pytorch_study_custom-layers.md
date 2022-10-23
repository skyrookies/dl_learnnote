# pytorch_study_custom-layers自定义层

#### 不带参数的层

在pytorch中 自定义的层首先还是一个从mudule继承的类，只需继承基础mudele类即可实现前向传播并进行简单处理
下列试例简单实验了一个从输入中减去均值的前向传播函数

```python
import torch
import torch.nn.functional as F
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
#super()函数用法见模块构造，调用父类的初始化函数
    def forward(self, X):
    #修改forward函数
        return X - X.mean()
```

该自定义的层可被sequencial函数整合作为组建构建到更复杂的函数中并使用，试例如下

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())
```

运行结果如下

```
tensor(3.7253e-09, grad_fn=<MeanBackward0>)
```

虽然函数本身应该减去其自身的平均值使平均值为0，但是由于计算精度，仍然有较小的误差 

#### 带参数的图层

层数本身应该都是Parameter这个类的实例，故自定带参数的layer时即调用Parameter类的方法去实现，试例如下：

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```



总结，自定义层用起来和自定义网络本身没有区别，都是继承mudule类，再重构所需要的参数。也可用来构造复杂网络，根据弹幕大神评价，可去pytorch官网查询文档选择需要的一些函数。原视频内容不做更细致记录，简单过即可。
