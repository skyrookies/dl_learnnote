# pytorch基础(参数管理parameters)

#### 学习目标

* 访问参数，用于调试、诊断和可视化。
* 参数初始化。
* 在不同模型组件间共享参数。

单隐藏层MLP情况
进行参数访问

> weight 权重
>
> bias 偏移

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
print(net[2].state_dict())
#state_dict() 访问这个全连接层的权重和偏移，这个层8*1
#net[2]	访问第二个全连接层 即第三个网络（0，1，2）的状态参数，包括有权重和偏移

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
#访问具体的偏移数据类型、偏移量本身、偏移的值
```

```
OrderedDict([('weight', tensor([[ 0.3062, -0.1421, -0.0474, -0.2200, -0.2792,  0.1353,  0.0392,  0.3300]])), ('bias', tensor([-0.2103]))])
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([-0.2103], requires_grad=True)
tensor([-0.2103])
```

输出结果如上，参数是复合的对象，包含值、梯度和额外信息，也可访问每个参数的梯度
通过.grad 访问梯度

```python
net[2].weight.grad == None
```

#### 一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
#.named_parameters()函数一次性访问所有参数
```

输出结果

```
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
```

先访问第一个全连接层的参数，再访问到第二个全连接层的参数，中间那层为relu函数 没有参数

知道连接层的名字后，可以指定名字访问参数

```python
net.state_dict()['2.bias'].data
```

#### 块的嵌套中访问参数

先构造一个多层嵌套的块

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
        #f'block {i}' 用i参数将传入的块重命名，方便后续识别
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)#打印参数输入多层嵌套的这个层后的结果
print(rgnet)#打印该多层嵌套的层，结果如下
tensor([[-0.5894],
        [-0.5896]], grad_fn=<AddmmBackward0>)
#输入参数2*4，该嵌套层4*4，最后一个嵌套层4*1，最后输出2*1的结果
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
#由上面结果发现一层一层嵌套复杂而简单，故引出后文可用嵌套列表索引一样的方式访问具体的参数值，试例如下
rgnet[0][1][0].bias.data

```

#### 内置参数初始化

```python
def init_normal(m):
  #此处m 表示mudule 
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
#apply函数，将所有嵌套的块/层遍历执行一遍传入的函数
net[0].weight.data[0], net[0].bias.data[0]
```

> 以_为结尾的函数表示一种替换函数，对输入的参数进行替换没有输出

###### 对其中某几个层应用不同的初始化方法

```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

###### 自定义初始化

以下试例仅表示初始化函数的自由度

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

#### 参数绑定

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
#不管怎么更新，第三个层和第五个层是一样的参数
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

