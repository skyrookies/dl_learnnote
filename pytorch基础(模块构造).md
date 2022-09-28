# pytorch基础(模块构造)

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        super().__init__()
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        #将内部的一些参数进行初始化，设置好
         # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        self.hidden = nn.Linear(20, 256)  
        # 隐藏层 全连接层 输入维度20，输出256
        #将该全连接层保存在类的成员变量中
        self.out = nn.Linear(256, 10)  
        # 输出层 同理
        #对于简单MLP所需所有层均在该函数中

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
      #输入放入hidden层中，然后调用relu函数处理输入
      #其中relu函数是调用nn模组中F已经实现的函数，在F中还有大量的其他实现的函数。观察包头 F 为torch.nn中导入的functional
      #完成激活后放入输出 out
```

> python调用父类函数关键词 super()说明
>
> super() 函数是用于调用父类(超类)的一个方法。
>
> super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
>
> ```tsx
> super(type[, object-or-type])
> ```
>
> - type -- 类。
>
> - object-or-type -- 类，一般是 self
>
>   单继承时可直接使用super()表示父类，但多继承时会涉及MRO(**继承父类方法时的顺序表**) 的调用排序问题
>
> `super().__init__`相对于`类名.__init__`，在单继承上用法基本无差
>
> 但在多继承上有区别，`super`方法能保证每个父类的方法只会执行一次，而使用类名的方法会导致方法被执行多次，可以尝试写个代码来看输出结果
>
> 多继承时，使用`super`方法，对父类的传参数，应该是由于python中super的算法导致的原因，必须把参数全部传递，否则会报错
>
> 单继承时，使用`super`方法，则不能全部传递，只能传父类方法所需的参数，否则会报错
>
> 多继承时，相对于使用`类名.__init__`方法，要把每个父类全部写一遍, 而使用super方法，只需写一句话便执行了全部父类的方法，这也是为何多继承需要全部传参的一个原因
>
> 使用`super()`注意，所有的父类都应该加上不定参数`*args , **kwargs` ，不然参数不对应是会报错的

测试

```
net = MLP()
net(X)
```

将MLP实例化，然后放入随机数据向量测试

## 实现sequential类

> Sequential 按顺序的，序列的 类似于容器 可用来包装各层（百度

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
# 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
# 变量_modules中。_module的类型是OrderedDict
#即按顺序将引入的*args（放入的层） 放入_module这个顺序字典中

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    #这里是按顺序将引入的层，一层一层的叠加嵌套，起到类似容器的效果
```

> enumerate()函数，将可遍历的数据对象组合为索引序列
>
> 本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置

> OrderedDict 有序字典，如果其顺序不同那么Python也会把他们当做是两个不同的对象

测试实现的sequential函数Mysequential()

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

展示的fixedhiddenMLP函数仅为显示在实际工作过程中，若不使用sequential函数，可以自定各种实例操作，更大自由度的实现，此处代码分析暂略。（注意是否参与反向传播，requires_grad值取False

## 嵌套使用各种组合块

各种块、层均由nn.mudule 继承而来，下列函数展示如何使用

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
          nn.Linear(20, 64), nn.ReLU(),
          nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(
 NestMLP(), nn.Linear(16, 20), FixedHiddenMLP()
)
chimera(X)
```

