# TORCH.OPTIM（torch 2.0）

[`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can also be easily integrated in the future.

是一个实现各种优化算法的包。 大多数常用的方法都已经支持，接口也足够通用，以便将来可以轻松集成更复杂的方法。

## How to use an optimizer

To use [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) you have to construct an optimizer object that will hold the current state and will update the parameters based on the computed gradients.

要使用 [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) 你必须构建一个优化器对象来保持当前状态并更新参数 基于计算出的梯度。

### Constructing it

To construct an [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) you have to give it an iterable containing the parameters (all should be `Variable` s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.

要构造一个 [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) 你必须给它一个包含参数的迭代器（都应该是 `Variable` s） 优化。 然后，您可以指定特定于优化器的选项，例如学习率、权重衰减等。

Example:

```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```



### Per-parameter options 每个参数选项

[`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) s also support specifying per-parameter options. To do this, instead of passing an iterable of `Variable` s, pass in an iterable of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) s. Each of them will define a separate parameter group, and should contain a `params` key, containing a list of parameters belonging to it. Other keys should match the keyword arguments accepted by the optimizers, and will be used as optimization options for this group.

[`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) 还支持指定每个参数选项。 为此，不要传递 `Variable` 的迭代，而是传递 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) 的迭代。 它们中的每一个都将定义一个单独的参数组，并且应该包含一个 params 键，包含属于它的参数列表。 其他键应该匹配优化器接受的关键字参数，并将用作该组的优化选项。

NOTE

You can still pass options as keyword arguments. They will be used as defaults, in the groups that didn’t override them. This is useful when you only want to vary a single option, while keeping all others consistent between parameter groups.

您仍然可以将选项作为关键字参数传递。 在没有覆盖它们的组中，它们将用作默认值。 当您只想改变一个选项，同时保持参数组之间的所有其他选项一致时，这很有用。

For example, this is very useful when one wants to specify per-layer learning rates:

例如，当想要指定每一层的学习率时，这非常有用：

```
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```



This means that `model.base`’s parameters will use the default learning rate of `1e-2`, `model.classifier`’s parameters will use a learning rate of `1e-3`, and a momentum of `0.9` will be used for all parameters.

这意味着 `model.base` 的参数将使用默认的学习率 `1e-2`，`model.classifier` 的参数将使用 `1e-3` 的学习率和 `0.9` 的动量 ` 将用于所有参数。

### Taking an optimization step 采取优化措施

All optimizers implement a [`step()`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) method, that updates the parameters. It can be used in two ways:

#### `optimizer.step()`

This is a simplified version supported by most optimizers. The function can be called once the gradients are computed using e.g. `backward()`.

所有优化器都实现 [`step()`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) 方法，更新参数 . 它可以通过两种方式使用：

这是大多数优化器支持的简化版本。 一旦使用例如计算梯度，就可以调用该函数的方法 `backward()`。

Example:

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```



#### `optimizer.step(closure)`

Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.

一些优化算法，如 Conjugate Gradient 和 LBFGS 需要多次重新计算函数，所以你必须传入一个闭包，让它们重新计算你的模型。 闭包应该清除梯度，计算损失，并返回它

Example:

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

## Base class 基础类

- *CLASS*torch.optim.Optimizer(*params*, *defaults*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer)

  Base class for all optimizers.

  WARNING

  Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that don’t satisfy those properties are sets and iterators over values of dictionaries.Parameters:**params** (*iterable*) – an iterable of [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) s or [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) s. Specifies what Tensors should be optimized.**defaults** – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).

  需要将参数指定为具有在运行之间保持一致的确定性排序的集合。 不满足这些属性的对象示例是字典值的集合和迭代器。参数：**params** (*iterable*) – [`torch.Tensor`](https://pytorch.org /docs/stable/tensors.html#torch.Tensor) s 或 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) s。 指定应优化哪些张量。**默认值** –（字典）：包含优化选项默认值的字典（在参数组未指定它们时使用）。

| [`Optimizer.add_param_group`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.add_param_group.html#torch.optim.Optimizer.add_param_group) | Add a param group to the [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) s param_groups. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`Optimizer.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict) | Loads the optimizer state.                                   |
| [`Optimizer.state_dict`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict) | Returns the state of the optimizer as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict). |
| [`Optimizer.step`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) | Performs a single optimization step (parameter update).      |
| [`Optimizer.zero_grad`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) | Sets the gradients of all optimized [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) s to zero. |

## Algorithms 算法

| [`Adadelta`](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta) | Implements Adadelta algorithm.                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`Adagrad`](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad) | Implements Adagrad algorithm.                                |
| [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) | Implements Adam algorithm.                                   |
| [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) | Implements AdamW algorithm.                                  |
| [`SparseAdam`](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam) | Implements lazy version of Adam algorithm suitable for sparse tensors. |
| [`Adamax`](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax) | Implements Adamax algorithm (a variant of Adam based on infinity norm). |
| [`ASGD`](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD) | Implements Averaged Stochastic Gradient Descent.             |
| [`LBFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS) | Implements L-BFGS algorithm, heavily inspired by [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html). |
| [`NAdam`](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam) | Implements NAdam algorithm.                                  |
| [`RAdam`](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam) | Implements RAdam algorithm.                                  |
| [`RMSprop`](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop) | Implements RMSprop algorithm.                                |
| [`Rprop`](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop) | Implements the resilient backpropagation algorithm.          |
| [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) |                                                              |

Many of our algorithms have various implementations optimized for performance, readability and/or generality, so we attempt to default to the generally fastest implementation for the current device if no particular implementation has been specified by the user.

我们的许多算法都有针对性能、可读性和/或通用性优化的各种实现，因此如果用户未指定特定实现，我们会尝试默认为当前设备的通常最快的实现。

We have 3 major categories of implementations: for-loop, foreach (multi-tensor), and fused. The most straightforward implementations are for-loops over the parameters with big chunks of computation. For-looping is usually slower than our foreach implementations, which combine parameters into a multi-tensor and run the big chunks of computation all at once, thereby saving many sequential kernel calls. A few of our optimizers have even faster fused implementations, which fuse the big chunks of computation into one kernel. We can think of foreach implementations as fusing horizontally and fused implementations as fusing vertically on top of that.

我们有 3 大类实现：for 循环、foreach（多张量）和融合。 最直接的实现是对参数进行大量计算的 for 循环。 For 循环通常比我们的 foreach 实现要慢，后者将参数组合成一个多张量并同时运行大量计算，从而节省了许多顺序内核调用。 我们的一些优化器具有更快的融合实现，将大量计算融合到一个内核中。 我们可以将 foreach 实现视为水平融合，将融合实现视为垂直融合。

In general, the performance ordering of the 3 implementations is fused > foreach > for-loop. So when applicable, we default to foreach over for-loop. Applicable means the foreach implementation is available, the user has not specified any implementation-specific kwargs (e.g., fused, foreach, differentiable), and all tensors are native and on CUDA. Note that while fused should be even faster than foreach, the implementations are newer and we would like to give them more bake-in time before flipping the switch everywhere. You are welcome to try them out though!

总的来说，这 3 种实现的性能顺序是 fused > foreach > for-loop。 因此，在适用的情况下，我们默认使用 foreach 而不是 for 循环。 适用意味着 foreach 实现可用，用户没有指定任何特定于实现的 kwargs（例如，融合、foreach、可微分），并且所有张量都是本机的并且在 CUDA 上。 请注意，虽然 fused 应该比 foreach 更快，但实现更新，我们希望在各处翻转开关之前给它们更多的烘焙时间。 不过，欢迎您尝试一下！

Below is a table showing the available and default implementations of each algorithm:

下表显示了每种算法的可用和默认实现:

| Algorithm                                                    | Default  | Has foreach? | Has fused? |
| ------------------------------------------------------------ | -------- | ------------ | ---------- |
| [`Adadelta`](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta) | foreach  | yes          | no         |
| [`Adagrad`](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad) | foreach  | yes          | no         |
| [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) | foreach  | yes          | yes        |
| [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) | foreach  | yes          | yes        |
| [`SparseAdam`](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam) | for-loop | no           | no         |
| [`Adamax`](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax) | foreach  | yes          | no         |
| [`ASGD`](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD) | foreach  | yes          | no         |
| [`LBFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS) | for-loop | no           | no         |
| [`NAdam`](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam) | foreach  | yes          | no         |
| [`RAdam`](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam) | foreach  | yes          | no         |
| [`RMSprop`](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop) | foreach  | yes          | no         |
| [`Rprop`](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop) | foreach  | yes          | no         |
| [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) | foreach  | yes          | no         |

## How to adjust learning rate 如何调整学习率

`torch.optim.lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs. [`torch.optim.lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) allows dynamic learning rate reducing based on some validation measurements.

`torch.optim.lr_scheduler` 提供了几种根据 epoch 数调整学习率的方法。 [`torch.optim.lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) 允许基于动态学习率降低 在一些验证测量上。

Learning rate scheduling should be applied after optimizer’s update; e.g., you should write your code this way:

优化器更新后应应用学习率调度； 例如，您应该这样编写代码：:

Example:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

Most learning rate schedulers can be called back-to-back (also referred to as chaining schedulers). The result is that each scheduler is applied one after the other on the learning rate obtained by the one preceding it.

大多数学习率调度器都可以称为背靠背（也称为链接调度器）。
 结果是每个调度器一个接一个地应用在前一个调度器获得的学习率上。

Example:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```

In many places in the documentation, we will use the following template to refer to schedulers algorithms.

在文档中的许多地方，我们将使用以下模板来引用调度程序算法:

```
>>> scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```



WARNING

Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before the optimizer’s update; 1.1.0 changed this behavior in a BC-breaking way. If you use the learning rate scheduler (calling `scheduler.step()`) before the optimizer’s update (calling `optimizer.step()`), this will skip the first value of the learning rate schedule. If you are unable to reproduce results after upgrading to PyTorch 1.1.0, please check if you are calling `scheduler.step()` at the wrong time.

在 PyTorch 1.1.0 之前，学习率调度器应该在优化器更新之前被调用； 1.1.0 以打破 BC 的方式改变了这种行为。 如果您在优化器更新（调用 optimizer.step() ）之前使用学习率调度程序（调用 scheduler.step() ），这将跳过学习率计划的第一个值。 如果您在升级到 PyTorch 1.1.0 后无法重现结果，请检查您是否在错误的时间调用了 scheduler.step() 。

| [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR) | Sets the learning rate of each parameter group to the initial lr times a given function. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR) | Multiply the learning rate of each parameter group by the factor given in the specified function. |
| [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) | Decays the learning rate of each parameter group by gamma every step_size epochs. |
| [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR) | Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. |
| [`lr_scheduler.ConstantLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR) | Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters. |
| [`lr_scheduler.LinearLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR) | Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters. |
| [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR) | Decays the learning rate of each parameter group by gamma every epoch. |
| [`lr_scheduler.PolynomialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR) | Decays the learning rate of each parameter group using a polynomial function in the given total_iters. |
| [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR) | Set the learning rate of each parameter group using a cosine annealing schedule, where ����*η**ma**x* is set to the initial lr and ����*T**c**u**r* is the number of epochs since the last restart in SGDR:使用余弦退火计划设置每个参数组的学习率，其中 η   设置为初始 lr 和 T是自 SGDR 中上次重启以来的纪元数 |
| [`lr_scheduler.ChainedScheduler`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler) | Chains list of learning rate schedulers.                     |
| [`lr_scheduler.SequentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR) | Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch. |
| [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) | Reduce learning rate when a metric has stopped improving.    |
| [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR) | Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR). |
| [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR) | Sets the learning rate of each parameter group according to the 1cycle learning rate policy. |
| [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) | Set the learning rate of each parameter group using a cosine annealing schedule, where ����*η**ma**x* is set to the initial lr, ����*T**c**u**r* is the number of epochs since the last restart and ��*T**i* is the number of epochs between two warm restarts in SGDR:<br />使用余弦退火计划设置每个参数组的学习率，其中 η max 设置为初始 lr，T cur 是自上次重启以来的纪元数，Ti 是 SGDR 中两次热重启之间的纪元数： |

## Stochastic Weight Averaging 随机权重平均

`torch.optim.swa_utils` implements Stochastic Weight Averaging (SWA). In particular, `torch.optim.swa_utils.AveragedModel` class implements SWA models, `torch.optim.swa_utils.SWALR` implements the SWA learning rate scheduler and `torch.optim.swa_utils.update_bn()` is a utility function used to update SWA batch normalization statistics at the end of training.

SWA has been proposed in [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407).

`torch.optim.swa_utils` 实现随机权重平均 (SWA)。 

特别是，`torch.optim.swa_utils.AveragedModel` 类实现 SWA 模型，

`torch.optim.swa_utils.SWALR` 实现 SWA 学习率调度程序，

`torch.optim.swa_utils.update_bn()` 是一个实用函数，用于 在训练结束时更新 SWA 批量归一化统计数据。

[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) 中提出了 SWA。

### Constructing averaged models 构建平均模型

AveragedModel class serves to compute the weights of the SWA model. You can create an averaged model by running:

AveragedModel 类用于计算 SWA 模型的权重。 您可以通过运行以下命令创建平均模型：

```
>>> swa_model = AveragedModel(model)
```

Here the model `model` can be an arbitrary [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) object. `swa_model` will keep track of the running averages of the parameters of the `model`. To update these averages, you can use the `update_parameters()` function:

这里的模型 `model` 可以是任意的 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) 对象。 `swa_model` 将跟踪 `model` 参数的运行平均值。 要更新这些平均值，您可以使用 update_parameters() 函数：

```
>>> swa_model.update_parameters(model)
```



### SWA learning rate schedules SWA学习率调整

Typically, in SWA the learning rate is set to a high constant value. `SWALR` is a learning rate scheduler that anneals the learning rate to a fixed value, and then keeps it constant. For example, the following code creates a scheduler that linearly anneals the learning rate from its initial value to 0.05 in 5 epochs within each parameter group:

通常，在 SWA 中，学习率被设置为一个较高的常数值。 `SWALR` 是一种学习率调度程序，它将学习率退火到固定值，然后保持恒定。 例如，以下代码创建了一个调度程序，它在每个参数组内的 5 个时期内将学习率从其初始值线性退火到 0.05：

```
>>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
>>>         anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)
```

You can also use cosine annealing to a fixed value instead of linear annealing by setting `anneal_strategy="cos"`.

您还可以通过设置 `anneal_strategy="cos"` 使用余弦退火到固定值而不是线性退火。

### Taking care of batch normalization 处理批量归一化

`update_bn()` is a utility function that allows to compute the batchnorm statistics for the SWA model on a given dataloader `loader` at the end of training:

`update_bn()` 是一个实用函数，允许在训练结束时计算给定数据加载器 `loader` 上 SWA 模型的 batchnorm 统计数据：

```
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
```

`update_bn()` applies the `swa_model` to every element in the dataloader and computes the activation statistics for each batch normalization layer in the model.

`update_bn()` 将 `swa_model` 应用于数据加载器中的每个元素，并计算模型中每个批量归一化层的激活统计信息。

WARNING

`update_bn()` assumes that each batch in the dataloader `loader` is either a tensors or a list of tensors where the first element is the tensor that the network `swa_model` should be applied to. If your dataloader has a different structure, you can update the batch normalization statistics of the `swa_model` by doing a forward pass with the `swa_model` on each element of the dataset.

`update_bn()` 假定数据加载器 `loader` 中的每个批次都是张量或张量列表，其中第一个元素是网络 `swa_model` 应该应用于的张量。 如果您的数据加载器具有不同的结构，您可以通过对数据集的每个元素执行前向传递来更新 swa_model 的批量归一化统计信息。

### Custom averaging strategies 自定义平均策略

By default, `torch.optim.swa_utils.AveragedModel` computes a running equal average of the parameters that you provide, but you can also use custom averaging functions with the `avg_fn` parameter. In the following example `ema_model` computes an exponential moving average.

默认情况下，`torch.optim.swa_utils.AveragedModel` 计算您提供的参数的运行相等平均值，但您也可以使用带有 `avg_fn` 参数的自定义平均函数。 在以下示例中，`ema_model` 计算指数移动平均线。

Example:

```
>>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
>>>         0.1 * averaged_model_parameter + 0.9 * model_parameter
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
```



### Putting it all together 

In the example below, `swa_model` is the SWA model that accumulates the averages of the weights. We train the model for a total of 300 epochs and we switch to the SWA learning rate schedule and start to collect SWA averages of the parameters at epoch 160:

在下面的示例中，`swa_model` 是累积权重平均值的 SWA 模型。 我们训练模型总共 300 个时期，然后切换到 SWA 学习率计划并开始收集第 160 个时期参数的 SWA 平均值：

```python
>>> loader, optimizer, model, loss_fn = ...
>>> swa_model = torch.optim.swa_utils.AveragedModel(model)
>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
>>> swa_start = 160
>>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>       if epoch > swa_start:
>>>           swa_model.update_parameters(model)
>>>           swa_scheduler.step()
>>>       else:
>>>           scheduler.step()
>>>
>>> # Update bn statistics for the swa_model at the end
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
>>> # Use swa_model to make predictions on test data
>>> preds = swa_model(test_input)
```