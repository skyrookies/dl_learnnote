# Pytroch-write-and-read_note

视频仅做简单介绍，使用下面几个函数

```python
torch.save()
torch.load()
#也可以用列表或者字典做存储
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
```

模型参数可用下面几个函数保存读取，注意保存读取不仅要保存参数还需要网络结构

```
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

为了恢复模型，我们[**实例化了原始多层感知机模型的一个备份。**] 这里我们不需要随机初始化模型参数，而是(**直接读取文件中存储的参数。**)