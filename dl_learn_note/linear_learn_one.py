#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 14:34
# @Author  : lanjiang
# @File    : linear_learn_one.py
# @Description :再次尝试从零实现线性回归，参考B站up 跟李沐学AI
import matplotlib
matplotlib.use('TkAGG')
import random
import torch
from d2l import torch as d2l

'''第一个函数 生成 𝐲=𝐗𝐰+𝑏+𝜖 的随机噪声'''

def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,size=(num_examples,len(w)))
    '''normal 函数原型torch.normal(means, std, out=None)
    means (Tensor) – 均值
    std (Tensor) – 标准差
    out (Tensor) – 可选的输出张量'''
    y=torch.matmul(X,w)+b
    '''matmul是tensor的乘法，输入可以是高维的'''
    y+=torch.normal(0,0.01,y.shape)
    '''shape tensor类的实例属性  返回torch.Size([2, 3])'''
    return X,y.reshape((-1,1))
'''reshape((-1,1)) 将y作为列向量返回，-1表示可以有任意多个，1表示仅为一位'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

"""测试函数 查看生成的噪音"""
#print('features:', features[0],'\nlabel:', labels[0])
#d2l.set_figsize()
#d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
'''MACOS下plt报错 暂且不修正'''

'''定义接受函数，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量'''
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    '''将num_examples序列化，能够被循环读取'''
    random.shuffle(indices)
    '''打乱序列顺序，实现随机读取'''
    for i in range(0,num_examples,batch_size):
        '''range(start,stop,step)'''
        batch_indices=torch.tensor(
        indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
'''对接受函数测试'''
batch_size=10
#for X,y in data_iter(batch_size,features,labels):
#    print(X,'\n',y)
#    break

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')