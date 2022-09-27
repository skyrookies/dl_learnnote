# -*- coding: utf-8 -*-
# @Time : 2022/8/26 13:49
# @Author : ljtj_test
# @Site : 
# @File : test_cuda.py
# @Software: PyCharm 
# @coding_meaning:
#
'''测试环境cuda是否正常运行'''

import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("驱动为：",device)
print("GPU型号： ",torch.cuda.get_device_name(0))

