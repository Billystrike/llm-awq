import torch
import torch.nn as nn
a=torch.arange(15,dtype=torch.float16).reshape(3,5)
b=[]
# b.append(a[0,None,:])
# b.append(a[1,None,:])
b.append(a[None,0].unsqueeze(0))
b.append(a[None,1].unsqueeze(0))
# b.append(a[None,0])
# b.append(a[None,1])
print(b[0].shape)
b=torch.cat(b,dim=0)
print(b.shape)
print(torch.amax(b,2))
print(torch.amax(b,2).shape)
print(torch.amax(b,0,keepdim=True).shape)

print('22')
from accelerate import init_empty_weights, load_checkpoint_and_dispatch








# DSA部分

#test
print('*'*40)
# 创建一个test_cuda.py文件
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'设备数量: {torch.cuda.device_count()}')
from torch.utils.cpp_extension import CUDA_HOME
print(f'PyTorch检测到的CUDA_HOME: {CUDA_HOME}')
print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"}')

import awq_inference_engine
