# 숙제

# 1. Conv2D Backward 구현
# Forward 에서 쓴 im2col을 이용해 gradient 계산
# pytorch nn.conv2d와 비교해서 상대오차 < 1e-3 달성

# 2. Maxpool 구현 (forward, backward)
# forward : 영역 내 최댓값 / backward : 최댓값 위치로 gradient 전달
# pytorch nn.maxpool2d와 비교 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Optional, List
from dataclasses import dataclass, field

import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Conv:
    def __init__(self, stride = 1, pad = 0):
        self.stride = stride
        self.pad = pad

    def im2col(self, input_data, filter_h, filter_w):
        N, C, H, W = input_data.shape 
        out_h = (H + 2 * self.pad - filter_h) // self.stride + 1 
        out_w = (W + 2 * self.pad - filter_w) // self.stride + 1 

        img = np.pad(input_data, [(0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)], 'constant') 
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) 

        for y in range(filter_h): 
            y_max = y + self.stride * out_h 
            for x in range(filter_w): 
                x_max = x + self.stride * out_w 
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3)
        # TODO: gradient 계산 추가
        col = col.reshape(N * out_h * out_w , -1) # col의 크기를 변환
        return col

    # convolution forward
    def conv_forward(self, x, W, b):
        FN, C, FH, FW = W.shape
        N, _, H, W_ = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W_ + 2 * self.pad - FW) // self.stride + 1

        col = self.im2col(x, FH, FW)
        col_W = W.reshape(FN, -1).T
        out = np.dot(col, col_W) + b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    # pytorch nn.conv2d convolution forward
    def conv_forward2(self, x, W, b):
        # tensor로 변환
        x_t = torch.from_numpy(x).float()
        W_t = torch.from_numpy(W).float()
        b_t = torch.from_numpy(b).float()
        # weight tensor shape : out channel, in channel, kernel height, kernel width
        o_C, i_C, k_h, k_w = W_t.shape
        
        # kernel size에는 height가 들어가야 함. 왜지?!!!?!?!?
        conv = nn.Conv2d(in_channels = i_C, out_channels = o_C, 
                        kernel_size = k_h, stride = self.stride, 
                        padding = self.pad)
        return conv(x_t)



@dataclass
class TrainingConfig:
    model = None
    epochs : int = 30
    seed : int = 42
    learning_rate : float = 1e-4
    batch_size : int = 64
    hidden_layers : List[int] = field(default_factory = lambda : [128, 64])

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")


class Trainer: 
    def __init__(self, model, train_loader, test_loader, optim, l_f,
                 # scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 # callbacks : Optional[List[BaseCallback]] = None,
                 device : str = "cuda"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = optim
        self.l_f = l_f
        #self.scheduler = scheduler
        #self.callbacks = callbacks if callbacks else []
        self.device = device
        #self.state = {}

    def fit(self, num_epochs : int, config : TrainingConfig):
        # 모델 학습 부분
        for i in range(num_epochs):
            for idx, data in enumerate(self.train_loader):
                input, output = data


# MNIST 불러오기
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./train_data", train=True, download=True, transform=transform)

conv = Conv(stride = 1, pad = 1)

# 샘플 이미지 확인
x, y = mnist[0]
print("Label : ", y)
print("Shape : ", x.shape)  # torch.Size([1,28,28])

# numpy 변환 후 convolution 적용
x_np = x.unsqueeze(0).numpy()  # (1,1,28,28)
W = np.random.randn(3, 1, 3, 3) * 0.01 # 아무 숫자나 넣은 거임.
b = np.zeros(3)

out = conv.conv_forward(x_np, W, b)
out2 = conv.conv_forward2(x_np, W, b)
                     
print("Conv output shape:", out.shape)  # (1,3,28,28)
print("Conv output shape:", out2.shape)