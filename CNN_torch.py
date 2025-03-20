import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__();

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, #输入图片的深度
                out_channels = 16, #输出图片的深度
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            # 输出 [16,28,28]，
            nn.ReLU(),
            # 池化层，2X2取最大值
            nn.MaxPool2d(kernel_size=2),
            # 输出[16,14,14]
        );

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, #输入图片的深度
                out_channels = 32, #输出图片的深度
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            # 输出 [32,14,14]，
            nn.ReLU(),
            # 池化层，2X2取最大值
            nn.MaxPool2d(kernel_size=2),
            # 输出[32,7,7]
        );

        # 输出层,flatten后输入，输出10个数字的概率
        self.output = nn.Linear(in_features=32*7*7,out_features=10);

    def forward(self,x):
        x = self.conv1(x);
        x = self.conv2(x);
        x = x.view(x.size(0),-1); #保留batch，将最后乘到一起[batch, 32*7*7]
        output = self.output(x);
        return output;