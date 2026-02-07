# ModelTool/Model.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class LWP(nn.Module):  # Lightweight Parameter Layer
    def __init__(self, num_features, eps=1e-5):
        super(LWP, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, num_features, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True)  # over time
        std = x.std(dim=-1, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        return self.scale * x_norm + self.shift

class TemporalBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, dilation=1, padding=0, dropout=0.2, mode='pre'):
        super(TemporalBlock_v2, self).__init__()
        self.mode = mode
        padding = (kernel_size - 1) * dilation  # 保证因果

        # 第一层 conv + LWP + ReLU + Dropout
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.lwp1 = LWP(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.lwp2 = LWP(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_final = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def get_trainable_params(self):
        """根据mode返回需要训练的参数列表"""
        if self.mode == 'pre':
            # 预训练模式：训练所有参数
            return list(self.parameters())
        else:
            # 元学习模式：只训练LWP层的参数，冻结卷积层
            trainable_params = []
            trainable_params.extend(list(self.lwp1.parameters()))
            trainable_params.extend(list(self.lwp2.parameters()))
            return trainable_params

    def forward(self, x):
        residual = x

        # 第一层
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.lwp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # 第二层
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.lwp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 残差
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = self.relu_final(out)

        return out