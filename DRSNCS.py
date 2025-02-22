# File: model/DRSNCS.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedThreshold(nn.Module):
    """通道共享阈值生成模块"""
    def __init__(self, channels, groups=16, reduction=16):
        super().__init__()
        self.groups = groups
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 每组共享一个阈值
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, groups),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        gap = self.gap(x).view(b, c)
        
        # 生成组阈值
        group_threshold = self.fc(gap)  # [B, groups]
        
        # 将阈值分配到各个通道
        threshold = group_threshold.unsqueeze(2).repeat(1, c//self.groups, 1)
        threshold = threshold.reshape(b, c, 1)
        
        # 自适应缩放
        threshold = threshold * x.abs().mean(dim=2, keepdim=True)
        return threshold

class ResidualShrinkageCS(nn.Module):
    """通道共享残差收缩模块"""
    def __init__(self, in_channels, out_channels, groups=16, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 阈值生成器
        self.threshold_gen = SharedThreshold(out_channels, groups=groups)
        
        # 下采样和通道调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 通道共享收缩
        threshold = self.threshold_gen(out)
        out = torch.sign(out) * F.relu(torch.abs(out) - threshold)
        
        # 残差连接
        out += residual
        return F.relu(out)

class DRSN_CS(nn.Module):
    """完整的DRSN-CS网络结构"""
    def __init__(self, num_classes=10, layers=[3,4,6,3], groups=16):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差阶段
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2, groups=groups)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, groups):
        layers = []
        layers.append(ResidualShrinkageCS(in_channels, out_channels, groups, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualShrinkageCS(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        return x

# 测试代码
if __name__ == "__main__":
    model = DRSN_CS(num_classes=10)
    dummy_input = torch.randn(2, 1, 2048)  # (batch, channel, length)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # torch.Size([2, 10])