# File: model/DBDRSNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 改进的Bottleneck层（1D版本）
class Bottleneck1D(nn.Module):
    def __init__(self, in_channels, growth_rate, use_bottleneck=True):
        super().__init__()
        inter_channels = 4 * growth_rate if use_bottleneck else in_channels
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(inter_channels)
        self.conv2 = nn.Conv1d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

# 通道注意力收缩模块（1D版本）
class ChannelAttentionShrinkage1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        b, c, _ = x.size()
        gap = self.gap(x).view(b, c)
        threshold = self.fc(gap).view(b, c, 1)
        threshold = threshold * x.abs().mean(dim=2, keepdim=True)
        x = torch.sign(x) * torch.relu(torch.abs(x) - threshold)
        return self.bn(x + residual)  # 添加残差连接

# DB-RSBU模块（核心改进模块）
class DB_RSBU1D(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=2):
        super().__init__()
        self.dense_layers = nn.ModuleList()
        current_channels = in_channels
        
        # 第一个Dense Block
        for _ in range(num_layers):
            layer = Bottleneck1D(current_channels, growth_rate)
            self.dense_layers.append(layer)
            current_channels += growth_rate
        
        # 降维层
        self.transition = nn.Sequential(
            nn.BatchNorm1d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(current_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # 注意力收缩模块
        self.shrinkage = ChannelAttentionShrinkage1D(in_channels)
        
        # 恒等映射调整
        self.identity_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1) if in_channels != current_channels else None
    
    def forward(self, x):
        identity = x
        for layer in self.dense_layers:
            x = layer(x)
        x = self.transition(x)
        if self.identity_conv:
            identity = self.identity_conv(identity)
        x = self.shrinkage(x)
        return F.relu(x + identity)

# 完整的DB-DRSN网络（1D版本）
class DB_DRSN1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 初始多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True))
        ])
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 堆叠DB-RSBU模块
        self.stage1 = self._make_stage(64, 64, num_blocks=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=4, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        
        # 分类层
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # 下采样调整
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride))
        
        # 添加DB-RSBU模块
        for _ in range(num_blocks):
            layers.append(DB_RSBU1D(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 多尺度特征提取
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i % 2 == 0:  # 交替池化
                x = self.pool(x)
        
        # 特征处理阶段
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # 分类输出
        x = self.gap(x).squeeze(-1)
        return self.fc(x)