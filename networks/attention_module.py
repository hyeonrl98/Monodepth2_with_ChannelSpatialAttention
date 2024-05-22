from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)

class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = StandardConv(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        pooled = self.compress(x)
        spatial_att = self.spatial(pooled)
        scale = torch.sigmoid(spatial_att)
        return x * scale

class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(ChannelSpatialAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        if not self.no_spatial:
            out = self.spatial_attention(out)
        return out