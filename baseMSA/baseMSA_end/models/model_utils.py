import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = True
class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()  # 激活函数
        self.conv = nn.Conv2d(inplanes, planes * upscale_factor ** 2, kernel_size=3, padding=1)  # 卷积层
        self.bn = nn.BatchNorm2d(planes * upscale_factor ** 2)  # 批量归一化
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 像素洗牌操作用于上采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)  # 通过PixelShuffle进行上采样
        x = self.conv2(x)
        return x

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation==1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,  padding=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
    
class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

# 一个快速聚拢感受野的方法，改编自STDC
class MFACB(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2,2,2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes *3, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # out1 = self.conv_list[0](x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1    
    
class SPASPP(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[6,12,18,24]):
        super(SPASPP, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[3]))
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes*5, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process3 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
#             nn.Dropout(0.2)
        )
    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        out2 = self.pooling(x)
        # out1 = self.conv_list[0](x)
        for idx in range(4):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out,out2), dim=1)
        return self.process3(self.process2(out) + out1)
    
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False,dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False, )
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmentheadCARAFE(nn.Module):

    def __init__(self, interplanes, outplanes, scale_factor=8, cp_rate=4):
        super(segmentheadCARAFE, self).__init__()
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.carafe = CARAFE(inC=interplanes, outC=outplanes, cp_rate=cp_rate, up_factor=scale_factor)
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.carafe(self.relu(self.bn2(x)))

        return out

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
    
class segmenthead_c(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_c, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
class segmenthead_drop(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_drop, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)

        self.drop = nn.Dropout(0.5)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.drop(self.relu(self.bn2(x)))
        out = self.conv2(x)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class segmenthead_l(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        self.bn3 = BatchNorm2d(outplanes, momentum=bn_mom)

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = self.bn3(out)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out




class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

# MSAF
class Muti_AFF(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(Muti_AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# MSAF_small
class MSAF_small(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSAF_small, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # self.context3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((16, 16)),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels)
        # )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        # c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        # c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
        
# MSA
class MSA(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x 
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei
        return xo

# MSA_small
class MSA_small(nn.Module):
    """
    原始多尺度注意力模块（不带跳跃连接）
    """
    def __init__(self, channels, r=4):
        super(MSA_small, self).__init__()
        self.channels = channels
        self.r = r
        
        # 通道注意力
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力（简化版）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        batch_size, channels, height, width = x.size()
        
        # 通道注意力
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        channel_att = self.fc(avg_pool).view(batch_size, channels, 1, 1)
        
        # 空间注意力
        spatial_att = self.spatial_att(x)
        
        # 应用注意力
        out = x * channel_att * spatial_att
        
        return out


class MSA_small_with_skip(nn.Module):
    """
    带跳跃连接的多尺度注意力模块
    """
    def __init__(self, channels, r=4, skip_ratio=0.3, skip_type='standard'):
        super(MSA_small_with_skip, self).__init__()
        self.channels = channels
        self.r = r
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        
        print(f"🚀 初始化MSA跳跃连接模块: skip_ratio={skip_ratio}, skip_type={skip_type}")
        
        # 基础MSA组件
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 🆕 跳跃连接相关组件
        if skip_type == 'learned':
            # 学习性跳跃连接：使用神经网络学习跳跃权重
            self.skip_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )
        elif skip_type == 'adaptive':
            # 自适应跳跃连接：基于特征统计自适应调整
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
            self.momentum = 0.1
        # 'standard' 类型不需要额外参数
        
    def _compute_skip_weight(self, x, attention_output):
        """计算跳跃连接权重"""
        if self.skip_type == 'standard':
            # 标准跳跃连接：固定比例
            return self.skip_ratio
            
        elif self.skip_type == 'learned':
            # 学习性跳跃连接：基于输入特征学习权重
            skip_weight = self.skip_gate(x)
            return skip_weight * self.skip_ratio
            
        elif self.skip_type == 'adaptive':
            # 自适应跳跃连接：基于特征变化自适应调整
            if self.training:
                # 训练时更新统计信息
                current_mean = torch.mean(attention_output)
                current_var = torch.var(attention_output)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            
            # 基于方差自适应调整跳跃比例
            adaptive_ratio = torch.clamp(self.running_var, 0.1, 1.0) * self.skip_ratio
            return adaptive_ratio
        else:
            return self.skip_ratio
    
    def forward(self, x):
        """
        前向传播，支持跳跃连接
        """
        # x: [B, C, H, W]
        batch_size, channels, height, width = x.size()
        
        # 🔍 保存原始输入用于跳跃连接
        residual = x
        
        # 通道注意力
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        channel_att = self.fc(avg_pool).view(batch_size, channels, 1, 1)
        
        # 空间注意力
        spatial_att = self.spatial_att(x)
        
        # 应用注意力得到增强特征
        attention_output = x * channel_att * spatial_att
        
        # 🆕 计算跳跃连接权重
        skip_weight = self._compute_skip_weight(x, attention_output)
        
        # 🆕 应用跳跃连接
        if self.skip_type == 'learned' and len(skip_weight.shape) > 1:
            # 学习性权重可能是多维的
            output = (1 - skip_weight) * attention_output + skip_weight * residual
        else:
            # 标准和自适应权重是标量
            output = (1 - skip_weight) * attention_output + skip_weight * residual
        
        return output


# 🆕 为兼容性提供工厂函数
def create_msa_with_skip(channels, r=4, skip_ratio=0.3, skip_type='standard'):
    """
    创建带跳跃连接的MSA模块的工厂函数
    """
    return MSA_small_with_skip(channels, r, skip_ratio, skip_type)


class MultiScale_MSA_with_skip(nn.Module):
    """
    多尺度MSA模块，带跳跃连接
    """
    def __init__(self, channels, r=4, scales=[1, 2, 4], skip_ratio=0.3, skip_type='standard'):
        super(MultiScale_MSA_with_skip, self).__init__()
        self.channels = channels
        self.scales = scales
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        
        # 为每个尺度创建MSA模块
        self.scale_modules = nn.ModuleList([
            MSA_small_with_skip(channels, r, skip_ratio, skip_type) 
            for _ in scales
        ])
        
        # 融合不同尺度的特征
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        多尺度注意力处理
        """
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # 原始尺度
                scale_out = self.scale_modules[i](x)
            else:
                # 下采样到不同尺度
                h, w = x.shape[2], x.shape[3]
                scaled_x = F.adaptive_avg_pool2d(x, (h//scale, w//scale))
                scaled_out = self.scale_modules[i](scaled_x)
                # 上采样回原始尺寸
                scale_out = F.interpolate(scaled_out, size=(h, w), mode='bilinear', align_corners=False)
            
            scale_outputs.append(scale_out)
        
        # 融合多尺度特征
        fused = torch.cat(scale_outputs, dim=1)
        output = self.fusion(fused)
        
        # 全局残差连接
        output = output + x * self.skip_ratio
        
        return output


# 🆕 高级跳跃连接MSA
class Advanced_MSA_with_skip(nn.Module):
    """
    高级跳跃连接MSA，支持多种跳跃连接策略
    """
    def __init__(self, channels, r=4, skip_ratio=0.3, skip_type='standard', 
                 use_multihead=True, num_heads=8):
        super(Advanced_MSA_with_skip, self).__init__()
        self.channels = channels
        self.r = r
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        self.use_multihead = use_multihead
        self.num_heads = num_heads
        
        if use_multihead:
            # 多头注意力机制
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=channels, 
                num_heads=num_heads, 
                batch_first=True
            )
        
        # 基础组件
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, 1, 1),
            nn.Sigmoid()
        )
        
        # 跳跃连接门控
        if skip_type == 'gated':
            self.skip_gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),  # 输入和注意力输出拼接
                nn.Sigmoid()
            )
    
    def forward(self, x):
        B, C, H, W = x.size()
        residual = x
        
        # 通道注意力
        ca = self.channel_att(x)
        x_ca = x * ca
        
        # 空间注意力
        sa = self.spatial_att(x_ca)
        attention_out = x_ca * sa
        
        # 多头自注意力（可选）
        if self.use_multihead:
            # 重塑为序列格式
            x_seq = attention_out.flatten(2).transpose(1, 2)  # [B, HW, C]
            attn_out, _ = self.multihead_attn(x_seq, x_seq, x_seq)
            attention_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # 应用跳跃连接
        if self.skip_type == 'gated':
            # 门控跳跃连接
            combined = torch.cat([residual, attention_out], dim=1)
            gate = self.skip_gate(combined)
            output = gate * attention_out + (1 - gate) * residual
        else:
            # 标准跳跃连接
            output = (1 - self.skip_ratio) * attention_out + self.skip_ratio * residual
        
        return output


# 🆕 为兼容性提供工厂函数
def create_msa_with_skip(channels, r=4, skip_ratio=0.3, skip_type='standard', 
                        variant='small'):
    """
    创建带跳跃连接的MSA模块的工厂函数
    
    Args:
        channels: 输入通道数
        r: 降维比例
        skip_ratio: 跳跃连接比例
        skip_type: 跳跃连接类型 ('standard', 'learned', 'adaptive', 'gated')
        variant: MSA变体 ('small', 'multiscale', 'advanced')
    """
    if variant == 'small':
        return MSA_small_with_skip(channels, r, skip_ratio, skip_type)
    elif variant == 'multiscale':
        return MultiScale_MSA_with_skip(channels, r, skip_ratio=skip_ratio, skip_type=skip_type)
    elif variant == 'advanced':
        return Advanced_MSA_with_skip(channels, r, skip_ratio, skip_type)
    else:
        raise ValueError(f"Unknown MSA variant: {variant}")


# 🧪 测试函数
def test_msa_skip_connections():
    """测试跳跃连接MSA模块"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    batch_size = 2
    channels = 768
    height, width = 8, 8
    
    x = torch.randn(batch_size, channels, height, width).to(device)
    
    print("🧪 测试MSA跳跃连接模块...")
    
    # 测试不同类型的跳跃连接
    skip_types = ['standard', 'learned', 'adaptive']
    
    for skip_type in skip_types:
        print(f"\n📊 测试 {skip_type} 跳跃连接:")
        
        msa = MSA_small_with_skip(
            channels=channels, 
            r=4, 
            skip_ratio=0.3, 
            skip_type=skip_type
        ).to(device)
        
        msa.eval()
        with torch.no_grad():
            output = msa(x)
            print(f"  输入形状: {x.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
            # 检查跳跃连接效果
            diff = torch.norm(output - x) / torch.norm(x)
            print(f"  与输入的相对差异: {diff:.4f}")
    
    print("\n✅ 所有测试完成!")


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


    

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18, 24], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


    

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
    
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2, cp_rate=4, ifBN=False):
        super(CARAFE, self).__init__()
        self.ifBN = ifBN
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // cp_rate, 1)
        self.encoder = nn.Conv2d(inC // cp_rate, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)
        self.bn = nn.BatchNorm2d(outC)
    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        if self.ifBN:
            out_tensor = self.bn(out_tensor)
        return out_tensor

if __name__ == '__main__':

    
    data = torch.rand(4, 160, 10, 10)
    carafe = CARAFE(160, 19, up_factor=8)
    print(carafe(data).size())
    
    test_msa_skip_connections()
