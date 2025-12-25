import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse
# from cbam1D import *
# from utils import weights_init_kaiming
import os
from thop import profile
from thop import clever_format
from torch.autograd import Variable
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_1d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool1d( x, 2, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_1d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



# double_conv model
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.s_conv(x)
        return x

class single_conv_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv_res, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.s_conv(x)
        out = torch.add(x, residual)
        return out

class conv11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv11, self).__init__()
        self.s_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.s_conv(x)
        return x


class conv33(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv33, self).__init__()
        self.s_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.s_conv(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class channel_attn_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(channel_attn_layer, self).__init__()

        self.avgPoolW = nn.AdaptiveAvgPool1d(1)
        self.maxPoolW = nn.AdaptiveMaxPool1d(1)

        self.conv_1x1 = nn.Conv1d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm1d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        self.F_h = nn.Sequential(  # 激发操作
            nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm1d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(  # 激发操作
            nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm1d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, L = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x_conv = self.conv_1x1(x_cat)
        # 如果长度维度为 1，在训练模式下 BatchNorm1d 会报错，所以需要特殊处理
        if L == 1 and self.training:
            x = self.Relu(x_conv)  # 跳过 BatchNorm
        else:
            x = self.Relu(self.bn(x_conv))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


# class Sep(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, bias=True):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=bias)
#         self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.conv2(x)
#         return x


class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv1d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool1d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv1d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm1d(n_feat)
        )
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = channel_attn_layer(n_feat, reduction)  ## Channel Attention

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv1d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv1d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 拼接
        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), size=x.size()[2:], mode='linear', align_corners=False)))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


# class _DCR_block(nn.Module):
#     def __init__(self, channel_in):
#         super(_DCR_block, self).__init__()
#         self.conv_1 = nn.Conv1d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
#                                 padding=1)
#         self.relu1 = nn.LeakyReLU()
#         self.conv_2 = nn.Conv1d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
#                                 stride=1, padding=1)
#         self.relu2 = nn.LeakyReLU()
#         self.conv_3 = nn.Conv1d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.LeakyReLU()
#
#     def forward(self, x):
#         residual = x
#         out = self.relu1(self.conv_1(x))
#         conc = torch.cat([x, out], 1)
#         out = self.relu2(self.conv_2(conc))
#         conc = torch.cat([conc, out], 1)
#         out = self.relu3(self.conv_3(conc))
#         out = torch.add(out, residual)
#         return out


class New_block(nn.Module):
    def __init__(self, channel_in, reduction):
        super(New_block, self).__init__()

        # RCSSC
        self.unit_1 = RCSSC(int(channel_in / 2.), reduction)
        self.unit_2 = RCSSC(int(channel_in / 2.), reduction)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                      padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=1, padding=0,
                      stride=1),  # 做压缩
            nn.Conv1d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        residual = x
        c1 = self.unit_1(self.conv1(x))
        x = torch.cat([residual, c1], 1)
        c2 = self.unit_2(self.conv2(x))
        x = torch.cat([c2, x], 1)
        x = self.conv3(x)
        x = torch.add(x, residual)
        return x


#

class ASCNet(nn.Module):

    def __init__(self, in_ch, out_ch, feats):
        super(ASCNet, self).__init__()
        self.features = []

        self.head = single_conv(in_ch, feats)
        self.dconv_encode0 = double_conv(feats, feats)  # → har

        self.identety1 = nn.Conv1d(in_channels=feats, out_channels=2 * feats, kernel_size=3, stride=2, padding=1)
        # 使用1D DWT：直接对1D数据使用DWT
        self.DWT = DWT1DForward(J=1, wave='haar')
        self.dconv_encode1 = single_conv(4 * feats, 2 * feats)

        # CNCM
        self.enhance1 = New_block(2 * feats, reduction=16)

        self.identety2 = nn.Conv1d(in_channels=2 * feats, out_channels=4 * feats, kernel_size=3, stride=2, padding=1)

        self.dconv_encode2 = single_conv(8 * feats, 4 * feats)

        self.dconv_encode3 = single_conv(16 * feats, 4 * feats)

        self.enhance2 = New_block(4 * feats, reduction=16)
        self.identety3 = nn.Conv1d(in_channels=4 * feats, out_channels=4 * feats, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool1d(2)
        self.enhance3 = New_block(4 * feats, reduction=16)

        self.mid1 = single_conv(8 * feats, 4 * feats)
        self.mid2 = single_conv(4 * feats, 4 * feats + 4 * feats)

        # decoder*****************************************************
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose1d(8 * feats, 4 * feats, kernel_size=2, stride=2),
            # nn.LeakyReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose1d(4 * feats, 2 * feats, kernel_size=2, stride=2),
            # nn.LeakyReLU(inplace=True)
        )

        self.upsample0 = nn.Sequential(
            nn.ConvTranspose1d(2 * feats, feats, kernel_size=2, stride=2),
            # nn.LeakyReLU(inplace=True)
        )
        self.IDWT = DWT1DInverse(wave='haar')

        # fair *******************************************************
        self.fair2 = nn.Conv1d(4 * feats, 4 * feats, kernel_size=3, padding=1)
        self.fair1 = nn.Conv1d(2 * feats, 2 * feats, kernel_size=3, padding=1)
        self.fair0 = nn.Conv1d(feats, feats, kernel_size=3, padding=1)

        # decoder*****************************************************
        self.dconv_decode2 = nn.Sequential(conv11(4 * feats + 4 * feats, 4 * feats),
                                           New_block(4 * feats, reduction=16))

        self.dconv_decode1 = nn.Sequential(conv11(2 * feats + 2 * feats, 2 * feats),
                                           New_block(2 * feats, reduction=16))

        self.dconv_decode0 = double_conv(feats + feats, feats)
        self.tail = nn.Sequential(nn.Conv1d(feats, out_ch, 1), nn.Tanh())

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        a = DMT1_yh[0]  # 高频分量 (B, C, L//2)
        list_tensor.append(DMT1_yl)  # 低频分量 (B, C, L//2)
        # 对于1D DWT，只有一个高频分量，为了匹配4通道结构，复制3次
        for i in range(3):
            list_tensor.append(a)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        yl = out[:, 0:C, :]  # 低频分量 (B, C, L//2)
        # 对于1D DWT，取第一个高频分量（其他两个是复制的，可以忽略）
        y1 = out[:, C:2 * C, :]
        yh.append(y1)  # 1D DWT只需要一个高频分量
        return yl, yh

    def forward(self, x):
        inputs = x

        x0 = self.dconv_encode0(self.head(x))

        # 直接使用1D DWT，无需reshape
        DMT1_yl, DMT1_yh = self.DWT(x0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)  # (B, 4*feats, L//2)
        x = self.dconv_encode1(DMT1)

        res1 = self.identety1(x0)
        out = torch.add(x, res1)

        x1 = self.enhance1(out)

        # 直接使用1D DWT，无需reshape
        DMT2_yl, DMT2_yh = self.DWT(x1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)  # (B, 8*feats, L//4)
        x = self.dconv_encode2(DMT2)

        res1 = self.identety2(x1)
        out2 = torch.add(x, res1)

        x2 = self.enhance2(out2)

        # 直接使用1D DWT，无需reshape
        DMT3_yl, DMT3_yh = self.DWT(x2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)  # (B, 16*feats, L//8)
        x = self.dconv_encode3(DMT3)

        res1 = self.identety3(x2)
        out3 = torch.add(x, res1)
        # MI = self.mid1(out3)
        x3 = self.mid2(self.enhance3(out3))

        x = self.upsample2(x3)
        x = self.fair2(x)
        x = self.dconv_decode2(torch.cat([x, x2], dim=1))
        x = self.upsample1(x)
        x = self.fair1(x)
        x = self.dconv_decode1(torch.cat([x, x1], dim=1))
        x = self.upsample0(x)
        x = self.fair0(x)

        x = self.dconv_decode0(torch.cat([x, x0], dim=1))
        x = self.tail(x)
        
        # 残差连接：如果输入和输出通道数相同，直接相加；否则只对第一个通道相加
        if x.shape[1] == inputs.shape[1]:
            out = x + inputs
        else:
            # 如果通道数不同，只对第一个通道进行残差连接
            out = x + inputs[:, 0:1, :]

        return out


if __name__ == '__main__':
    # 测试输入3通道、输出1通道的情况
    net = ASCNet(3, 1, feats=32)
    net.eval()  # 设置为评估模式以避免 BatchNorm 问题
    input = torch.zeros((1, 3, 256), dtype=torch.float32)  # (batch_size, channels, length)
    output = net(input)

    flops, params = profile(net, (input,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print('Input shape:', input.shape)
    print('Output shape:', output.shape)

