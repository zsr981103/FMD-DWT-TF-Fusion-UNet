import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward


# ==================== 基础组件 ====================

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.d_conv(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        # 最大池化与平均池化沿通道聚合
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ==================== ASCNet 的 SpatialGate ====================

class SpatialGate(nn.Module):
    """空间注意力模块（来自ASCNet）"""
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


# ==================== DWT 编码/解码模块 ====================

class DWT_Down(nn.Module):
    """使用 DWT 进行下采样的模块"""

    def __init__(self, in_channels, out_channels, wave="haar", J=1):
        super(DWT_Down, self).__init__()
        self.DWT = DWT1DForward(J=J, wave=wave)
        self.conv = double_conv(in_channels * 2, out_channels)

    def _transformer(self, yl, yh):
        # 拼接低频与第一层高频
        return torch.cat([yl, yh[0]], dim=1)

    def forward(self, x):
        yl, yh = self.DWT(x)
        dwt_out = self._transformer(yl, yh)
        return self.conv(dwt_out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool1d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


# ==================== 四通道融合上采样模块 ====================

class Hybrid_Up_4Channel(nn.Module):
    """融合4通道特征：SpatialGate(dwt_x) + dwt_x + SpatialGate(unet_x) + unet_x"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Hybrid_Up_4Channel, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            # 4特征加权融合后通道数仍为skip_channels
            self.conv = double_conv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = double_conv(in_channels // 2 + skip_channels, out_channels)

        # DWT和UNet分支的SpatialGate
        self.spatial_gate_dwt = SpatialGate()
        self.spatial_gate_unet = SpatialGate()

        # 可学习的融合权重（四分支）
        self.weight_dwt_spatial = nn.Parameter(torch.tensor(0.0 / 4))  # SpatialGate(dwt_x)权重
        self.weight_dwt_original = nn.Parameter(torch.tensor(1.5 / 4))  # dwt_x原始权重
        self.weight_unet_spatial = nn.Parameter(torch.tensor(0.0 / 4))  # SpatialGate(unet_x)权重
        self.weight_unet_original = nn.Parameter(torch.tensor(2.5 / 4))  # unet_x原始权重

    def forward(self, x1, skip_dwt, skip_unet):
        x1 = self.up(x1)

        # 对齐长度
        target_size = max(skip_dwt.size(2), skip_unet.size(2))
        diff = target_size - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x1 = x1[:, :, :target_size]

        # 对齐跳跃连接的长度与通道数
        min_size = min(skip_dwt.size(2), skip_unet.size(2))
        skip_dwt = skip_dwt[:, :, :min_size]
        skip_unet = skip_unet[:, :, :min_size]

        min_channels = min(skip_dwt.size(1), skip_unet.size(1))
        skip_dwt = skip_dwt[:, :min_channels, :]
        skip_unet = skip_unet[:, :min_channels, :]

        # 对DWT和UNet分支分别应用SpatialGate
        skip_dwt_spatial = self.spatial_gate_dwt(skip_dwt)  # DWT的空间注意力特征
        skip_unet_spatial = self.spatial_gate_unet(skip_unet)  # UNet的空间注意力特征

        # 权重归一化，防止梯度爆炸/负值
        weight_sum = (
            torch.abs(self.weight_dwt_spatial)
            + torch.abs(self.weight_dwt_original)
            + torch.abs(self.weight_unet_spatial)
            + torch.abs(self.weight_unet_original)
        )
        w_dwt_spatial = torch.abs(self.weight_dwt_spatial) / weight_sum
        w_dwt_original = torch.abs(self.weight_dwt_original) / weight_sum
        w_unet_spatial = torch.abs(self.weight_unet_spatial) / weight_sum
        w_unet_original = torch.abs(self.weight_unet_original) / weight_sum

        # 四通道加权融合
        skip_fused = (w_dwt_spatial * skip_dwt_spatial 
                     + w_dwt_original * skip_dwt 
                     + w_unet_spatial * skip_unet_spatial 
                     + w_unet_original * skip_unet)

        # 最终对齐 skip_fused 和 x1 的长度
        final_size = min(skip_fused.size(2), x1.size(2))
        skip_fused = skip_fused[:, :, :final_size]
        x1 = x1[:, :, :final_size]

        # 拼接融合后的跳跃连接和上采样特征
        x = torch.cat([skip_fused, x1], dim=1)
        return self.conv(x)


# ==================== DWT-S-UNet 主网络 =====================

class DWT_S_UNet_ReLU(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 wave: str = "haar",
                 J: int = 1):
        super(DWT_S_UNet_ReLU, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # DWT 编码分支
        self.dwt_in_conv = double_conv(in_channels, base_c)
        self.dwt_down1 = DWT_Down(base_c, base_c * 2, wave=wave, J=J)
        self.dwt_down2 = DWT_Down(base_c * 2, base_c * 4, wave=wave, J=J)
        self.dwt_down3 = DWT_Down(base_c * 4, base_c * 8, wave=wave, J=J)
        factor = 2 if bilinear else 1
        self.dwt_down4 = DWT_Down(base_c * 8, base_c * 16 // factor, wave=wave, J=J)

        # UNet 编码分支
        self.unet_in_conv = DoubleConv(in_channels, base_c)
        self.unet_down1 = Down(base_c, base_c * 2)
        self.unet_down2 = Down(base_c * 2, base_c * 4)
        self.unet_down3 = Down(base_c * 4, base_c * 8)
        self.unet_down4 = Down(base_c * 8, base_c * 16 // factor)

        # 融合瓶颈特征
        self.bottleneck_fusion = nn.Conv1d(base_c * 16 // factor * 2, base_c * 16 // factor, kernel_size=1)

        # 解码器（使用四通道融合：SpatialGate(dwt_x) + dwt_x + SpatialGate(unet_x) + unet_x）
        self.up1 = Hybrid_Up_4Channel(base_c * 16 // factor, base_c * 8, base_c * 8 // factor, bilinear)
        self.up2 = Hybrid_Up_4Channel(base_c * 8 // factor, base_c * 4, base_c * 4 // factor, bilinear)
        self.up3 = Hybrid_Up_4Channel(base_c * 4 // factor, base_c * 2, base_c * 2 // factor, bilinear)
        self.up4 = Hybrid_Up_4Channel(base_c * 2 // factor, base_c, base_c, bilinear)

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv1d(base_c, num_classes, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # DWT 编码
        dwt_x1 = self.dwt_in_conv(x)
        dwt_x2 = self.dwt_down1(dwt_x1)
        dwt_x3 = self.dwt_down2(dwt_x2)
        dwt_x4 = self.dwt_down3(dwt_x3)
        dwt_x5 = self.dwt_down4(dwt_x4)

        # UNet 编码
        unet_x1 = self.unet_in_conv(x)
        unet_x2 = self.unet_down1(unet_x1)
        unet_x3 = self.unet_down2(unet_x2)
        unet_x4 = self.unet_down3(unet_x3)
        unet_x5 = self.unet_down4(unet_x4)

        # 融合瓶颈（对齐长度）
        min_size = min(dwt_x5.size(2), unet_x5.size(2))
        dwt_x5 = dwt_x5[:, :, :min_size]
        unet_x5 = unet_x5[:, :, :min_size]
        bottleneck = torch.cat([dwt_x5, unet_x5], dim=1)
        x = self.bottleneck_fusion(bottleneck)

        # 解码（四通道融合：SpatialGate(dwt_x) + dwt_x + SpatialGate(unet_x) + unet_x）
        x = self.up1(x, dwt_x4, unet_x4)
        x = self.up2(x, dwt_x3, unet_x3)
        x = self.up3(x, dwt_x2, unet_x2)
        x = self.up4(x, dwt_x1, unet_x1)

        logits = self.out_conv(x)

        # 残差输出，若通道数一致则加残差
        out = logits + x if logits.shape[1] == x.shape[1] else logits
        return out


if __name__ == "__main__":
    # 简单自测：输入 (B, C, L)
    net = DWT_S_UNet_ReLU(in_channels=1, num_classes=1, bilinear=True, base_c=64, wave="haar", J=1)
    dummy = torch.zeros((2, 1, 256))
    with torch.no_grad():
        out = net(dummy)
    print("Input:", dummy.shape, "Output:", out.shape)
    print("Network created successfully!")
