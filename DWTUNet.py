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


# ==================== DWT 下采样模块 ====================

class DWT_Down(nn.Module):
    """使用 DWT 进行下采样的模块"""

    def __init__(self, in_channels, out_channels, wave="haar", J=1):
        super(DWT_Down, self).__init__()
        self.DWT = DWT1DForward(J=J, wave=wave)
        # DWT 输出：低频 yl 和第一层高频 yh[0]，通道数翻倍
        self.conv = double_conv(in_channels * 2, out_channels)

    def _transformer(self, yl, yh):
        # 拼接低频与第一层高频
        return torch.cat([yl, yh[0]], dim=1)

    def forward(self, x):
        yl, yh = self.DWT(x)
        dwt_out = self._transformer(yl, yh)
        return self.conv(dwt_out)


# ==================== 解码端上采样模块 ====================

class DWTUp(nn.Module):
    """标准 1D UNet 上采样模块（单分支），用于 DWTUNet 消融网络"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(DWTUp, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = double_conv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = double_conv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x_skip):
        # 上采样
        x1 = self.up(x1)

        # 对齐长度
        diff = x_skip.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x1 = x1[:, :, :x_skip.size(2)]

        # 拼接 skip 和 up 特征
        x = torch.cat([x_skip, x1], dim=1)
        return self.conv(x)


# ==================== 仅用 DWT 下采样的 UNet 主网络 ====================

class DWTUNet(nn.Module):
    """
    仅使用 DWT 下采样的 1D UNet（消融实验用）
    - 编码：DWT_Down 进行下采样
    - 解码：标准上采样 + concat skip（无 UNet 池化分支、无 SpatialGate 融合）
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 wave: str = "haar",
                 J: int = 1):
        super(DWTUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码：仅 DWT 分支
        self.in_conv = double_conv(in_channels, base_c)
        self.down1 = DWT_Down(base_c, base_c * 2, wave=wave, J=J)
        self.down2 = DWT_Down(base_c * 2, base_c * 4, wave=wave, J=J)
        self.down3 = DWT_Down(base_c * 4, base_c * 8, wave=wave, J=J)
        factor = 2 if bilinear else 1
        self.down4 = DWT_Down(base_c * 8, base_c * 16 // factor, wave=wave, J=J)

        # 解码：标准 UNet 上采样
        self.up1 = DWTUp(base_c * 16 // factor, base_c * 8, base_c * 8 // factor, bilinear)
        self.up2 = DWTUp(base_c * 8 // factor, base_c * 4, base_c * 4 // factor, bilinear)
        self.up3 = DWTUp(base_c * 4 // factor, base_c * 2, base_c * 2 // factor, bilinear)
        self.up4 = DWTUp(base_c * 2 // factor, base_c, base_c, bilinear)

        # 输出层（保持与 DWT_S_UNet 一致，便于对比）
        self.out_conv = nn.Sequential(
            nn.Conv1d(base_c, num_classes, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码（仅 DWT）
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码（上采样 + skip）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        # 残差输出：若通道一致则加残差
        out = logits + x if logits.shape[1] == x.shape[1] else logits
        return out


if __name__ == "__main__":
    # 简单自测：输入 (B, C, L)
    net = DWTUNet(in_channels=1, num_classes=1, bilinear=True, base_c=64, wave="haar", J=1)
    dummy = torch.zeros((2, 1, 256))
    with torch.no_grad():
        out = net(dummy)
    print("DWTUNet Input:", dummy.shape, "Output:", out.shape)
    print("DWTUNet created successfully!")


