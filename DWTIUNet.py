import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse


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


# ==================== IDWT 上采样模块 ====================

class IDWT_Up(nn.Module):
    """使用 IDWT 进行上采样的模块，与 DWT_Down 对称"""

    def __init__(self, in_channels, skip_channels, out_channels, wave="haar", J=1):
        super(IDWT_Up, self).__init__()
        self.IDWT = DWT1DInverse(wave=wave)
        # 将 in_channels 转换为 out_channels * 2，以便分离为 yl 和 yh
        self.conv_before_idwt = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)
        # IDWT 后与 skip connection 拼接，再通过卷积
        self.conv_after_idwt = double_conv(out_channels + skip_channels, out_channels)

    def _itransformer(self, x):
        """将通道数分离为低频 yl 和高频 yh"""
        # x shape: (B, out_channels * 2, L//2)
        C = x.shape[1] // 2
        yl = x[:, :C, :]  # 低频分量 (B, out_channels, L//2)
        yh = [x[:, C:, :]]  # 高频分量 (B, out_channels, L//2)
        return yl, yh

    def forward(self, x1, x_skip):
        """
        Args:
            x1: 来自解码器的特征 (B, in_channels, L//2)
            x_skip: skip connection 特征 (B, skip_channels, L)
        Returns:
            上采样后的特征 (B, out_channels, L)
        """
        # 将 x1 转换为 out_channels * 2 通道
        x1 = self.conv_before_idwt(x1)  # (B, out_channels * 2, L//2)
        
        # 分离为低频和高频分量
        yl, yh = self._itransformer(x1)
        
        # 使用 IDWT 进行上采样，长度翻倍
        x1 = self.IDWT((yl, yh))  # (B, out_channels, L)

        # 对齐长度（处理可能的尺寸不匹配）
        diff = x_skip.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x1 = x1[:, :, :x_skip.size(2)]

        # 拼接 skip 和上采样后的特征
        x = torch.cat([x_skip, x1], dim=1)  # (B, skip_channels + out_channels, L)
        return self.conv_after_idwt(x)  # (B, out_channels, L)


# ==================== 使用 DWT/IDWT 对称结构的 UNet 主网络 ====================

class DWTIUNet(nn.Module):
    """
    使用 DWT/IDWT 对称结构的 1D UNet
    - 编码：DWT_Down 进行下采样
    - 解码：IDWT_Up 进行上采样（与 DWT 下采样对称）
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 base_c: int = 64,
                 wave: str = "haar",
                 J: int = 1):
        super(DWTIUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 编码：DWT 下采样
        self.in_conv = double_conv(in_channels, base_c)
        self.down1 = DWT_Down(base_c, base_c * 2, wave=wave, J=J)
        self.down2 = DWT_Down(base_c * 2, base_c * 4, wave=wave, J=J)
        self.down3 = DWT_Down(base_c * 4, base_c * 8, wave=wave, J=J)
        self.down4 = DWT_Down(base_c * 8, base_c * 16, wave=wave, J=J)

        # 解码：IDWT 上采样（与 DWT 下采样对称）
        self.up1 = IDWT_Up(base_c * 16, base_c * 8, base_c * 8, wave=wave, J=J)
        self.up2 = IDWT_Up(base_c * 8, base_c * 4, base_c * 4, wave=wave, J=J)
        self.up3 = IDWT_Up(base_c * 4, base_c * 2, base_c * 2, wave=wave, J=J)
        self.up4 = IDWT_Up(base_c * 2, base_c, base_c, wave=wave, J=J)

        # 输出层（保持与 DWT_S_UNet 一致，便于对比）
        self.out_conv = nn.Sequential(
            nn.Conv1d(base_c, num_classes, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码（DWT 下采样）
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码（IDWT 上采样 + skip）
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
    net = DWTIUNet(in_channels=1, num_classes=1, base_c=64, wave="haar", J=1)
    dummy = torch.zeros((2, 1, 256))
    with torch.no_grad():
        out = net(dummy)
    print("DWTIUNet Input:", dummy.shape, "Output:", out.shape)
    print("DWTIUNet created successfully!")






