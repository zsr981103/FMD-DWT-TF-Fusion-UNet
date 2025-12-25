import torch
import torch.nn as nn




class SpectralConv1D(nn.Module):
    """1D version of SpectralConv matching SMCNN_Model.py structure"""

    def __init__(self, in_channel, out_channel, K):
        super(SpectralConv1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.K = K
        # 1D version of multi-scale convolutions
        self.conv1d_3 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, spectral_vol):
        conv1 = self.conv1d_3(spectral_vol)
        conv2 = self.conv1d_5(spectral_vol)
        conv3 = self.conv1d_7(spectral_vol)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output


class SpatialConv1D(nn.Module):
    """1D version of SpatialConv matching SMCNN_Model.py structure"""

    def __init__(self, in_channel, out_channel):
        super(SpatialConv1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # 1D version of multi-scale convolutions
        self.conv1d_3 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, spatial_band):
        conv1 = self.conv1d_3(spatial_band)
        conv2 = self.conv1d_5(spatial_band)
        conv3 = self.conv1d_7(spatial_band)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output

def param_free_norm_1d(x, epsilon=1e-5):
    """1D version of parameter-free normalization"""
    x_var, x_mean = torch.var_mean(x, dim=[2], keepdim=True)
    x_std = torch.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std


class ssmm1d(nn.Module):
    """1D spectral self-modulation module"""
    def __init__(self, in_channel, out_channel, k):
        super(ssmm1d, self).__init__()
        # 修复通道数匹配问题
        self.conv1d_3 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        # 调整输入通道数以匹配实际的 spectral_volume
        self.conv1d_5 = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x_init, adj_spectral):
        x = param_free_norm_1d(x_init)
        tmp = self.conv1d_5(adj_spectral)
        tmp = self.relu(tmp)
        noisemap_gamma = self.conv1d_3(tmp)
        noisemap_beta = self.conv1d_3(tmp)
        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x


class ssmrb1d(nn.Module):
    """1D spectral self-modulation residual block"""
    def __init__(self, in_channel, out_channel, k=24):
        super(ssmrb1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.ssmm = ssmm1d(in_channel, out_channel, k)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x_init, adj_spectral):
        x = self.ssmm(x_init, adj_spectral)
        x = self.lrelu(x)
        x = self.conv1d(x)
        x = self.ssmm(x, adj_spectral)
        x = self.lrelu(x)
        x = self.conv1d(x)
        return x + x_init


class ConvBlock1D(nn.Module):
    """1D version of ConvBlock matching SMCNN_Model.py structure"""

    def __init__(self, in_channel, out_channel):
        super(ConvBlock1D, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=out_channel, out_channels=
                                  out_channel, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=out_channel, out_channels=
                                  int(out_channel/4), kernel_size=3, padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=out_channel, out_channels=
                                  1, kernel_size=3, padding=1)
        self.ssmrb = ssmrb1d(out_channel, out_channel)
        self.relu = nn.ReLU()

    def forward(self, spectral_volume, volume):
        # For 1D case, spectral_volume is already 1D, no need to squeeze
        # 但是需要确保 spectral_volume 是单通道的
        if spectral_volume.shape[1] > 1:
            spectral_volume = spectral_volume[:, :1, :]  # 只取第一个通道
        
        conv1 = self.relu(self.conv1d_1(volume))
        conv2 = self.ssmrb(conv1, spectral_volume)
        conv2 = self.ssmrb(conv2, spectral_volume)
        conv3 = self.relu(self.conv1d_2(conv2))
        conv4 = self.ssmrb(conv3, spectral_volume)
        conv4 = self.ssmrb(conv4, spectral_volume)
        conv5 = self.relu(self.conv1d_2(conv4))
        conv6 = self.ssmrb(conv5, spectral_volume)
        conv6 = self.ssmrb(conv6, spectral_volume)
        conv7 = self.relu(self.conv1d_2(conv6))
        conv8 = self.ssmrb(conv7, spectral_volume)
        conv8 = self.ssmrb(conv8, spectral_volume)
        conv9 = self.relu(self.conv1d_2(conv8))

        f_conv3 = self.conv1d_3(conv3)
        f_conv5 = self.conv1d_3(conv5)
        f_conv7 = self.conv1d_3(conv7)
        f_conv9 = self.conv1d_3(conv9)
        final_volume = torch.cat([f_conv3, f_conv5, f_conv7, f_conv9], dim=1)
        final_volume = self.relu(final_volume)
        clean_band = self.conv1d_4(final_volume)
        return clean_band


class SMCNN1D(nn.Module):
    """1D variant mirroring the original SMCNN pattern.

    forward(x):
      - x: (N, C, 256), C can be 1 or 4
      - splits a reference band spatial_band = x[:, :1, :]
      - spectral_conv over full x, spatial_conv over spatial_band
      - concatenates and predicts residue, adds back to spatial_band
      - returns (N, 1, 256)
    """

    def __init__(self, in_channels: int, num_3d_filters: int, num_2d_filters: int, num_conv_filters: int, K=24):
        super().__init__()
        self.spectral_conv = SpectralConv1D(in_channel=in_channels, out_channel=num_3d_filters, K=K)
        self.spatial_conv = SpatialConv1D(in_channel=in_channels, out_channel=num_2d_filters)  # 现在处理所有通道
        self.conv_block = ConvBlock1D(
            in_channel=num_2d_filters*3 + num_3d_filters*3,
            out_channel=num_conv_filters,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持多通道输入：使用所有通道进行spatial处理
        spatial_vol = self.spatial_conv(x)  # 现在处理所有通道
        spectral_vol = self.spectral_conv(x)
        fused = torch.cat([spatial_vol, spectral_vol], dim=1)
        residue = self.conv_block(x, fused)
        # 返回时只取第一个通道作为主要输出
        return residue + x[:, :1, :]


def SMCNN(in_channels: int, num_3d_filters: int = 32, num_2d_filters: int = 32, num_conv_filters: int = 64) -> nn.Module:
    return SMCNN1D(
        in_channels=in_channels,
        num_3d_filters=num_3d_filters,
        num_2d_filters=num_2d_filters,
        num_conv_filters=num_conv_filters,
    )


