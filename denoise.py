import numpy as np
import torch
from AGC import *

from torch import nn
import torch
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from get_patches import *


from UNet import *
from DWT_S_UNet import *
from DWT_S_UNet_ReLU import *
from ASCNet import *
from SMCNN import *
from DWTUNet import *
###数据和模型载入###
# 数据载入
fmd = False

Data_Size = 256 # 256,192,128,64
Data_stride = Data_Size//2

# 人工合成地震数据
# path = 'data/data_mat_npy_sgy/part4/2007BP_part4_11shot.sgy'
# # path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
# noise_data_original = np.load('data/data_mat_npy_sgy/npy/snr_-2.npy')
# noise_data = noise_data_original


# 现场地震数据
# origin, nSample,extent_time = get_info_seg('data/field_data/Sea_0_1_shot.sgy')

# path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
path = 'data/field_data/real3.mat'
origin = get_mat(path)
noise_data_original = origin
noise_data = noise_data_original
field = True
if field:
    # 根据参数计算extent_time
    # 参数定义
    dt = 1  # 时间采样间隔（秒）
    # 根据实际数据形状获取参数
    nSample, nTrace = origin.shape  # origin形状为 (采样点数, 道数)
    # 计算时间长度
    time_length = nSample * dt  # 时间长度（秒）
    # extent_time格式: [x_min, x_max, y_max, y_min] = [0, 道数, 时间长度(秒), 0]
    extent_time = [0, nTrace, time_length, 0]
    print(f"数据形状: {origin.shape}")
    print(f"采样点数: {nSample}, 道数: {nTrace}, 时间长度: {time_length:.3f}秒")
    print(f"extent_time: {extent_time}")

    # 截取数据到4000个采样点
    # target_samples = nSample
    # if nSample > target_samples:
    #     origin = origin[:target_samples, :]  # 截取前4000个采样点
    #     # 按比例更新extent_time中的时间长度
    #     time_length_ratio = target_samples / nSample
    #     extent_time[2] = extent_time[2] * time_length_ratio
    #     nSample = target_samples
    #     print(f"数据已截取到 {target_samples} 个采样点")
    # # noise_data_original = get_mat('data/field_data/Sea_0_1_shot.mat')







noise_patches, origin_patches = predict_data_extract_paired_patches(noise_data=noise_data,clean_data=origin,patch_length=Data_Size,stride=Data_stride)
if fmd:
    noise_patches = np.load("FMD_snr_-2_256.npy")


# 模型参数载入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(noise_patches.shape)
B, C, T = noise_patches.shape
input_channels = C
if fmd:
    C=C+1
# model = ASCNet(input_channels, 1, feats=64).to(device).to(device)
model = SMCNN(in_channels=C).to(device)
# model = DWT_S_UNet_ReLU(in_channels=C).to(device)
# model = DWT_S_UNet(in_channels=C).to(device)
# model = UNet(C).to(device)
# model = DWTUNet(in_channels=C).to(device)

weights_path = "data/record_result/train_result/LR1e4-rate0.1-40epoch-256_data_size/SMCNN/model.pth"
# weights_path = "data/result/model.pth"
# weights_path = "data/Bayesian/run_alpha_1.000_beta_0.054/model.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

###数据预处理###

# 归一化
def normalization(data, _range):
    return data / _range


# 预测集归一化
range_p = np.max(np.abs(noise_patches))
range_o = np.max(np.abs(origin_patches))
range_p = max(range_p, range_o)

p_norm = normalization(noise_patches, range_p)
o_norm = normalization(origin_patches, range_p)


if fmd == True:
    # FMD分解后noise_patches是(B, 3, S)，拼接origin_patches (B, 1, S)后是(B, 4, S)
    p_norm_tensor = torch.from_numpy(p_norm).type(torch.FloatTensor)
    o_norm_tensor = torch.from_numpy(o_norm).type(torch.FloatTensor)
    p_data = torch.cat([o_norm_tensor, p_norm_tensor], dim=1)
else:
    p_data = torch.from_numpy(p_norm).type(torch.FloatTensor)
o_data = torch.from_numpy(o_norm).type(torch.FloatTensor)

###数据去噪###
# 网络预测
train_start_time = time.time()
model.eval()
with torch.no_grad():
    output = model(p_data.to(device))

    output = output.cpu().detach().numpy()

# 数据重排和反归一化
output = output * range_p

print(output.shape)

# 复用上方的 Data_Size/Data_stride，保持一处配置
data_size = Data_Size
stride = Data_stride
useful_start = stride // 2  # 每个片段中使用的起始位置
useful_end = data_size - useful_start   # 每个片段中使用的结束位置
useful_len = useful_end - useful_start  # 中间有效长度



# 降噪结果重建

total_batch, _, _data_size = output.shape  # 例如 output.shape = (4000, 1, 256)
n_samples, n_traces = noise_data.shape
segments_per_trace = total_batch // n_traces  # 每条震道用了多少个batch，应该是5

# 计算期望长度（用于后续重建）
expected_len = (segments_per_trace - 1) * stride + data_size

reconstructed_data = np.zeros((expected_len, n_traces))



for i in range(n_traces):
    start_batch = i * segments_per_trace
    end_batch = (i + 1) * segments_per_trace
    trace_batches = output[start_batch:end_batch, 0, :]  # shape: (segments_per_trace, 256)

    for j, segment in enumerate(trace_batches):
        seg_start = j * stride
        is_first = (j == 0)
        is_last = (j == segments_per_trace - 1)

        if is_first:
            # 第一个patch，使用前192个点
            patch_part = segment[:useful_end]  # 0～useful_end
            write_start = 0
            write_end = useful_end
        elif is_last:
            # 最后一个patch，使用后192～256的64个点
            patch_part = segment[useful_start:]  # useful_start～data_size
            write_start = seg_start + useful_start
            write_end = seg_start + data_size
            if write_end > expected_len:  # 越界保护
                patch_part = patch_part[:expected_len - write_start]
                write_end = expected_len
        else:
            # 中间patch，使用中间部分64～192
            patch_part = segment[useful_start:useful_end]
            write_start = seg_start + useful_start
            write_end = seg_start + useful_end

        # 写入重建矩阵（直接替换，无需平均）
        reconstructed_data[write_start:write_end, i] = patch_part


# 裁剪成原始长度（n_samples）,gain为agc增益
reconstructed_data = reconstructed_data[:n_samples, :]
remove_noise = noise_data_original-reconstructed_data
noise = noise_data_original - origin
train_end_time = time.time()
print(f"Time: {train_end_time - train_start_time:.4f}")
print(f"Rmse: {calculate_rmse(origin,reconstructed_data):.4f}")
print(f"Snr: {calculate_snr(origin,reconstructed_data):.4f}")

# 如果看不清则增益
# reconstructed_data = gain(reconstructed_data,0.008)
# origin = gain(origin,0.001)
# remove_noise = gain(remove_noise,0.008)
# noise = gain(noise,0.008)
# plot_seismic_npy(noise_data,extent_time,show=True)

# 绘制原始地震数据剖面图（合成数据，合成噪声数据，噪声）
# plot_seismic_npy(origin,extent_time,show=True)
# plot_seismic_npy(noise,extent_time,show=True)
# plot_seismic_npy(noise_data_original,extent_time,show=True)

# 绘制原始地震数据剖面图（降噪数据，去除的噪声）
plot_seismic_npy(reconstructed_data,extent_time,show=True)
plot_seismic_npy(remove_noise,extent_time,show=True)

# 绘制F-K图，频谱波数图
# plot_seismic_f_k_npy(origin,show=True)
# plot_seismic_f_k_npy(noise,show=True)
# plot_seismic_f_k_npy(noise_data_original,show=True)
#
# plot_seismic_f_k_npy(reconstructed_data,show=True)
# plot_seismic_f_k_npy(remove_noise,show=True)

# np.save('denoise_result', reconstructed_data)