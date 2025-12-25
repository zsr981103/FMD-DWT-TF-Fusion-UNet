import numpy as np
import segyio
import os

from get_patches import *

# 合成地震数据
# path = 'data/data_mat_npy_sgy/part4/2007BP_part4_11shot.sgy'
# # path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
# noise_data_original = np.load('data/data_mat_npy_sgy/npy/snr_-2.npy')
# noise_data = noise_data_original
# 现场地震数据

# path = 'data/field_data/nod001.sgy'
# path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
noise_data = get_mat('data/field_data/real3.mat')

origin = noise_data
field = False
if field:
    # 根据参数计算extent_time
    # 参数定义
    dt = 1 # 时间采样间隔（秒）
    # 根据实际数据形状获取参数
    nSample, nTrace = origin.shape  # origin形状为 (采样点数, 道数)
    # 计算时间长度
    time_length = nSample * dt  # 时间长度（秒）
    # extent_time格式: [x_min, x_max, y_max, y_min] = [0, 道数, 时间长度(秒), 0]
    extent_time = [0, nTrace, time_length, 0]
    print(f"数据形状: {origin.shape}")
    print(f"采样点数: {nSample}, 道数: {nTrace}, 时间长度: {time_length:.3f}秒")
    print(f"extent_time: {extent_time}")
# noise_data_original = get_mat('data/field_data/Sea_0_1_shot.mat')
# np.save('Sea_0_1_shot',noise_data_original)
# noise_data = noise_data_original


# 裁剪成原始长度（n_samples）,gain为agc增益
# dataset_s = np.loadtxt('dataset_sample.dat')
reconstructed_data = get_mat("data/record_result/real3/VMD/VMD_2D_denoise.mat")
remove_noise = origin-reconstructed_data
# noise = noise_data_original - origin

# print(f"Rmse: {calculate_rmse(origin,reconstructed_data):.4f}")
# print(f"Snr: {calculate_snr(origin,reconstructed_data):.4f}")

# reconstructed_data = gain(reconstructed_data,0.008)
# origin = gain(origin,0.008)
# remove_noise = gain(remove_noise,0.008)
# noise = gain(noise,0.008)
# plot_seismic_npy(noise_data,extent_time,show=True)

# plot_seismic_npy(origin,extent_time,show=True)
# plot_seismic_npy(noise,extent_time,show=True)
# plot_seismic_npy(noise_data_original,extent_time,show=True)

plot_seismic_npy(reconstructed_data,extent_time,show=True)
plot_seismic_npy(remove_noise,extent_time,show=True)

# plot_seismic_f_k_npy(origin,show=True)
# plot_seismic_f_k_npy(noise,show=True)
# plot_seismic_f_k_npy(noise_data_original,show=True)

# plot_seismic_f_k_npy(reconstructed_data,show=True)
# plot_seismic_f_k_npy(remove_noise,show=True)