import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
import json
import pywt
import torch.fft as fft
from scipy.io import savemat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from Get_FMD_patches import *
from get_patches import *
from DnCNN import *
from UNet import *
from DWTUNet import *
from DWTIUNet import *



from ASCNet import *
from SMCNN import *
from DWT_S_UNet import *
from DWT_S_UNet_ReLU import *
from DWT_DS_UNet import *
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###超参数设置###
Data_Size = 128 # 256,192,128,64
Data_stride = Data_Size//2
print(f"Data_stride: {Data_stride}")
EPOCH = 200   # 遍历数据集次数 100
BATCH_SIZE_s = 100  # 批处理尺寸(batch_size)
BATCH_SIZE_v = 100
LR = 0.0001 # 0.000012
rate = 0.90  # 学习率衰变
iteration = 10 # 每50次衰减
fmd = False
###真实数据读取###


###数据读取###
data_path = 'data/2007BP_synthetic_train.sgy'
data,seismic_time,time_length = get_info_seg(data_path)
#只要part4的数据
data = data[:, -4000:]
np.save('2007BP_synthetic_train', data)
# print(data.shape)
# clean_data = data

###原始数据###
# 固定随机种子
np.random.seed(417)

sigma = 800
noise = np.random.normal(0, sigma / 255.0, data.shape)
original_noise_data = data + noise
original_clean_data = data
original_denoised_data = np.load('data/shots_train_noise/shots_denoise.npy')


print(calculate_snr(original_clean_data,original_noise_data))

noise_data, clean_data = extract_paired_patches(clean_data=original_clean_data, noise_data=original_noise_data,
                                                    patch_length=Data_Size, stride=Data_stride)
denoise_data, _ = extract_paired_patches(clean_data=original_clean_data, noise_data=original_denoised_data,
                                                    patch_length=Data_Size, stride=Data_stride)
denoise_data = noise_data
if fmd == True:
    print("使用FMD分解数据")

    file_name = f"data_{Data_Size}.npy"  # 根据 Data_Size 生成文件名

    if os.path.exists(file_name):
        print(f"加载已有 {file_name}...")
        noise_data = np.load(file_name, allow_pickle=True)
    else:
        print(f"{file_name} 不存在，正在生成并保存...")
        noise_data = FMD_data(noise_data)
        np.save(file_name, noise_data)

x_train, x_val, y_train, y_val, d_train, d_val = train_test_split(
    clean_data, noise_data, denoise_data, test_size=0.2, random_state=417
)
# 截取数据为BATCH倍数
train_len = (len(x_train) // BATCH_SIZE_s) * BATCH_SIZE_s
val_len = (len(x_val) // BATCH_SIZE_v) * BATCH_SIZE_v


x_train = x_train[:train_len]
y_train = y_train[:train_len]
d_train = d_train[:train_len]
x_val = x_val[:val_len]
y_val = y_val[:val_len]
d_val = d_val[:val_len]
print(f"总样本数: {len(noise_data)}")
print(f"训练集: {len(x_train)}, 验证集: {len(x_val)}")
Ls = len(x_train)
Lv = len(x_val)
###数据预处理###
# 归一化
def normalization(data, _range):
    return data / _range



# 训练集归一化
range_x = np.max(np.abs(x_train))
range_y = np.max(np.abs(y_train))
range_s = max(range_x, range_y)
x_t = normalization(x_train, range_s)
y_t = normalization(y_train, range_s)
d_t = normalization(d_train, range_s)

# 验证集归一化
range_x = np.max(np.abs(x_val))
range_y = np.max(np.abs(y_val))
range_v = max(range_x, range_y)
x_v = normalization(x_val, range_v)
y_v = normalization(y_val, range_v)
d_v = normalization(d_val, range_s)

# 训练集格式转换
x1_s = torch.from_numpy(x_t)
x2_s = torch.from_numpy(y_t)
x3_s = torch.from_numpy(d_t)
x1_s = x1_s.type(torch.FloatTensor)
x2_s = x2_s.type(torch.FloatTensor)
x3_s = x3_s.type(torch.FloatTensor)
# 验证集格式转换
x1_v = torch.from_numpy(x_v)
x2_v = torch.from_numpy(y_v)
x3_v = torch.from_numpy(d_v)
x1_v = x1_v.type(torch.FloatTensor)
x2_v = x2_v.type(torch.FloatTensor)
x3_v = x3_v.type(torch.FloatTensor)
if fmd == True :
    x2_s = torch.cat([x1_s, x2_s], dim=1)
    x2_v = torch.cat([x1_v, x2_v], dim=1)

#
# x2_s = torch.cat([x3_s, x1_s], dim=1)
# x2_v = torch.cat([x3_v, x1_v], dim=1)
print(x2_v.shape)
B,C,T= x2_v.shape
input_channels = C
# 模型定义
# net = DWTUNet(in_channels=input_channels).to(device)

# net = DWT_S_UNet_ReLU(in_channels=input_channels).to(device)
# net = UNet(in_channels=input_channels).to(device)
# net = DWTIUNet(in_channels=input_channels).to(device)

net = DWT_S_UNet(in_channels=input_channels).to(device)


# net = DWT_S_UNet_ReLU(in_channels=input_channels).to(device)
# net = SMCNN(in_channels=input_channels).to(device)
# net = ASCNet(input_channels, 1, feats=64).to(device)


# 数据封装打乱顺序
train_data = TensorDataset(x2_s, x1_s, x3_s)
val_data = TensorDataset(x2_v, x1_v, x3_v)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_s, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE_v, shuffle=False, num_workers=0, drop_last=True)

###网络训练###
criterion = nn.MSELoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iteration, gamma=rate)

# 开始训练
# 记录信噪比
snrs_x_n = []
snrs_x_p = []
results_dir = 'data/result'
os.makedirs(results_dir, exist_ok=True)
Losslist_s = []
Losslist_v = []
best_loss = 100

print("Start Training!")
train_start_time = time.time()
for epoch in range(EPOCH):

    loss_s = 0.0
    loss_v = 0.0

    snr_before_list = []
    snr_after_list = []
    snr_val_before_list = []
    snr_val_after_list = []

    start_time = time.time()
    # for i in range(Ls//BATCH_SIZE_s):
    for i, data_s in enumerate(train_loader):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        input_s, target_s, denoise_s = data_s
        input_s = input_s.to(device)
        target_s = target_s.to(device)
        denoise_s = denoise_s.to(device)
        # output_s = net(input_s)*denoise_s
        output_s = net(input_s)
        loss_s0 = criterion(output_s, target_s)
        loss_s0.backward()
        optimizer.step()
        loss_s += loss_s0.item()

        snr_denoise = calculate_snr(target_s, output_s)
        snr_after_list.append(snr_denoise)

    net.eval()
    with torch.no_grad():
        # for j in range(Lv//BATCH_SIZE_v):
        for j, data_v in enumerate(val_loader):
            input_v, target_v, denoise_v = data_v
            input_v = input_v.to(device)
            target_v = target_v.to(device)
            denoise_v = denoise_v.to(device)
            # output_v = net(input_v)*denoise_v
            output_v = net(input_v)

            loss_v0 = criterion(output_v, target_v)
            loss_v += loss_v0.item()
            snr_denoise = calculate_snr(target_v, output_v)
            snr_val_after_list.append(snr_denoise)
            # 保存最优模型
            if loss_v0 < best_loss:
                best_loss = loss_v0
                model_name = f'model.pth'
                torch.save(net.state_dict(), os.path.join(results_dir, model_name))

    epoch_avg_train_loss = loss_s / (Ls//BATCH_SIZE_s)
    epoch_avg_snr_after = sum(snr_after_list) / len(snr_after_list)
    epoch_avg_val_loss = loss_v / (Lv//BATCH_SIZE_s)
    epoch_avg_val_snr_after = sum(snr_val_after_list) / len(snr_val_after_list)
    elapsed_time = time.time() - start_time
    Losslist_v.append(epoch_avg_val_loss)
    Losslist_s.append(epoch_avg_train_loss)

    # 学习率衰减
    scheduler.step()  # 这一行会更新学习率

    # ===== 打印 epoch 信息 =====
    # print(f"Epoch {epoch + 1} Summary:")
    print(f"Train Loss: {epoch_avg_train_loss:.12f}")
    print(f"_Val_ Loss: {epoch_avg_val_loss:.12f}")
    print(f"Train SNR  After: {epoch_avg_snr_after:.4f}")
    print(f"_Val_ SNR  After: {epoch_avg_val_snr_after:.4f}")
    print('Epoch %d, Time: %3.3f' % (epoch + 1, elapsed_time))
    # ===== 写入 txt 文件（仅数字） =====
    with open(os.path.join(results_dir, 'train_loss.txt'), 'a') as f:
        f.write(f"{epoch_avg_train_loss:.10f}\n")
    with open(os.path.join(results_dir, 'val_loss.txt'), 'a') as f:
        f.write(f"{epoch_avg_val_loss:.10f}\n")
    with open(os.path.join(results_dir, 'snr_train_after.txt'), 'a') as f:
        f.write(f"{epoch_avg_snr_after:.4f}\n")
    with open(os.path.join(results_dir, 'snr_val_after.txt'), 'a') as f:
        f.write(f"{epoch_avg_val_snr_after:.4f}\n")


print('finished training')
train_end_time = time.time()
elapsed_seconds = int(train_end_time - train_start_time)
with open(os.path.join(results_dir, 'time.txt'), 'a') as f:
    f.write(f"{elapsed_seconds:.8f}\n")


###绘图###
# 格式转换

print(output_v.shape)
if fmd:
    input_v = denoise_v

input_v = input_v.cpu()
input_v = input_v.detach().numpy()
output_v = output_v.cpu()
output_v = output_v.detach().numpy()
target_v = target_v.cpu()
target_v = target_v.detach().numpy()

# Loss变化
x = range(1, EPOCH + 1)
y_s = Losslist_s
y_v = Losslist_v
plt.semilogy(x, y_s, 'b.-')
plt.semilogy(x, y_v, 'r.-')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.show()
# plt.savefig("accuracy_loss.jpg")

# 去噪前后绘图
col = 85 # 显示第几个数据去噪效果

x = range(0, len(target_v[0, 0, :]))
y1 = target_v[col, 0, :]
# y2 = np.sum(input_v[col, :, :], axis=0)
y3 = output_v[col, 0, :]
plt.plot(x, y1, 'r.-')
# plt.plot(x, y2, 'r.-')
plt.plot(x, y3, 'g.-')
plt.xlabel('Time')
plt.ylabel('Ampulitude')
plt.show()
# plt.savefig()

###SNR###
# 去噪前
origSignal = target_v[:, 0, :]
errorSignal = target_v[:, 0, :] - np.sum(input_v, axis=1)
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues1 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues1)

# 去噪后
origSignal = target_v[:, 0, :]
errorSignal = target_v[:, 0, :] - output_v[:, 0, :]
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues2 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues2)

end = time.time()

