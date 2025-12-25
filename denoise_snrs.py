import os
import time
import glob
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from DWT_S_UNet_ReLU import  *
from SMCNN import *
from ASCNet import *
from DnCNN import *
from DWTIUNet import *
from UNet import *
from DWTUNet import *
from get_patches import *
from Get_FMD_patches import *


# ------------------ 配置 ------------------ #
Data_Size = 256  # 可根据需要调整
Data_stride = Data_Size // 2
model_name = "denoise_result"
# weights_path = "data/record_result/train_result/256_data_size/UNet/model.pth"
weights_path = "data/result/model.pth"
# 数据路径
sgy_path = "data/data_npy_sgy/part4/2007BP_part4_11shot.sgy"
npy_dir = "data/data_npy_sgy/npy"
fmd = False  # 是否使用FMD分解

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 根据FMD选项设置输入通道数
if fmd == True:
    # FMD分解后noise_patches是(B, 3, S)，拼接origin_patches (B, 1, S)后是(B, 4, S)
    input_channels = 4
else:
    input_channels = 1

# model = DWTUNet(in_channels=input_channels).to(device)
model = DWTIUNet(in_channels=input_channels).to(device)
# model = ASCNet(input_channels, 1, feats=64).to(device)
# model = SMCNN(in_channels=input_channels).to(device)
# model = DWT_S_UNet(in_channels=input_channels).to(device)
# model = DWT_S_UNet_ReLU(in_channels=input_channels).to(device)
# model = UNet(in_channels=input_channels).to(device)


# 输出 Excel：放在权重同级目录，文件名用上一级目录名
excel_dir = os.path.dirname(weights_path)
excel_name = os.path.basename(excel_dir) + ".xlsx"
os.makedirs(excel_dir, exist_ok=True)
excel_path = os.path.join(excel_dir, excel_name)


def load_model(device):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_denoise(model, device, noise_path):
    # 读取噪声数据
    noise_data_original = np.load(noise_path)
    noise_data = noise_data_original


    # 获取原始数据
    origin, nSample, extent_time = get_info_seg(sgy_path)

    # 取 patch
    noise_patches, origin_patches = predict_data_extract_paired_patches(
        noise_data=noise_data,
        clean_data=origin,
        patch_length=Data_Size,
        stride=Data_stride,
    )
    # if fmd == True:
    #     print("使用FMD模式：输入格式为 [noise_patches, clean_patches]")
    #     # 确保origin_patches的形状匹配
    #     if origin_patches.shape[1] != 1:
    #         origin_patches = origin_patches.reshape(-1, 1, Data_Size)
    #     p_data = torch.cat([torch.from_numpy(p_norm), torch.from_numpy(o_norm)], dim=1)
    # else:
    #     p_data = torch.from_numpy(p_norm)

    if fmd == True:
        print("使用FMD分解数据")

        # 根据noise_path生成唯一的文件名，避免不同SNR数据互相覆盖
        noise_file_basename = os.path.splitext(os.path.basename(noise_path))[0]
        file_name = f"FMD_{noise_file_basename}_{Data_Size}.npy"

        if os.path.exists(file_name):
            print(f"加载已有 {file_name}...")
            noise_patches = np.load(file_name, allow_pickle=True)
        else:
            print(f"{file_name} 不存在，正在生成并保存...")
            noise_patches = FMD_data(noise_patches)
            np.save(file_name, noise_patches)




    # 归一化
    range_p = np.max(np.abs(noise_patches))
    range_o = np.max(np.abs(origin_patches))
    range_p = max(range_p, range_o)
    p_norm = noise_patches / range_p
    o_norm = origin_patches / range_p

    # 转换为torch tensor并拼接
    if fmd == True:
        # FMD分解后noise_patches是(B, 3, S)，拼接origin_patches (B, 1, S)后是(B, 4, S)
        p_norm_tensor = torch.from_numpy(p_norm).type(torch.FloatTensor)
        o_norm_tensor = torch.from_numpy(o_norm).type(torch.FloatTensor)
        p_data = torch.cat([o_norm_tensor, p_norm_tensor], dim=1)
    else:
        p_data = torch.from_numpy(p_norm).type(torch.FloatTensor)
    
    p_data = p_data.to(device)

    # 推理
    t0 = time.time()
    with torch.no_grad():
        output = model(p_data)
        output = output.cpu().detach().numpy()
    t1 = time.time()

    # 反归一化
    output = output * range_p

    # 重建
    total_batch, _, _data_size = output.shape
    n_samples, n_traces = noise_data.shape
    segments_per_trace = total_batch // n_traces
    expected_len = (segments_per_trace - 1) * Data_stride + Data_Size
    reconstructed_data = np.zeros((expected_len, n_traces))

    useful_start = Data_stride // 2
    useful_end = Data_Size - useful_start

    for i in range(n_traces):
        start_batch = i * segments_per_trace
        end_batch = (i + 1) * segments_per_trace
        trace_batches = output[start_batch:end_batch, 0, :]

        for j, segment in enumerate(trace_batches):
            seg_start = j * Data_stride
            is_first = (j == 0)
            is_last = (j == segments_per_trace - 1)

            if is_first:
                patch_part = segment[:useful_end]
                write_start = 0
                write_end = useful_end
            elif is_last:
                patch_part = segment[useful_start:]
                write_start = seg_start + useful_start
                write_end = seg_start + Data_Size
                if write_end > expected_len:
                    patch_part = patch_part[: expected_len - write_start]
                    write_end = expected_len
            else:
                patch_part = segment[useful_start:useful_end]
                write_start = seg_start + useful_start
                write_end = seg_start + useful_end

            reconstructed_data[write_start:write_end, i] = patch_part

    reconstructed_data = reconstructed_data[:n_samples, :]

    # 计算指标
    rmse = calculate_rmse(origin, reconstructed_data)
    snr = calculate_snr(origin, reconstructed_data)
    elapsed = t1 - t0

    return elapsed, rmse, snr


def save_plot(records, avg_time, excel_dir):
    plot_records = [r for r in records if r.get("snr_file") != "average"]
    if not plot_records:
        return

    snr_levels = [
        int(r["snr_file"].split("_")[1].split(".")[0]) for r in plot_records
    ]
    rmse_vals = [r["Rmse"] for r in plot_records]
    snr_vals = [r["Snr"] for r in plot_records]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(snr_levels, rmse_vals, marker="o", label="RMSE", color="tab:blue")
    ax1.set_xlabel("Input SNR (dB)")
    ax1.set_ylabel("RMSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(5, 0)  # RMSE 轴 5 -> 0

    ax2 = ax1.twinx()
    ax2.plot(snr_levels, snr_vals, marker="s", label="Output SNR", color="tab:orange")
    ax2.set_ylabel("Output SNR (dB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(-5, 25)  # 输出 SNR 轴 -5 -> 25

    title = f"RMSE & SNR vs Input SNR (avg time {avg_time:.3f}s)"
    ax1.set_title(title)
    fig.tight_layout()

    plot_path = os.path.join(
        excel_dir, os.path.basename(excel_dir) + "_curves.png"
    )
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def main():

    model = load_model(device)

    # 选择 snr_-10 ~ snr_6
    target_snrs = list(range(-10, 7))
    records = []

    for snr_level in target_snrs:
        fname = f"snr_{snr_level}.npy"
        fpath = os.path.join(npy_dir, fname)
        if not os.path.isfile(fpath):
            print(f"skip missing {fname}")
            continue

        elapsed, rmse, snr = run_denoise(model, device, fpath)
        print(f"{fname} -> Time: {elapsed:.4f}, Rmse: {rmse:.4f}, Snr: {snr:.4f}")
        records.append(
            {"snr_file": fname, "Time": elapsed, "Rmse": rmse, "Snr": snr}
        )

    if not records:
        print("No records generated.")
        return

    # 统计平均时间
    avg_time = np.mean([r["Time"] for r in records])
    records.append({"snr_file": "average", "Time": avg_time, "Rmse": None, "Snr": None})

    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False)
    print(f"Saved metrics to {excel_path}")
    print(f"Average inference time: {avg_time:.4f}s")
    save_plot(records, avg_time, excel_dir)


if __name__ == "__main__":
    main()
